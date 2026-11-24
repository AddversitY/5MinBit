"""Append-only, hash-chained paper-trading ledger.

Each line is a JSON record with ``seq``, ``prev_hash`` and ``hash`` = SHA-256 of the canonical
JSON of the record without ``hash``. Records are never rewritten: settlements are separate
records that reference a prediction by ``ref_hash``. ``verify()`` recomputes the whole chain,
so any edit, deletion or reordering of a past line is detected.

Integrity rules enforced at write time:
* a prediction must be written before its round ends (``ts < round_end``);
* a settlement can only reference an existing prediction and is accepted once per prediction;
* a settlement must be written after the round ended.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

from btc5m.config import load_config

GENESIS = "0" * 64


def _canon(rec: dict) -> bytes:
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), default=float).encode()


def record_hash(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "hash"}
    return hashlib.sha256(_canon(body)).hexdigest()


class LedgerError(Exception):
    pass


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = 0
        self._head = GENESIS
        self._preds: dict[str, dict] = {}  # hash -> prediction (for settlement checks)
        self._settled: set[str] = set()
        self._keys: set[tuple] = set()  # (round, checkpoint, model_version) to avoid duplicates
        if self.path.exists():
            self._load()

    @classmethod
    def default(cls) -> "Ledger":
        return cls(load_config().path("ledger"))

    def _load(self) -> None:
        with open(self.path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                self._seq = rec["seq"]
                self._head = rec["hash"]
                if rec["type"] == "prediction":
                    self._preds[rec["hash"]] = rec
                    self._keys.add((rec["round_start"], rec["checkpoint"], rec["model_version"]))
                elif rec["type"] == "settlement":
                    self._settled.add(rec["ref_hash"])

    def _append(self, rec: dict) -> dict:
        with self._lock:
            rec = dict(rec)
            rec["seq"] = self._seq + 1
            rec["prev_hash"] = self._head
            rec["written_at"] = time.time()
            rec["hash"] = record_hash(rec)
            line = json.dumps(rec, sort_keys=True, separators=(",", ":"), default=float)
            with open(self.path, "a") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._seq = rec["seq"]
            self._head = rec["hash"]
            return rec

    def add_prediction(self, rec: dict) -> dict | None:
        key = (rec["round_start"], rec["checkpoint"], rec["model_version"])
        if key in self._keys:
            return None
        if time.time() >= rec["round_start"] + 300:
            raise LedgerError("prediction written after round end")
        rec = {"type": "prediction", **rec}
        out = self._append(rec)
        self._preds[out["hash"]] = out
        self._keys.add(key)
        return out

    def add_settlement(self, ref_hash: str, rec: dict) -> dict | None:
        if ref_hash not in self._preds:
            raise LedgerError("unknown prediction")
        if ref_hash in self._settled:
            return None
        pred = self._preds[ref_hash]
        if time.time() < pred["round_start"] + 300:
            raise LedgerError("settlement before round end")
        out = self._append({"type": "settlement", "ref_hash": ref_hash, "round_start": pred["round_start"], **rec})
        self._settled.add(ref_hash)
        return out

    def pending(self) -> list[dict]:
        return [p for h, p in self._preds.items() if h not in self._settled]

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(x) for x in f if x.strip()]

    @property
    def head(self) -> str:
        return self._head

    def verify(self) -> tuple[bool, int, str]:
        prev, n = GENESIS, 0
        if not self.path.exists():
            return True, 0, "empty"
        with open(self.path) as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("prev_hash") != prev:
                    return False, n, f"broken chain at line {i}"
                if record_hash(rec) != rec.get("hash"):
                    return False, n, f"hash mismatch at line {i} (record modified)"
                if rec.get("seq") != n + 1:
                    return False, n, f"sequence gap at line {i}"
                if rec["type"] == "prediction" and rec["written_at"] >= rec["round_start"] + 300:
                    return False, n, f"prediction after round end at line {i}"
                prev = rec["hash"]
                n += 1
        return True, n, f"head={prev[:16]}"
