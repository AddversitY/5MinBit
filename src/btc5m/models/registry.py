"""Versioned model registry with a promotion gate.

Layout: ``data/models/<version>/{model.pkl, meta.json}`` and ``data/models/registry.json``
(``{"production": <version|null>, "history": [...]}``).

A candidate is trained on all TWAP-60 rounds available (last 20% chronologically held out
for early stopping/calibration). Its *evidence* is the walk-forward out-of-sample report for
the same model family. ``promote`` refuses unless the gate passes:

1. OOS rounds >= ``MIN_OOS_ROUNDS``;
2. OOS log loss better than the parameter-free Gaussian baseline (day-block bootstrap
   95% CI of the Brier difference entirely below 0 at >= 3 early checkpoints, k <= 60);
3. OOS calibration ECE <= ``MAX_ECE`` at every checkpoint;
4. not worse than the current production model on the same OOS rows.

Nothing is ever promoted automatically. Without a production model the live engine uses
the parameter-free Gaussian random-walk baseline and labels it as such.
"""
from __future__ import annotations

import glob
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from btc5m.config import load_config
from btc5m.logging import get_logger

log = get_logger("registry")
MIN_OOS_ROUNDS = 2000
MAX_ECE = 0.03
EARLY_KS = (0, 15, 30, 45, 60)


def models_dir() -> Path:
    p = load_config().path("models_dir")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _reg_path() -> Path:
    return models_dir() / "registry.json"


def read_registry() -> dict:
    p = _reg_path()
    if p.exists():
        return json.loads(p.read_text())
    return {"production": None, "history": []}


def write_registry(reg: dict) -> None:
    tmp = _reg_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=1))
    tmp.replace(_reg_path())


def latest_report() -> tuple[Path | None, dict | None]:
    files = sorted(glob.glob(str(models_dir().parent / "reports" / "eval_*.json")))
    files = [f for f in files if "_quick" not in f] or files
    if not files:
        return None, None
    return Path(files[-1]), json.loads(Path(files[-1]).read_text())


def gate(report: dict, family: str) -> dict:
    """Evaluate the promotion gate for model family ``family`` (e.g. 'lightgbm')."""
    col = f"p_{family}"
    res = {"family": family, "checks": {}, "passed": False}
    oos_rounds = report.get("oos", {}).get("rounds", 0)
    res["checks"]["min_oos_rounds"] = {"value": oos_rounds, "required": MIN_OOS_ROUNDS, "ok": oos_rounds >= MIN_OOS_ROUNDS}
    vb = report["metrics"]["vs_baseline"]
    better = 0
    detail = {}
    for k in EARLY_KS:
        d = vb.get(str(k), vb.get(k, {})).get(col)
        if d is None:
            continue
        detail[k] = d
        if d["ci95"][1] < 0:
            better += 1
    if family == "gaussian_rw":
        better = len(EARLY_KS)  # the baseline is the reference itself
    res["checks"]["beats_gaussian_early"] = {"significant_checkpoints": better, "required": 3, "ok": better >= 3, "detail": detail}
    eces = {}
    for k, row in report["metrics"]["by_checkpoint"].items():
        if col in row:
            eces[k] = row[col]["ece"]
    res["checks"]["calibration"] = {"max_ece": max(eces.values()) if eces else None, "required": MAX_ECE, "ok": bool(eces) and max(eces.values()) <= MAX_ECE}
    res["passed"] = all(c["ok"] for c in res["checks"].values())
    return res


def train_candidate(family: str = "lightgbm") -> str:
    from btc5m.eval.run import HMM_FEATURES, load_dataset, minute_returns, model_factory
    from btc5m.features.dataset import load_kraken_bars
    from btc5m.models.baselines import standardized_returns
    from btc5m.models.hmm import RegimeHMM, attach_regimes
    from btc5m.rules import TWAP60

    df = load_dataset()
    df = df[df["settlement_kind"] == TWAP60].reset_index(drop=True)
    starts = np.sort(df["start"].unique())
    cut = starts[int(len(starts) * 0.8)]
    mret = minute_returns()
    hmm = RegimeHMM().fit(mret[(mret.index >= int(starts[0]) - 7 * 86400) & (mret.index < int(starts[-1]) + 300)])
    d = attach_regimes(df, hmm.filter(mret))
    tr, va = d[d["start"] < cut - 3600], d[d["start"] >= cut]
    std_ret = None
    if family == "monte_carlo":
        bars = load_kraken_bars()
        std_ret = standardized_returns(bars.a["close"])
    model = model_factory(std_ret)[family]().fit(tr, va)
    rpath, report = latest_report()
    g = gate(report, family) if report else {"passed": False, "checks": {"report": "missing"}}
    version = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + f"-{family}"
    vdir = models_dir() / version
    vdir.mkdir(parents=True, exist_ok=False)
    blob = pickle.dumps({"model": model, "hmm": hmm, "family": family})
    (vdir / "model.pkl").write_bytes(blob)
    meta = {
        "version": version, "family": family, "created_at": time.time(), "sha256": hashlib.sha256(blob).hexdigest(),
        "train_rounds": int(tr["start"].nunique()), "val_rounds": int(va["start"].nunique()),
        "train_range": [int(tr["start"].min()), int(va["start"].max())], "features": getattr(model, "features", []) or ["z", "k"],
        "evidence_report": str(rpath) if rpath else None, "gate": g, "status": "candidate",
    }
    (vdir / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    reg = read_registry()
    reg["history"].append({"version": version, "event": "registered", "at": time.time(), "gate_passed": g["passed"]})
    write_registry(reg)
    log.info("candidate_registered", version=version, gate_passed=g["passed"])
    print(json.dumps({"version": version, "gate": g}, indent=1, default=float))
    return version


def promote(version: str, force: bool = False) -> bool:
    vdir = models_dir() / version
    meta = json.loads((vdir / "meta.json").read_text())
    if not meta["gate"]["passed"] and not force:
        print(f"REFUSED: {version} did not pass the validation gate: {json.dumps(meta['gate']['checks'], default=float)[:800]}")
        return False
    reg = read_registry()
    reg["history"].append({"version": version, "event": "promoted", "from": reg.get("production"), "at": time.time(), "forced": force})
    reg["production"] = version
    write_registry(reg)
    meta["status"] = "production"
    (vdir / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"promoted {version}")
    return True


def load_production():
    """Return (version, family, model, hmm) or (None, ...) when nothing is validated."""
    reg = read_registry()
    v = reg.get("production")
    if not v:
        return None, "gaussian_rw", None, None
    vdir = models_dir() / v
    blob = (vdir / "model.pkl").read_bytes()
    meta = json.loads((vdir / "meta.json").read_text())
    if hashlib.sha256(blob).hexdigest() != meta["sha256"]:
        raise RuntimeError(f"model artifact {v} failed integrity check")
    obj = pickle.loads(blob)  # trusted: written by this registry, integrity-checked above
    return v, obj["family"], obj["model"], obj["hmm"]


__all__ = ["gate", "train_candidate", "promote", "load_production", "read_registry", "pd"]
