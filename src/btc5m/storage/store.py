"""Parquet + DuckDB storage.

Layout (all under ``paths.data_dir``)::

    raw/live/<stream>/date=YYYY-MM-DD/<HHMMSS>-<pid>-<n>.parquet   live collector output
    hist/<dataset>/...                                              historical backfills
    ledger/predictions.jsonl                                        append-only paper ledger

Files are immutable once written (write to temp name, then atomic rename), so readers never
see partial files. DuckDB is used as a query engine over the Parquet globs.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from btc5m.config import load_config


def data_dir() -> Path:
    p = load_config().path("data_dir")
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_parquet_atomic(table: pa.Table, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
    return path


class ParquetSink:
    """Thread-safe buffered writer for one live stream; flushes every ``flush_seconds``."""

    def __init__(self, stream: str, flush_seconds: float = 10.0, root: Path | None = None):
        self.stream = stream
        self.flush_seconds = flush_seconds
        self.root = (root or data_dir()) / "raw" / "live" / stream
        self._rows: list[dict] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self._n = 0
        self.rows_written = 0

    def add(self, row: dict) -> None:
        with self._lock:
            self._rows.append(row)
        if time.time() - self._last_flush >= self.flush_seconds:
            self.flush()

    def extend(self, rows: Iterable[dict]) -> None:
        with self._lock:
            self._rows.extend(rows)
        if time.time() - self._last_flush >= self.flush_seconds:
            self.flush()

    def flush(self) -> Path | None:
        with self._lock:
            rows, self._rows = self._rows, []
            self._last_flush = time.time()
        if not rows:
            return None
        now = datetime.now(timezone.utc)
        self._n += 1
        path = self.root / f"date={now:%Y-%m-%d}" / f"{now:%H%M%S}-{os.getpid()}-{self._n}.parquet"
        table = pa.Table.from_pylist(rows)
        write_parquet_atomic(table, path)
        self.rows_written += len(rows)
        return path


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    cfg = load_config()
    db = cfg.path("duckdb")
    db.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db), read_only=read_only)


def live_glob(stream: str) -> str:
    return str(data_dir() / "raw" / "live" / stream / "date=*" / "*.parquet")


def hist_path(dataset: str) -> Path:
    p = data_dir() / "hist" / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p


def query_df(sql: str, params: list | None = None):
    con = duckdb.connect()
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()
