"""Backfill Coinbase Exchange BTC-USD 1-minute candles (GET /products/{id}/candles).

Coinbase returns at most 300 candles per request as [time, low, high, open, close, volume].
Used for longer-horizon volatility/regime context; sub-minute features come from tick data.
"""
from __future__ import annotations

import time

import pandas as pd
import pyarrow as pa

from btc5m.config import load_config
from btc5m.http import client, get_json
from btc5m.logging import get_logger
from btc5m.storage.store import hist_path, write_parquet_atomic
from btc5m.timeutil import iso

log = get_logger("coinbase_candles")


def out_file():
    return hist_path("coinbase_candles_1m") / "candles.parquet"


def backfill(days: float, pace_s: float = 0.25) -> pd.DataFrame:
    cfg = load_config()
    prod = cfg.exchanges.coinbase.product
    url = f"{cfg.exchanges.coinbase.rest}/products/{prod}/candles"
    c = client()
    end = int(time.time() // 60) * 60
    start = end - int(days * 86400)
    f = out_file()
    have = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["ts", "low", "high", "open", "close", "volume"])
    have_ts = set(have["ts"].astype(int).tolist())
    rows = []
    t = start
    while t < end:
        t2 = min(t + 300 * 60, end)
        if all(x in have_ts for x in range(t, t2, 60)):
            t = t2
            continue
        j = get_json(c, url, params={"granularity": 60, "start": iso(t), "end": iso(t2)}, min_interval=pace_s)
        for k in j:
            rows.append({"ts": int(k[0]), "low": float(k[1]), "high": float(k[2]), "open": float(k[3]), "close": float(k[4]), "volume": float(k[5])})
        t = t2
    df = pd.concat([have, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["ts"] = df["ts"].astype("int64")
    write_parquet_atomic(pa.Table.from_pandas(df, preserve_index=False), f)
    log.info("coinbase_candles_done", rows=len(df), new=len(rows))
    return df
