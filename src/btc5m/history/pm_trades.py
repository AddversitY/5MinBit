"""Backfill Polymarket taker trades for historical rounds via Data API v2 ``GET /v2/trades``.

``taker_only`` defaults to true, so every fill appears once on its taker side, which gives the
aggressor direction. Up to 20 condition ids per request; pages follow ``next_cursor``.
These trades are the only public, historical, sub-minute record of the market-implied
probability (the CLOB has no historical order-book endpoint and 1-minute price history is
retained only ~7 days). Taker BUY prints approximate the ask and taker SELL prints the bid,
which lets us reconstruct an approximate historical bid/ask (see features/market_implied.py).

Wallet identifiers are dropped: they are not needed for this research.
"""
from __future__ import annotations

import time

import pandas as pd
import pyarrow as pa

from btc5m.config import load_config
from btc5m.history.gamma_rounds import load_rounds
from btc5m.http import client, get_json
from btc5m.logging import get_logger
from btc5m.storage.store import hist_path, write_parquet_atomic

log = get_logger("pm_trades")
BATCH = 20


def _batch_file(first_start: int):
    return hist_path("pm_trades") / f"batch-{first_start}.parquet"


def fetch_batch(c, cids: list[str]) -> list[dict]:
    cfg = load_config()
    url = f"{cfg.polymarket.data_api_url}/v2/trades"
    rows, cursor = [], None
    while True:
        params = {"condition": ",".join(cids)}
        if cursor:
            params["cursor"] = cursor
        else:
            params["limit"] = 1000
        j = get_json(c, url, params=params)
        for d in j.get("data") or []:
            rows.append(
                {
                    "condition_id": d["condition_id"],
                    "token_id": d["token_id"],
                    "outcome_index": int(d.get("outcome_index", 999)),
                    "outcome": d.get("outcome"),
                    "side": d.get("side"),
                    "price": float(d["price"]),
                    "size": float(d["size"]),
                    "ts": int(d["timestamp"]),
                    "tx": d.get("transaction_hash"),
                }
            )
        pg = j.get("pagination") or {}
        cursor = pg.get("next_cursor")
        if not cursor or not pg.get("has_more", True):
            break
    return rows


def backfill(days: float) -> None:
    rounds = load_rounds()
    if not len(rounds):
        raise RuntimeError("run the rounds backfill first")
    cutoff = time.time() - days * 86400
    r = rounds[(rounds["start"] >= cutoff) & rounds["resolved"].fillna(False) & rounds["condition_id"].notna()]
    r = r.sort_values("start", ascending=False)
    c = client(timeout=60)
    starts = r["start"].tolist()
    cids = r["condition_id"].tolist()
    t0 = time.time()
    for i in range(0, len(starts), BATCH):
        f = _batch_file(int(min(starts[i : i + BATCH])))
        if f.exists():
            continue
        rows = fetch_batch(c, cids[i : i + BATCH])
        df = pd.DataFrame(rows)
        if not len(df):
            df = pd.DataFrame({k: pd.Series(dtype=t) for k, t in [("condition_id", "str"), ("token_id", "str"), ("outcome_index", "int64"), ("outcome", "str"), ("side", "str"), ("price", "float64"), ("size", "float64"), ("ts", "int64"), ("tx", "str")]})
        write_parquet_atomic(pa.Table.from_pandas(df, preserve_index=False), f)
        log.info("pm_trades_batch", i=i // BATCH, of=(len(starts) + BATCH - 1) // BATCH, trades=len(df), elapsed=round(time.time() - t0))
