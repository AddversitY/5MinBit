"""Backfill Kraken XBT/USD public trades (tick level) via GET /0/public/Trades.

Kraken returns up to 1000 trades per call starting at ``since`` (nanosecond cursor) and a
``last`` cursor for the next page. Rows: [price, volume, time, side(b/s), type(m/l), misc, id].
``side`` is the taker (aggressor) side. Public rate limits are conservative, so we pace
requests (~1/s) and back off on errors.

Output: one Parquet file per UTC day under hist/kraken_trades/date=YYYY-MM-DD/trades.parquet.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pyarrow as pa

from btc5m.config import load_config
from btc5m.http import client, get_json
from btc5m.logging import get_logger
from btc5m.storage.store import hist_path, write_parquet_atomic

log = get_logger("kraken_trades")


def _day_path(day: str):
    return hist_path("kraken_trades") / f"date={day}" / "trades.parquet"


def backfill(days: float, pace_s: float = 1.05) -> None:
    cfg = load_config()
    url = f"{cfg.exchanges.kraken.rest}/0/public/Trades"
    pair = cfg.exchanges.kraken.rest_pair
    c = client()
    now = time.time()
    day0 = int((now - days * 86400) // 86400) * 86400
    today = int(now // 86400) * 86400
    for d in range(today - 86400, day0 - 1, -86400):  # complete days only, most recent first
        day = datetime.fromtimestamp(d, tz=timezone.utc).strftime("%Y-%m-%d")
        if _day_path(day).exists():
            continue
        since = int(d * 1e9)
        rows = {"ts": [], "price": [], "size": [], "taker_side": [], "ord_type": [], "trade_id": []}
        n_calls = 0
        while True:
            j = get_json(c, url, params={"pair": pair, "since": since, "count": 1000})
            n_calls += 1
            if j.get("error"):
                log.warning("kraken_error", err=j["error"])
                time.sleep(5)
                continue
            res = j["result"]
            key = next(k for k in res if k != "last")
            trades = res[key]
            stop = False
            for t in trades:
                ts = float(t[2])
                if ts >= d + 86400:
                    stop = True
                    break
                rows["ts"].append(ts)
                rows["price"].append(float(t[0]))
                rows["size"].append(float(t[1]))
                rows["taker_side"].append(1 if t[3] == "b" else -1)
                rows["ord_type"].append(t[4])
                rows["trade_id"].append(int(t[6]) if len(t) > 6 else -1)
            new_since = int(res["last"])
            if stop or not trades or new_since == since:
                break
            since = new_since
            time.sleep(pace_s)
        tbl = pa.table(rows)
        write_parquet_atomic(tbl, _day_path(day))
        log.info("kraken_day_done", day=day, trades=len(rows["ts"]), calls=n_calls)
