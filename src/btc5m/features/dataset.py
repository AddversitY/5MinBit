"""Build the historical point-in-time checkpoint dataset.

One row per (round, checkpoint k). Inputs:

* rounds (Gamma): official price_to_beat, final_price, outcome, settlement kind, fees
* Kraken XBT/USD tick trades -> 1 s bars (primary exchange history)
* Coinbase 1-minute candles -> longer-horizon volatility context
* Polymarket taker trades (optional) -> market-implied probability and executable prices

Reference-price proxy (Chainlink is not available historically without a paid key):
``ref[s] = kraken_close[s] + basis`` with ``basis = price_to_beat - mean(kraken_close[start-60, start))``
for TWAP rounds (``price_to_beat - kraken_close[start-1]`` for spot rounds). The basis uses only
the official price to beat (published at the round start) and pre-start exchange data, so it
introduces no look-ahead. Its quality is measured per round by reconstructing the official
final price (``proxy_final_err``) and outcome (``proxy_outcome_agree``).
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from btc5m.config import load_config
from btc5m.features.bars import BarFrame
from btc5m.features.compute import LOOKBACK, Snapshot, compute_features
from btc5m.features.market_implied import fill_price, implied_at, prepare_round_trades
from btc5m.history.gamma_rounds import load_rounds
from btc5m.logging import get_logger
from btc5m.rules import SPOT, FeeSchedule
from btc5m.storage.store import data_dir

log = get_logger("dataset")
DATA_LAG_S = 1  # exchange data for second s is assumed usable from s+1+DATA_LAG_S
# Data API trade timestamps are block times; measured against the live WebSocket match
# timestamps they lag by a median 2.2 s (p05 1.2 s; RESEARCH.md). Market lookups are shifted by
# this offset so the historical market price at t reflects trades matched before ~t. Rounding
# down to 2 s errs in the market's favour (slight look-ahead for the market, never the model).
PM_TS_OFFSET_S = 2.0

_G: dict = {}  # globals shared with worker processes (fork copy-on-write)


def load_kraken_bars(days: list[str] | None = None) -> BarFrame:
    files = sorted(glob.glob(str(data_dir() / "hist" / "kraken_trades" / "date=*" / "trades.parquet")))
    if days:
        files = [f for f in files if any(d in f for d in days)]
    if not files:
        raise RuntimeError("no Kraken trades on disk; run `btc5m backfill-kraken`")
    df = pd.concat([pd.read_parquet(f, columns=["ts", "price", "size", "taker_side"]) for f in files], ignore_index=True)
    t0 = int(df["ts"].min() // 86400 * 86400)
    t1 = int(df["ts"].max() // 86400 * 86400 + 86400)
    bf = BarFrame.from_trades(df["ts"].to_numpy(), df["price"].to_numpy(), df["size"].to_numpy(), df["taker_side"].to_numpy(), t0, t1)
    # mark which days are covered so rounds on missing days are skipped
    covered = np.zeros(bf.n // 86400 + 1, dtype=bool)
    for f in files:
        d = f.split("date=")[1][:10]
        di = (int(pd.Timestamp(d, tz="UTC").timestamp()) - t0) // 86400
        if 0 <= di < len(covered):
            covered[di] = True
    bf.day_covered = covered
    return bf


def minute_vol_context(candles: pd.DataFrame) -> pd.DataFrame:
    """Per-minute rv_1h, rv_4h (USD/sqrt(s)) and 7-day percentile of rv_1h, using minutes
    that have fully closed. Indexed by minute-end timestamp (a value is usable at t >= index)."""
    c = candles.sort_values("ts").set_index("ts")["close"]
    full = c.reindex(np.arange(int(c.index.min()), int(c.index.max()) + 60, 60)).ffill()
    lr = np.log(full).diff()
    sq = lr.pow(2)
    out = pd.DataFrame(index=full.index + 60)  # candle at ts covers [ts, ts+60): known at ts+60
    out["rv_1h"] = np.sqrt(sq.rolling(60, min_periods=45).sum().to_numpy() / 3600.0) * full.to_numpy()
    out["rv_4h"] = np.sqrt(sq.rolling(240, min_periods=180).sum().to_numpy() / 14400.0) * full.to_numpy()
    rel = out["rv_1h"] / full.to_numpy()
    out["vol_pct_7d"] = rel.rolling(10080, min_periods=1440).rank(pct=True).to_numpy()
    return out


def _vol_ctx_at(vc: pd.DataFrame, t: int) -> tuple[float, float, float]:
    i = vc.index.searchsorted(t, side="right") - 1
    if i < 0:
        return (np.nan, np.nan, np.nan)
    row = vc.iloc[i]
    if t - vc.index[i] > 600:  # stale context
        return (np.nan, np.nan, np.nan)
    return (float(row["rv_1h"]), float(row["rv_4h"]), float(row["vol_pct_7d"]))


def _round_rows(r: dict) -> list[dict]:
    bf: BarFrame = _G["bars"]
    vc: pd.DataFrame = _G["volctx"]
    checkpoints = _G["checkpoints"]
    pm: dict = _G["pm"]
    start, end = int(r["start"]), int(r["end"])
    lo_needed = start - LOOKBACK - 120
    if lo_needed < bf.t0 or end + 1 > bf.t1:
        return []
    days = [(lo_needed - bf.t0) // 86400, (end - bf.t0) // 86400]
    if not all(bf.day_covered[d] for d in days):
        return []
    close = bf.a["close"]
    o = bf.t0
    ptb = float(r["price_to_beat"])
    kind = r["settlement_kind"]
    L = int(r["twap_lookback_s"] or 0)
    if kind == SPOT or L <= 0:
        anchor = close[start - 1 - o]
        basis = ptb - anchor
        proxy_final = close[end - 1 - o] + basis
    else:
        anchor = float(np.nanmean(close[start - L - o : start - o]))
        basis = ptb - anchor
        proxy_final = float(np.nanmean(close[end - L - o : end - o])) + basis
    if not np.isfinite(basis):
        return []
    # liquidity guard: require trades in most minutes of the lookback
    ntr = bf.a["ntr"][start - LOOKBACK - o : start - o]
    if np.count_nonzero(ntr) < 0.2 * LOOKBACK:
        return []
    y = int(r["outcome_up"])
    final = r.get("final_price")
    common = {
        "slug": r["slug"], "start": start, "y": y, "settlement_kind": kind, "twap_lookback_s": L, "price_to_beat": ptb,
        "final_price": final if final is not None and np.isfinite(final) else np.nan,
        "basis": basis, "proxy_final_err": (proxy_final - final) if final is not None and np.isfinite(final) else np.nan,
        "proxy_outcome_agree": float(int(proxy_final >= ptb) == y),
        "fee_rate": r.get("fee_rate"), "fee_exponent": r.get("fee_exponent"), "volume": r.get("volume"),
    }
    trades = pm.get(r["condition_id"])
    rt = prepare_round_trades(trades, r["token_up"]) if trades is not None and len(trades) else None
    rows = []
    for k in checkpoints:
        t = start + int(k)
        td = t - DATA_LAG_S  # data horizon
        ref = close[td - LOOKBACK - o : td - o] + basis
        bars = bf.window(td, LOOKBACK)
        rv1h, rv4h, vpct = _vol_ctx_at(vc, td)
        snap = Snapshot(t=t, start=start, end=end, ptb=ptb, kind=kind, lookback_s=L, ref=ref, bars=bars, rv_1h=rv1h, rv_4h=rv4h, vol_pct_7d=vpct)
        f = compute_features(snap)
        f.update(common)
        if rt is not None:
            mi = implied_at(rt, t + PM_TS_OFFSET_S)
            f.update(mi)
            for lat in (2.0, 5.0):
                f[f"fill_up_l{int(lat)}"] = fill_price(rt, t + PM_TS_OFFSET_S, True, latency=lat)
                f[f"fill_dn_l{int(lat)}"] = fill_price(rt, t + PM_TS_OFFSET_S, False, latency=lat)
        rows.append(f)
    return rows


def _init_worker(g):
    _G.update(g)


def load_pm_trades(token_up: dict[str, str] | None = None) -> dict:
    """Polymarket taker trades per condition id, reduced to the columns features need.

    Read one batch file at a time (each holds complete markets) so peak memory stays small:
    ~20M rows of text ids would otherwise need ~9 GB. Exact duplicate rows (same transaction,
    price, size, time) are dropped. With ``token_up`` ({condition_id: Up token id}) the token
    and side strings are replaced by booleans ``is_up`` and ``buy``.
    """
    files = sorted(glob.glob(str(data_dir() / "hist" / "pm_trades" / "batch-*.parquet")))
    out: dict = {}
    cols = ["condition_id", "token_id", "side", "price", "size", "ts", "tx"]
    for f in files:
        t = pd.read_parquet(f, columns=cols).drop_duplicates().drop(columns="tx")
        for cid, g in t.groupby("condition_id", sort=False):
            if token_up is None:
                out[cid] = g.drop(columns="condition_id").reset_index(drop=True)
                continue
            up = token_up.get(cid)
            if up is None:
                continue
            out[cid] = pd.DataFrame({
                "is_up": g["token_id"].to_numpy() == up,
                "buy": g["side"].to_numpy() == "BUY",
                "price": g["price"].to_numpy(),
                "size": g["size"].to_numpy(),
                "ts": g["ts"].to_numpy(),
            })
    return out


def code_version() -> str:
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for f in sorted(list(here.glob("*.py")) + [here.parent / "rules.py"]):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def build_dataset(source: str = "kraken", workers: int | None = None, out: Path | None = None) -> Path:
    cfg = load_config()
    t_start = time.time()
    rounds = load_rounds()
    rounds = rounds[rounds["resolved"].fillna(False) & rounds["price_to_beat"].notna() & rounds["outcome_up"].notna()].copy()
    bars = load_kraken_bars()
    candles = pd.read_parquet(data_dir() / "hist" / "coinbase_candles_1m" / "candles.parquet")
    volctx = minute_vol_context(candles)
    pm = load_pm_trades(dict(zip(rounds["condition_id"], rounds["token_up"])))
    g = {"bars": bars, "volctx": volctx, "checkpoints": cfg.rounds["checkpoints_s"], "pm": pm}
    recs = rounds.to_dict("records")
    log.info("dataset_build_start", rounds=len(recs), pm_markets=len(pm), bars_days=int(bars.day_covered.sum()))
    workers = workers or max(1, (os.cpu_count() or 2))
    rows: list[dict] = []
    _G.update(g)
    if workers == 1:
        for r in recs:
            rows.extend(_round_rows(r))
    else:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(g,)) as ex:
            for res in ex.map(_round_rows, recs, chunksize=200):
                rows.extend(res)
    df = pd.DataFrame(rows)
    out = out or (data_dir() / "datasets" / f"checkpoints_{source}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    meta = {
        "built_at": time.time(), "rows": len(df), "rounds": int(df["start"].nunique()) if len(df) else 0,
        "code_version": code_version(), "data_lag_s": DATA_LAG_S, "pm_ts_offset_s": PM_TS_OFFSET_S, "lookback_s": LOOKBACK, "source": source,
        "first_round": int(df["start"].min()) if len(df) else None, "last_round": int(df["start"].max()) if len(df) else None,
        "pm_markets": len(pm), "elapsed_s": round(time.time() - t_start, 1),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    log.info("dataset_built", **meta)
    return out


__all__ = ["build_dataset", "load_kraken_bars", "minute_vol_context", "FeeSchedule", "math"]
