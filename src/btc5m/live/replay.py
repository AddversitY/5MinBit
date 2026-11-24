"""Replay recorded live streams through the exact live engine code path.

Recorded Parquet rows (``data/raw/live/*``) are pushed into a fresh ``MarketState`` in
receive-time order, and ``Engine.snapshot(T)`` is evaluated at each checkpoint T. Only rows
with ``rx <= T`` are visible at T, so the replay reproduces what the live engine could have
known. The result is a dataset of live-only features (order books, Chainlink-vs-exchange gap,
futures basis, liquidations, Polymarket book), labelled with official outcomes. It can be
evaluated once enough live rounds have been captured (see ROADMAP Phase 7).

Usage: ``python -m btc5m.live.replay`` -> data/datasets/checkpoints_live.parquet
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from btc5m.config import load_config
from btc5m.history.gamma_rounds import load_rounds
from btc5m.live.engine import Engine
from btc5m.live.state import Book, MarketState, PolyRound
from btc5m.logging import get_logger
from btc5m.storage.store import data_dir, live_glob

log = get_logger("replay")


def _read(stream: str, cols: str = "*") -> pd.DataFrame:
    try:
        return duckdb.connect().execute(f"select {cols} from read_parquet('{live_glob(stream)}', union_by_name=true)").df()
    except Exception:
        return pd.DataFrame()


def _book(row) -> Book:
    bids = [(row[f"bp{i}"], row[f"bs{i}"]) for i in range(10) if pd.notna(row.get(f"bp{i}"))]
    asks = [(row[f"ap{i}"], row[f"as{i}"]) for i in range(10) if pd.notna(row.get(f"ap{i}"))]
    return Book(bids=bids, asks=asks, ts=row["ts"], rx=row["rx"])


def build_live_dataset(out_path=None, every_s: int | None = None) -> pd.DataFrame:
    cfg = load_config()
    cps = [int(c) for c in cfg.rounds["checkpoints_s"]]
    ev = []  # (rx, kind, row)
    for stream, kind in (("chainlink", "cl"), ("kraken_trades", "kr"), ("coinbase_trades", "cbt"), ("coinbase_book", "cbb"), ("kraken_book", "krb"),
                         ("pm_book", "pmb"), ("kraken_fut_ticker", "kft"), ("kraken_fut_trades", "kfx"), ("pm_rounds", "pmr")):
        df = _read(stream)
        if df.empty:
            continue
        if kind == "pmr":
            df = df[df["slug"].notna()] if "slug" in df else df.iloc[0:0]
            df["rx"] = df["fetched_at"]
        df = df.sort_values("rx")
        ev.append(df.assign(_kind=kind))
    if not ev:
        raise RuntimeError("no live recordings found")
    events = pd.concat(ev, ignore_index=True).sort_values("rx", kind="stable").reset_index(drop=True)
    import time as _time

    from btc5m.history.gamma_rounds import backfill as _rounds_backfill

    rounds = load_rounds()
    if not len(rounds) or rounds["start"].max() < events["rx"].max() - 900:
        rounds = _rounds_backfill(days=(_time.time() - float(events["rx"].min())) / 86400 + 0.05)
    res = rounds.set_index("start")
    t_lo = int(events["rx"].min()) + 900
    t_hi = int(events["rx"].max())
    # k=0 is defined as the instant the price to beat is observable (~1.6 s after the open)
    checkpoints = sorted(s + c + (2 if c == 0 else 0) for s in range((t_lo // 300 + 1) * 300, t_hi - 300, 300) for c in cps)
    if every_s:  # dense evaluation grid (for seconds-scale efficiency studies)
        checkpoints = list(range((t_lo // 300 + 1) * 300 + 2, t_hi - 2, every_s))
    st = MarketState()
    eng = Engine(st, record=False, refresh_candles=False)
    from btc5m.live.candles import CandleCache

    from btc5m.history.coinbase_candles import backfill as _candles_backfill

    _candles_backfill(days=(_time.time() - float(events["rx"].min())) / 86400 + 1)
    cc = CandleCache(keep_days=400)
    f = data_dir() / "hist" / "coinbase_candles_1m" / "candles.parquet"
    if f.exists():
        from btc5m.features.dataset import minute_vol_context

        cc.df = pd.read_parquet(f)
        cc.ctx = minute_vol_context(cc.df)
    eng.candles = cc
    rows, i, n = [], 0, len(events)
    rx = events["rx"].to_numpy()
    for T in checkpoints:
        while i < n and rx[i] <= T:
            r = events.iloc[i]
            k = r["_kind"]
            if k == "cl":
                series = {"chainlink_spot": st.chainlink_spot, "chainlink_twap60": st.chainlink_twap, "binance_relay": st.binance_relay}.get(r["kind"])
                if series is not None:
                    series.add(r["src_ts"], r["rx"], r["value"])
            elif k == "kr":
                st.bars["kraken"].add_trade(r["ts"], r["price"], r["size"], r["taker_side"])
            elif k == "cbt":
                st.bars["coinbase"].add_trade(r["ts"], r["price"], r["size"], r["taker_side"])
            elif k == "cbb":
                st.books["coinbase"] = _book(r)
            elif k == "krb":
                st.books["kraken"] = _book(r)
            elif k == "pmb" and pd.notna(r.get("token")):
                st.pm_books[r["token"]] = _book(r)
            elif k == "kft" and pd.notna(r.get("mark")):
                st.futures_mark.add(r["ts"], r["rx"], float(r["mark"]))
            elif k == "kfx" and r.get("type") == "liquidation":
                st.liquidations.append((r["ts"], r["taker_side"], r["size"], r["price"]))
            elif k == "pmr":
                st.rounds[int(r["start"])] = PolyRound(start=int(r["start"]), slug=r["slug"], condition_id=r["condition_id"], token_up=r["token_up"], token_down=r["token_down"],
                                                       settlement_kind=r["settlement_kind"], twap_lookback_s=int(r["twap_lookback_s"] or 0),
                                                       fee_rate=float(r["fee_rate"] or 0.05), fee_exponent=float(r["fee_exponent"] or 1.0))
            i += 1
        out, feats = eng.snapshot(float(T))
        S = out["round_start"]
        if S not in res.index or not bool(res.loc[S, "resolved"]) or out.get("p_up") is None:
            continue
        row = {"start": S, "k": float(max(c for c in cps if c <= T - S)) if not every_s else float(T - S), "t_eval": T, "y": int(res.loc[S, "outcome_up"]), "p_engine": out["p_up"], "status": out["status"], "data_status": out["data_status"],
               "ptb_status": out["price_to_beat_status"], "ptb_err": (out["price_to_beat"] - res.loc[S, "price_to_beat"]) if out["price_to_beat"] else np.nan}
        row.update({kk: v for kk, v in feats.items() if isinstance(v, (int, float, np.floating))})
        m = out["market"]
        row.update({"pm_mid": m.get("implied_p_up_mid"), "pm_up_ask": m.get("up_ask"), "pm_down_ask": m.get("down_ask")})
        rows.append(row)
    df = pd.DataFrame(rows)
    out_path = out_path or data_dir() / "datasets" / "checkpoints_live.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("live_dataset_built", rows=len(df), rounds=int(df["start"].nunique()) if len(df) else 0)
    return df


if __name__ == "__main__":
    d = build_live_dataset()
    print(d.shape)
    if len(d):
        print(d.groupby("k")[["p_engine", "y"]].mean())
