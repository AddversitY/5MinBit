"""Hypothetical taker-strategy simulation (paper only — NOT realised profits).

At a chosen checkpoint k, buy one share of Up if  p_up - (ask_up + fee(ask_up) + slip) > edge,
or one share of Down if (1 - p_up) - (ask_dn + fee(ask_dn) + slip) > edge. At most one position
per round. Execution price is the first observed taker print on that side after
t + latency (features/market_implied.fill_price); rounds without such evidence are skipped.
Payoff per share: 1 if the bought outcome wins, else 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from btc5m.eval.metrics import day_block_bootstrap
from btc5m.rules import FeeSchedule


def simulate(df: pd.DataFrame, p: np.ndarray, edge: float = 0.0, latency: int = 2, fee: FeeSchedule | None = None, slip: float = 0.01, use_market_fee: bool = False) -> pd.DataFrame:
    """Return one row per executed hypothetical trade."""
    fee = fee or FeeSchedule(0.05, 1.0)
    p = np.asarray(p, float)
    y = df["y"].to_numpy()
    if use_market_fee:
        rate = df["fee_rate"].fillna(fee.rate).to_numpy(float)
        expo = df["fee_exponent"].fillna(fee.exponent).to_numpy(float)
    else:
        rate = np.full(len(df), fee.rate)
        expo = np.full(len(df), fee.exponent)

    def leg(px, pw):
        ok = np.isfinite(px) & (px > 0) & (px < 1)
        cost = np.minimum(np.where(ok, px, np.nan) + slip, 0.999)
        f = np.round(rate * (cost * (1 - cost)) ** expo, 5)
        ev = pw - cost - f
        return cost, f, np.where(ok, ev, -np.inf)

    cu, fu, evu = leg(df[f"fill_up_l{latency}"].to_numpy(float), p)
    cd, fd, evd = leg(df[f"fill_dn_l{latency}"].to_numpy(float), 1 - p)
    up = evu >= evd
    ev = np.where(up, evu, evd)
    take = ev > edge
    if not take.any():
        return pd.DataFrame(columns=["start", "k", "side", "price", "fee", "exp_edge", "win", "pnl"])
    cost = np.where(up, cu, cd)[take]
    f = np.where(up, fu, fd)[take]
    win = np.where(up[take], y[take] == 1, y[take] == 0).astype(int)
    return pd.DataFrame({
        "start": df["start"].to_numpy()[take].astype(int), "k": df["k"].to_numpy()[take].astype(float),
        "side": np.where(up[take], "up", "down"), "price": cost, "fee": f, "exp_edge": ev[take], "win": win, "pnl": win - cost - f,
    })


def max_drawdown(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    return float(np.max(peak - eq))


def summarize_trades(tr: pd.DataFrame, n_rounds: int) -> dict:
    if tr is None or len(tr) == 0:
        return {"trades": 0, "rounds": n_rounds}
    tr = tr.sort_values("start")
    pnl = tr["pnl"].to_numpy()
    days = (tr["start"] // (4 * 3600)).to_numpy()  # 4-hour bootstrap blocks
    point, lo, hi = day_block_bootstrap(lambda idx: float(np.mean(pnl[idx])) if len(idx) else 0.0, days, n_boot=500)
    return {
        "rounds": n_rounds, "trades": int(len(tr)), "trade_rate": len(tr) / max(n_rounds, 1),
        "hit_rate": float(tr["win"].mean()), "avg_price": float(tr["price"].mean()), "avg_fee": float(tr["fee"].mean()),
        "mean_pnl_per_share": point, "mean_pnl_ci95": [lo, hi], "total_pnl_shares": float(pnl.sum()),
        "max_drawdown_shares": max_drawdown(pnl), "avg_expected_edge": float(tr["exp_edge"].mean()),
        "pnl_sd": float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
    }
