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
    up_px = df[f"fill_up_l{latency}"].to_numpy()
    dn_px = df[f"fill_dn_l{latency}"].to_numpy()
    y = df["y"].to_numpy()
    rows = []
    for i in range(len(df)):
        fs = FeeSchedule(df["fee_rate"].iat[i], df["fee_exponent"].iat[i]) if use_market_fee else fee
        best = None
        for side, px, pw in (("up", up_px[i], p[i]), ("down", dn_px[i], 1 - p[i])):
            if not np.isfinite(px) or px <= 0 or px >= 1:
                continue
            cost = min(px + slip, 0.999)
            f = fs.taker_fee_per_share(cost)
            ev = pw - cost - f
            if ev > edge and (best is None or ev > best[3]):
                best = (side, cost, f, ev)
        if best:
            side, cost, f, ev = best
            win = (y[i] == 1) if side == "up" else (y[i] == 0)
            rows.append({"start": int(df["start"].iat[i]), "k": float(df["k"].iat[i]), "side": side, "price": cost, "fee": f, "exp_edge": ev, "win": int(win), "pnl": float(win) - cost - f})
    return pd.DataFrame(rows)


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
    days = (tr["start"] // 86400).to_numpy()
    point, lo, hi = day_block_bootstrap(lambda idx: float(np.mean(pnl[idx])) if len(idx) else 0.0, days, n_boot=500)
    return {
        "rounds": n_rounds, "trades": int(len(tr)), "trade_rate": len(tr) / max(n_rounds, 1),
        "hit_rate": float(tr["win"].mean()), "avg_price": float(tr["price"].mean()), "avg_fee": float(tr["fee"].mean()),
        "mean_pnl_per_share": point, "mean_pnl_ci95": [lo, hi], "total_pnl_shares": float(pnl.sum()),
        "max_drawdown_shares": max_drawdown(pnl), "avg_expected_edge": float(tr["exp_edge"].mean()),
        "pnl_sd": float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
    }
