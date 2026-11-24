"""Market-implied probability and executable-price reconstruction from Polymarket taker trades.

Polymarket's Up and Down books are mirrored (a Down bid at q is an Up offer at 1-q). From
taker-side trades (Data API v2, ``taker_only=true``):

    ask_Up evidence  : taker BUY  Up at p          -> p
                       taker SELL Down at q        -> 1 - q
    bid_Up evidence  : taker SELL Up at p          -> p
                       taker BUY  Down at q        -> 1 - q

For a decision at time t we only use trades with timestamp < t. The Data API timestamp is the
on-chain block timestamp, which is at or after the off-chain match, so this is conservative.

Executable price for a hypothetical taker order sent at t: the first ask evidence for that
token in [t + latency, t + latency + window) plus configurable slippage. No evidence => no
fill (the trade is skipped, which avoids assuming liquidity that may not have existed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_round_trades(tr: pd.DataFrame, token_up: str) -> pd.DataFrame:
    """Add up-equivalent ask/bid evidence columns; sort by time."""
    t = tr.sort_values("ts", kind="stable").reset_index(drop=True)
    is_up = (t["token_id"] == token_up).to_numpy()
    buy = (t["side"] == "BUY").to_numpy()
    p = t["price"].to_numpy()
    up_px = np.where(is_up, p, 1.0 - p)
    # +1: evidence of Up ask (someone lifted Up / hit a Down bid); -1: evidence of Up bid
    kind = np.where(is_up == buy, 1, -1)
    t["up_px"] = up_px
    t["ev"] = kind
    # signed Up-flow in shares: +size for buying Up exposure
    t["up_flow"] = np.where(kind > 0, t["size"].to_numpy(), -t["size"].to_numpy())
    return t


def implied_at(t_rt: pd.DataFrame, t: float, window: float = 30.0) -> dict:
    """Market-implied state at decision time t using trades strictly before t."""
    ts = t_rt["ts"].to_numpy()
    hi = np.searchsorted(ts, t, side="left")
    lo = np.searchsorted(ts, t - window, side="left")
    seg = t_rt.iloc[lo:hi]
    out = {"pm_n30": float(len(seg)), "pm_flow30": float(seg["up_flow"].sum()) if len(seg) else 0.0}
    if hi == 0:
        out.update(pm_ask=np.nan, pm_bid=np.nan, pm_mid=np.nan, pm_last=np.nan, pm_age=np.nan)
        return out
    past = t_rt.iloc[max(0, hi - 400):hi]
    asks = past[past["ev"] > 0]
    bids = past[past["ev"] < 0]
    ask = float(asks["up_px"].iloc[-1]) if len(asks) else np.nan
    bid = float(bids["up_px"].iloc[-1]) if len(bids) else np.nan
    age_a = t - float(asks["ts"].iloc[-1]) if len(asks) else np.inf
    age_b = t - float(bids["ts"].iloc[-1]) if len(bids) else np.inf
    # stale evidence (> window) is discarded
    if age_a > window:
        ask = np.nan
    if age_b > window:
        bid = np.nan
    if np.isfinite(ask) and np.isfinite(bid):
        mid = 0.5 * (ask + bid) if ask >= bid else float(past["up_px"].iloc[-1])
    elif np.isfinite(ask):
        mid = ask - 0.005
    elif np.isfinite(bid):
        mid = bid + 0.005
    else:
        mid = np.nan
    out.update(pm_ask=ask, pm_bid=bid, pm_mid=mid, pm_last=float(past["up_px"].iloc[-1]), pm_age=float(min(age_a, age_b)))
    return out


def fill_price(t_rt: pd.DataFrame, t: float, buy_up: bool, latency: float = 2.0, window: float = 15.0) -> float:
    """First executable-price evidence for buying Up (or Down) after t + latency. NaN if none.

    Returns the price of the token being bought (Up price if buy_up else Down price).
    """
    ts = t_rt["ts"].to_numpy()
    lo = np.searchsorted(ts, t + latency, side="left")
    hi = np.searchsorted(ts, t + latency + window, side="left")
    seg = t_rt.iloc[lo:hi]
    if buy_up:
        ev = seg[seg["ev"] > 0]
        return float(ev["up_px"].iloc[0]) if len(ev) else np.nan
    ev = seg[seg["ev"] < 0]  # Up bid evidence == Down ask at 1 - bid
    return float(1.0 - ev["up_px"].iloc[0]) if len(ev) else np.nan
