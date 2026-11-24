"""Per-second bar arrays for historical replays, with the same ``window()`` API as the live
``SecondBars`` ring buffer, so feature code is identical in backtests and live trading.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from btc5m.live.state import _ffill

FIELDS = ("close", "vol", "buy", "sell", "ntr", "pv", "maxsz")


class BarFrame:
    """Dense 1-second bars for seconds [t0, t1). ``close`` is forward-filled."""

    def __init__(self, t0: int, arrays: dict[str, np.ndarray], valid: np.ndarray):
        self.t0 = int(t0)
        self.n = len(valid)
        self.a = arrays
        self.valid = valid

    @property
    def t1(self) -> int:
        return self.t0 + self.n

    @classmethod
    def from_trades(cls, ts: np.ndarray, price: np.ndarray, size: np.ndarray, taker_side: np.ndarray, t0: int | None = None, t1: int | None = None) -> "BarFrame":
        order = np.argsort(ts, kind="stable")
        ts, price, size, taker_side = ts[order], price[order], size[order], taker_side[order]
        sec = np.floor(ts).astype(np.int64)
        t0 = int(sec.min()) if t0 is None else int(t0)
        t1 = int(sec.max()) + 1 if t1 is None else int(t1)
        keep = (sec >= t0) & (sec < t1)
        sec, price, size, taker_side = sec[keep], price[keep], size[keep], taker_side[keep]
        n = t1 - t0
        i = sec - t0
        vol = np.bincount(i, weights=size, minlength=n)
        buy = np.bincount(i, weights=size * (taker_side > 0), minlength=n)
        sell = np.bincount(i, weights=size * (taker_side < 0), minlength=n)
        ntr = np.bincount(i, minlength=n).astype(np.float64)
        pv = np.bincount(i, weights=price * size, minlength=n)
        maxsz = np.zeros(n)
        np.maximum.at(maxsz, i, size)
        close = np.full(n, np.nan)
        # last trade in each second: since sorted by time, later writes win
        close[i] = price
        valid = ntr > 0
        return cls(t0, {"close": _ffill(close), "vol": vol, "buy": buy, "sell": sell, "ntr": ntr, "pv": pv, "maxsz": maxsz}, valid)

    def window(self, t_end: int, n: int) -> dict[str, np.ndarray]:
        lo, hi = t_end - n - self.t0, t_end - self.t0
        if lo < 0 or hi > self.n:
            raise IndexError(f"window [{t_end - n},{t_end}) outside bars [{self.t0},{self.t1})")
        out = {f: self.a[f][lo:hi] for f in FIELDS}
        out["valid"] = self.valid[lo:hi]
        return out

    def close_at(self, s: int) -> float:
        """Close of second s (last trade at or before end of second s)."""
        return float(self.a["close"][s - self.t0])

    def mean_close(self, a: int, b: int) -> float:
        """Mean of per-second close over seconds [a, b)."""
        return float(np.nanmean(self.a["close"][a - self.t0 : b - self.t0]))


def minute_log_returns(candles: pd.DataFrame) -> pd.Series:
    c = candles.set_index("ts")["close"].sort_index()
    full = c.reindex(np.arange(c.index.min(), c.index.max() + 60, 60)).ffill()
    return np.log(full).diff().dropna()
