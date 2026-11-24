"""In-memory market state shared by live collectors, the prediction engine and the API.

Design rule: everything the engine reads is keyed by the *receive* time we observed it
(``rx``) as well as the venue's source timestamp. Feature code only ever uses data whose
receive time is <= the decision time, so live predictions cannot see the future, and the
same per-second bar representation is used for historical replays (features/bars.py).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

RING_S = 3600  # seconds of per-second bars kept in memory


class SecondBars:
    """Ring buffer of 1-second trade bars (taker-signed volume, VWAP, close)."""

    FIELDS = ("close", "vol", "buy", "sell", "ntr", "pv", "maxsz")

    def __init__(self, ring_s: int = RING_S):
        self.ring = ring_s
        self.sec = np.full(ring_s, -1, dtype=np.int64)
        self.arr = {f: np.zeros(ring_s, dtype=np.float64) for f in self.FIELDS}
        self.arr["close"][:] = np.nan
        self.last_price = np.nan
        self.last_ts = 0.0
        self._lock = threading.Lock()

    def add_trade(self, ts: float, price: float, size: float, taker_side: int) -> None:
        s = int(ts)
        i = s % self.ring
        with self._lock:
            if self.sec[i] != s:
                self.sec[i] = s
                for f in self.FIELDS:
                    self.arr[f][i] = 0.0
                self.arr["close"][i] = np.nan
            a = self.arr
            a["close"][i] = price
            a["vol"][i] += size
            if taker_side > 0:
                a["buy"][i] += size
            elif taker_side < 0:
                a["sell"][i] += size
            a["ntr"][i] += 1
            a["pv"][i] += price * size
            a["maxsz"][i] = max(a["maxsz"][i], size)
            self.last_price = price
            self.last_ts = ts

    def window(self, t_end: int, n: int) -> dict[str, np.ndarray]:
        """Bars for seconds [t_end - n, t_end) (i.e. strictly before t_end). close is ffilled."""
        secs = np.arange(t_end - n, t_end)
        idx = secs % self.ring
        with self._lock:
            valid = self.sec[idx] == secs
            out = {f: np.where(valid, self.arr[f][idx], 0.0) for f in self.FIELDS}
            close = np.where(valid, self.arr["close"][idx], np.nan)
        out["close"] = _ffill(close)
        out["valid"] = valid
        return out


def _ffill(x: np.ndarray) -> np.ndarray:
    mask = np.isnan(x)
    if not mask.any():
        return x
    idx = np.where(~mask, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    if mask[0]:
        first = np.argmax(~mask) if (~mask).any() else len(x)
        out[:first] = np.nan
    return out


@dataclass
class Book:
    """Top-N order book snapshot (sorted best-first)."""

    bids: list = field(default_factory=list)  # [(price, size), ...] descending
    asks: list = field(default_factory=list)  # ascending
    ts: float = 0.0
    rx: float = 0.0

    def best(self):
        bb = self.bids[0] if self.bids else (np.nan, np.nan)
        ba = self.asks[0] if self.asks else (np.nan, np.nan)
        return bb, ba


class PointSeries:
    """(source_ts, rx_ts, value) observations, e.g. Chainlink spot/TWAP ticks."""

    def __init__(self, maxlen: int = 7200):
        self.q: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, src_ts: float, rx: float, value: float) -> None:
        with self._lock:
            if self.q and src_ts <= self.q[-1][0]:
                # snapshot replays / duplicates: keep monotone source time
                if src_ts == self.q[-1][0]:
                    self.q[-1] = (src_ts, min(rx, self.q[-1][1]), value)
                return
            self.q.append((src_ts, rx, value))

    def asof(self, t: float, by: str = "rx"):
        """Latest observation with (rx|src) time <= t."""
        k = 1 if by == "rx" else 0
        with self._lock:
            for item in reversed(self.q):
                if item[k] <= t:
                    return item
        return None

    def value_at_src(self, src_ts: float):
        with self._lock:
            for s, rx, v in reversed(self.q):
                if s <= src_ts:
                    return (s, rx, v)
        return None

    def per_second(self, t_end: int, n: int, known_by: float | None = None) -> np.ndarray:
        """Values on the source-second grid [t_end - n, t_end), ffilled; entries not yet
        received by ``known_by`` are excluded (NaN then ffilled)."""
        out = np.full(n, np.nan)
        with self._lock:
            items = list(self.q)
        base = t_end - n
        for s, rx, v in items:
            si = int(s) - base
            if 0 <= si < n and (known_by is None or rx <= known_by):
                out[si] = v
        return _ffill(out)

    def last(self):
        with self._lock:
            return self.q[-1] if self.q else None


@dataclass
class PolyRound:
    start: int
    slug: str
    condition_id: str | None = None
    token_up: str | None = None
    token_down: str | None = None
    price_to_beat: float | None = None
    settlement_kind: str = "twap60"
    twap_lookback_s: int = 60
    fee_rate: float = 0.05
    fee_exponent: float = 1.0
    tick_size: float = 0.01


class MarketState:
    """Singleton-ish container wired into every collector."""

    def __init__(self):
        self.bars = {"coinbase": SecondBars(), "kraken": SecondBars(), "binance": SecondBars(), "kraken_fut": SecondBars()}
        self.books = {"coinbase": Book(), "kraken": Book()}
        self.chainlink_spot = PointSeries()
        self.chainlink_twap = PointSeries()
        self.binance_relay = PointSeries()  # Polymarket RTDS relay of Binance BTCUSDT
        self.futures_mark = PointSeries()
        self.liquidations: deque = deque(maxlen=5000)  # (ts, side, size, price)
        self.pm_books: dict[str, Book] = {}
        self.pm_trades: deque = deque(maxlen=20000)  # (ts, token, side, price, size)
        self.rounds: dict[int, PolyRound] = {}
        self.health: dict[str, object] = {}
        self.started = time.time()

    def register_health(self, name: str, health) -> None:
        self.health[name] = health

    def health_snapshot(self) -> list[dict]:
        now = time.time()
        return [h.snapshot(now) for h in self.health.values()]


STATE = MarketState()
