"""Live 1-minute candle cache (Coinbase REST) for long-horizon volatility and HMM regimes."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from btc5m.config import load_config
from btc5m.features.dataset import minute_vol_context
from btc5m.http import client, get_json
from btc5m.logging import get_logger
from btc5m.storage.store import data_dir
from btc5m.timeutil import iso

log = get_logger("candles")


class CandleCache:
    def __init__(self, keep_days: float = 8.0):
        self.keep = int(keep_days * 86400)
        self.df = pd.DataFrame(columns=["ts", "low", "high", "open", "close", "volume"])
        self.ctx: pd.DataFrame | None = None
        self.last_refresh = 0.0
        self.c = None

    def _fetch(self, t0: int, t1: int) -> list[dict]:
        cfg = load_config()
        url = f"{cfg.exchanges.coinbase.rest}/products/{cfg.exchanges.coinbase.product}/candles"
        rows = []
        t = t0
        while t < t1:
            t2 = min(t + 300 * 60, t1)
            j = get_json(self.c, url, params={"granularity": 60, "start": iso(t), "end": iso(t2)}, retries=3)
            rows += [{"ts": int(k[0]), "low": float(k[1]), "high": float(k[2]), "open": float(k[3]), "close": float(k[4]), "volume": float(k[5])} for k in j]
            t = t2
        return rows

    def refresh(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_refresh < 30:
            return
        self.c = self.c or client()
        if self.df.empty:
            f = data_dir() / "hist" / "coinbase_candles_1m" / "candles.parquet"
            if f.exists():
                self.df = pd.read_parquet(f)
                self.df = self.df[self.df["ts"] >= now - self.keep]
        last = int(self.df["ts"].max()) if len(self.df) else int(now - self.keep)
        try:
            rows = self._fetch(max(last - 300, int(now - self.keep)), int(now // 60 * 60))
        except Exception as e:
            log.warning("candle_refresh_failed", err=repr(e))
            return
        if rows:
            self.df = pd.concat([self.df, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("ts", keep="last")
            self.df = self.df[self.df["ts"] >= now - self.keep].sort_values("ts").reset_index(drop=True)
            # only closed candles: drop the currently forming minute
            self.df = self.df[self.df["ts"] + 60 <= now]
            self.ctx = minute_vol_context(self.df)
        self.last_refresh = now

    def vol_at(self, t: int) -> tuple[float, float, float]:
        if self.ctx is None or self.ctx.empty:
            return (np.nan, np.nan, np.nan)
        i = self.ctx.index.searchsorted(t, side="right") - 1
        if i < 0 or t - self.ctx.index[i] > 600:
            return (np.nan, np.nan, np.nan)
        r = self.ctx.iloc[i]
        return float(r["rv_1h"]), float(r["rv_4h"]), float(r["vol_pct_7d"])

    def minute_returns(self) -> pd.Series:
        if self.df.empty:
            return pd.Series(dtype=float)
        c = self.df.set_index("ts")["close"].sort_index()
        full = c.reindex(np.arange(int(c.index.min()), int(c.index.max()) + 60, 60)).ffill()
        return np.log(full).diff().dropna()
