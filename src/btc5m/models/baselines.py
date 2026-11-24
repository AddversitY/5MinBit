"""Baselines: constant (climatology), Gaussian random walk (TWAP-aware), Platt-calibrated
volatility model, and a Monte-Carlo model whose terminal distribution is bootstrapped from
observed 1-second returns.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from btc5m.models.base import Model, clip, sigmoid
from btc5m.rules import effective_variance_time


class Constant(Model):
    """Coin flip / climatology: training-set Up frequency (≈ 0.5)."""

    name = "constant"

    def fit(self, train, val=None):
        self.p = float(train["y"].mean())
        return self

    def predict(self, df):
        return np.full(len(df), self.p)

    def params(self):
        return {"p": self.p}


class Coin(Model):
    """Pure 50/50."""

    name = "coin_flip"

    def predict(self, df):
        return np.full(len(df), 0.5)


class GaussianRW(Model):
    """Driftless arithmetic random walk with the TWAP-aware variance; no fitted parameters.

    p = Phi(z), z = (E[settlement] - price_to_beat) / (sigma * sqrt(var_time)).
    """

    name = "gaussian_rw"

    def predict(self, df):
        return clip(norm.cdf(df["z"].to_numpy()))


class PlattVol(Model):
    """Volatility model with a fitted scale per checkpoint: p = sigmoid(a_k * z + b_k).

    Corrects systematic mis-scaling of sigma (e.g. proxy noise, fat tails)."""

    name = "vol_platt"

    def fit(self, train, val=None):
        self.coef = {}
        for k, g in train.groupby("k"):
            x = g[["z"]].to_numpy()
            lr = LogisticRegression(C=100.0, max_iter=1000).fit(np.clip(x, -10, 10), g["y"].to_numpy())
            self.coef[float(k)] = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
        return self

    def predict(self, df):
        out = np.empty(len(df))
        for i, (k, z) in enumerate(zip(df["k"].to_numpy(), df["z"].to_numpy())):
            a, b = self.coef.get(float(k), (1.7, 0.0))
            out[i] = sigmoid(a * np.clip(z, -10, 10) + b)
        return clip(out)

    def params(self):
        return {str(k): v for k, v in self.coef.items()}


class MonteCarloBootstrap(Model):
    """Monte Carlo with block-bootstrapped, volatility-standardised 1 s returns.

    From the training period we take 1 s reference-price changes divided by the trailing
    300 s realised volatility (``std_returns``). For each remaining horizon h we simulate
    ``n_paths`` paths by concatenating random 30 s blocks, apply the settlement rule (spot or
    trailing TWAP), and record the standardised terminal statistic
    Z = (A - S_now) / sqrt(var_time). Then P(Up) = P(Z > -z). This keeps the empirical tail
    shape and short-range autocorrelation that the Gaussian model ignores.
    """

    name = "monte_carlo"

    def __init__(self, std_returns: np.ndarray | None = None, n_paths: int = 20000, block: int = 30, seed: int = 7):
        self.std_returns = std_returns
        self.n_paths = n_paths
        self.block = block
        self.seed = seed
        self.dists: dict = {}

    def _simulate(self, h: int, L: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + h * 7 + L)
        r = self.std_returns
        nb = int(math.ceil(h / self.block))
        starts = rng.integers(0, len(r) - self.block, size=(self.n_paths, nb))
        idx = (starts[:, :, None] + np.arange(self.block)[None, None, :]).reshape(self.n_paths, -1)[:, :h]
        steps = r[idx]
        path = np.cumsum(steps, axis=1)
        if L > 0 and h >= 1:
            w = min(L, h)
            # mean of the path over the last w seconds (+ already-observed part treated as 0
            # displacement when h < L, matching projected_settlement's convention)
            a = path[:, -w:].sum(axis=1) / L if h < L else path[:, -L:].mean(axis=1)
        else:
            a = path[:, -1]
        vt = effective_variance_time(0.0, float(h), L)
        return np.sort(a / math.sqrt(max(vt, 1e-9)))

    def fit(self, train, val=None):
        if self.std_returns is None:
            raise ValueError("MonteCarloBootstrap needs std_returns from the training period")
        self.dists = {}
        keys = train[["rem", "twap_lookback_s"]].drop_duplicates().itertuples(index=False)
        for rem, L in keys:
            self.dists[(int(rem), int(L))] = self._simulate(int(rem), int(L))
        return self

    def predict(self, df):
        out = np.empty(len(df))
        for i, (rem, L, z) in enumerate(zip(df["rem"].to_numpy(), df["twap_lookback_s"].to_numpy(), df["z"].to_numpy())):
            key = (int(rem), int(L))
            if key not in self.dists:
                self.dists[key] = self._simulate(int(rem), int(L))
            d = self.dists[key]
            # P(Z > -z)
            out[i] = 1.0 - np.searchsorted(d, -z, side="right") / len(d)
        return clip(out)

    def params(self):
        return {"n_paths": self.n_paths, "block": self.block, "n_returns": int(len(self.std_returns)) if self.std_returns is not None else 0}


def standardized_returns(close: np.ndarray, window: int = 300) -> np.ndarray:
    """1 s price changes divided by trailing realised vol (USD/sqrt(s)), NaNs dropped."""
    d = np.diff(close)
    s = pd.Series(d * d).rolling(window, min_periods=window // 2).mean().shift(1).to_numpy()
    sd = np.sqrt(s)
    z = d / sd
    z = z[np.isfinite(z)]
    return np.clip(z, -50, 50)
