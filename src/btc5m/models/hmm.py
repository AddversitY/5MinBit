"""Hidden Markov regime model on 1-minute log returns (Coinbase candles).

Parameters are fitted on the training period only. State probabilities are computed with
the FORWARD (filtering) recursion, so the regime at minute m uses returns up to m only.
(hmmlearn's ``predict_proba`` is forward-backward smoothing, which would leak the future.)
States are ordered by their return volatility (state 0 = calmest).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import norm


class RegimeHMM:
    def __init__(self, n_states: int = 3, seed: int = 5):
        self.n = n_states
        self.seed = seed

    def fit(self, minute_returns: pd.Series) -> "RegimeHMM":
        x = minute_returns.to_numpy().reshape(-1, 1) * 1e4  # bps
        m = GaussianHMM(n_components=self.n, covariance_type="diag", n_iter=200, random_state=self.seed)
        m.fit(x)
        sd = np.sqrt(m.covars_.reshape(self.n, -1)[:, 0])
        order = np.argsort(sd)
        self.means = m.means_[order, 0]
        self.sds = sd[order]
        self.trans = m.transmat_[np.ix_(order, order)]
        self.start = m.startprob_[order]
        return self

    def filter(self, minute_returns: pd.Series) -> pd.DataFrame:
        """Filtered P(state | returns up to and including minute m), indexed like the input."""
        x = minute_returns.to_numpy() * 1e4
        logB = np.column_stack([norm.logpdf(x, self.means[j], self.sds[j]) for j in range(self.n)])
        logA = np.log(self.trans + 1e-300)
        la = np.log(self.start + 1e-300) + logB[0]
        la -= logsumexp(la)
        out = np.empty((len(x), self.n))
        out[0] = np.exp(la)
        for t in range(1, len(x)):
            la = logsumexp(la[:, None] + logA, axis=0) + logB[t]
            la -= logsumexp(la)
            out[t] = np.exp(la)
        cols = [f"hmm_p{j}" for j in range(self.n)]
        df = pd.DataFrame(out, index=minute_returns.index, columns=cols)
        df["hmm_vol"] = out @ self.sds  # expected per-minute vol (bps) under the filtered state
        return df

    def params(self) -> dict:
        return {"means_bps": self.means.tolist(), "sds_bps": self.sds.tolist(), "trans": self.trans.tolist()}


def attach_regimes(df: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """Join filtered regime probabilities as of the last fully closed minute before each
    decision time t = start + k (minus the data lag)."""
    t = (df["start"] + df["k"]).to_numpy() - 1
    # candle starting at ts closes at ts+60; usable if ts + 60 <= t
    idx = regimes.index.to_numpy()
    pos = np.searchsorted(idx + 60, t, side="right") - 1
    out = df.copy()
    for c in regimes.columns:
        vals = regimes[c].to_numpy()
        out[c] = np.where(pos >= 0, vals[np.clip(pos, 0, None)], np.nan)
    return out
