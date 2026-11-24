"""Chronological walk-forward splits with purge and embargo.

Rounds are never shuffled. Each fold:

    [ train ........ | val (last 20% of train window) ] [purge gap] [ test block ]

* Expanding training window; test blocks are consecutive and non-overlapping.
* A purge gap (default 1 h) separates the last training/validation round from the first
  test round, so no training label is determined after a test feature window begins.
* All checkpoints of a round stay in the same split (rows of one round are dependent).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    i: int
    train_starts: np.ndarray
    val_starts: np.ndarray
    test_starts: np.ndarray

    def describe(self) -> dict:
        f = lambda a: (int(a.min()), int(a.max()), int(len(a))) if len(a) else (None, None, 0)  # noqa: E731
        return {"fold": self.i, "train": f(self.train_starts), "val": f(self.val_starts), "test": f(self.test_starts)}


def walk_forward(starts: np.ndarray, min_train_days: float = 21, test_days: float = 7, val_frac: float = 0.2, purge_s: int = 3600, max_folds: int | None = None) -> list[Fold]:
    s = np.sort(np.unique(starts))
    t0, t_last = s[0], s[-1]
    folds = []
    test_lo = t0 + int(min_train_days * 86400)
    i = 0
    while test_lo <= t_last:
        test_hi = test_lo + int(test_days * 86400)
        test = s[(s >= test_lo) & (s < test_hi)]
        trv = s[s + 300 <= test_lo - purge_s]  # round end must precede test start by purge_s
        if len(test) == 0 or len(trv) < 100:
            test_lo = test_hi
            continue
        cut = trv[int(len(trv) * (1 - val_frac))]
        train = trv[trv < cut - purge_s]
        val = trv[trv >= cut]
        folds.append(Fold(i, train, val, test))
        i += 1
        test_lo = test_hi
    if max_folds:
        folds = folds[-max_folds:]
    return folds


def take(df: pd.DataFrame, starts: np.ndarray) -> pd.DataFrame:
    return df[df["start"].isin(set(starts.tolist()))]
