"""Common model interface: every model outputs a calibrated P(Up) per row."""
from __future__ import annotations

import numpy as np
import pandas as pd

P_MIN, P_MAX = 1e-4, 1 - 1e-4


def clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), P_MIN, P_MAX)


def logit(p) -> np.ndarray:
    p = clip(p)
    return np.log(p / (1 - p))


def sigmoid(x) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


class Model:
    name = "model"
    features: list[str] = []

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "Model":
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def params(self) -> dict:
        return {}
