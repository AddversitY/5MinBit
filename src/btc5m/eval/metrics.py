"""Probabilistic forecast metrics + block-bootstrap confidence intervals.

Rows from the same day are dependent (volatility clustering), so confidence intervals use a
day-block bootstrap: whole days are resampled with replacement.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-4


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def logloss(p, y) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(p, y) -> float:
    return float(np.mean((np.asarray(p) >= 0.5) == (np.asarray(y) == 1)))


def ece(p, y, bins: int = 10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def reliability(p, y, bins: int = 10) -> list[dict]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() > 0:
            out.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()), "mean_p": float(p[m].mean()), "freq_up": float(y[m].mean())})
    return out


def precision_recall(p, y) -> dict:
    p, y = np.asarray(p), np.asarray(y)
    pred = p >= 0.5
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1))
    tn = np.sum(~pred & (y == 0))
    return {
        "precision_up": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "recall_up": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "precision_down": float(tn / (tn + fn)) if tn + fn else float("nan"),
        "recall_down": float(tn / (tn + fp)) if tn + fp else float("nan"),
    }


def auc(p, y) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def summary(p, y) -> dict:
    out = {"n": int(len(y)), "accuracy": accuracy(p, y), "brier": brier(p, y), "logloss": logloss(p, y), "ece": ece(p, y), "auc": auc(p, y)}
    out["brier_skill_vs_coin"] = 1.0 - out["brier"] / 0.25
    out.update(precision_recall(p, y))
    return out


def day_block_bootstrap(stat_fn, groups: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap a statistic computed on row indices, resampling whole groups (days).

    ``stat_fn(idx)`` receives an array of row indices. Returns (point, lo, hi).
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    point = stat_fn(np.arange(len(groups)))
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        stats.append(stat_fn(idx))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)
