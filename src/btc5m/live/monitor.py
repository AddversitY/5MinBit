"""Drift and degradation monitoring.

* Feature drift: Population Stability Index (PSI) of key live features vs the training
  dataset (PSI > 0.25 is conventionally a significant shift).
* Performance decay: rolling Brier of settled live predictions vs the out-of-sample Brier
  expected for the same checkpoint (from the latest walk-forward report).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from btc5m.live.paper_report import ledger_frame
from btc5m.models.registry import latest_report
from btc5m.storage.store import data_dir

KEY_FEATURES = ["z", "sigma_rel", "m60", "ofi60", "vol_pct_7d"]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected, actual = expected[np.isfinite(expected)], actual[np.isfinite(actual)]
    if len(expected) < 100 or len(actual) < 30:
        return float("nan")
    qs = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return float("nan")
    e = np.histogram(expected, qs)[0] / len(expected)
    a = np.histogram(np.clip(actual, qs[0], qs[-1]), qs)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


class DriftMonitor:
    def __init__(self):
        self.ref = None
        f = data_dir() / "datasets" / "checkpoints_kraken.parquet"
        if f.exists():
            try:
                self.ref = pd.read_parquet(f, columns=[c for c in KEY_FEATURES if c != "sigma_rel"] + ["sigma_rel"]).sample(frac=1.0, random_state=0).iloc[:50000]
            except Exception:
                self.ref = None
        self.live: dict[str, list] = {c: [] for c in KEY_FEATURES}

    def observe(self, features: dict) -> None:
        for c in KEY_FEATURES:
            v = features.get(c)
            if v is not None:
                self.live[c].append(v)
                if len(self.live[c]) > 20000:
                    self.live[c] = self.live[c][-10000:]

    def report(self) -> dict:
        out = {"feature_psi": {}, "performance": {}}
        if self.ref is not None:
            for c in KEY_FEATURES:
                out["feature_psi"][c] = psi(self.ref[c].to_numpy(), np.asarray(self.live[c][-5000:], float))
        _, rep = latest_report()
        df = ledger_frame()
        if rep and not df.empty:
            exp = rep["metrics"]["by_checkpoint"]
            for k, g in df.groupby("checkpoint"):
                g = g.tail(200)
                live_b = float(np.mean((g["p_up"] - g["y"]) ** 2))
                fam = g["model_version"].iloc[-1].split("-")[-1] if "-" in g["model_version"].iloc[-1] else "gaussian_rw"
                e = (exp.get(str(int(k))) or exp.get(int(k)) or {}).get(f"p_{fam}") or (exp.get(str(int(k))) or {}).get("p_gaussian_rw") or {}
                out["performance"][int(k)] = {"live_brier_last200": live_b, "n": int(len(g)), "expected_oos_brier": e.get("brier"),
                                              "degraded": bool(e.get("brier") is not None and len(g) >= 100 and live_b > e["brier"] + 0.02)}
        out["alerts"] = [f"feature drift {c} (PSI={v:.2f})" for c, v in out["feature_psi"].items() if np.isfinite(v) and v > 0.25]
        out["alerts"] += [f"performance degraded at k={k}" for k, v in out["performance"].items() if v["degraded"]]
        return out
