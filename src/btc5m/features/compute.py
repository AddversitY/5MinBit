"""Point-in-time feature computation for one (round, decision time) snapshot.

The SAME function is used for historical dataset construction and for live predictions.
Inputs are arrays covering the ``LOOKBACK`` seconds strictly before the decision time, so no
future information can enter. Prices are in USD; most features are made scale-free by
dividing by a volatility estimate sigma (USD per sqrt(second)).

Reference price (``ref``): the series that settles the market. Live, this is the Chainlink
BTC/USD stream from Polymarket RTDS. Historically, it is an exchange-trade proxy shifted by a
per-round basis anchored on the official price-to-beat (see features/dataset.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from btc5m.rules import TWAP60, effective_variance_time, norm_cdf

LOOKBACK = 900
MOM_H = (5, 15, 30, 60, 120, 300, 600)
EPS = 1e-12


@dataclass
class Snapshot:
    t: int                      # decision time, integer epoch second (data < t is known)
    start: int
    end: int
    ptb: float                  # official price to beat
    kind: str                   # settlement kind (spot | twap60)
    lookback_s: int             # TWAP window (0 for spot)
    ref: np.ndarray             # per-second reference price for seconds [t-LOOKBACK, t)
    bars: dict                  # exchange 1 s trade bars over the same seconds
    rv_1h: float = float("nan")         # USD/sqrt(s) from 1-minute candles, last hour
    rv_4h: float = float("nan")
    vol_pct_7d: float = float("nan")    # percentile of rv_1h within trailing 7 days
    extra: dict = field(default_factory=dict)  # live-only / market features (already scale-free)


def _rv(x: np.ndarray) -> float:
    d = np.diff(x)
    d = d[~np.isnan(d)]
    if len(d) < 5:
        return float("nan")
    return float(math.sqrt(np.mean(d * d)))


def sigma_blend(f: dict) -> float:
    comps = [f.get("rv300"), f.get("rv900"), f.get("rv_1h")]
    comps = [c for c in comps if c is not None and np.isfinite(c) and c > 0]
    if not comps:
        return float("nan")
    return float(math.sqrt(np.mean(np.square(comps))))


def projected_settlement(ref: np.ndarray, t: int, end: int, kind: str, lookback_s: int) -> tuple[float, float]:
    """(expected settlement value, variance-time in seconds) under a driftless random walk.

    For a trailing TWAP, the part of the averaging window already observed is averaged in.
    """
    now = float(ref[-1])
    if kind != TWAP60 and lookback_s <= 0:
        return now, max(end - t, 0)
    L = lookback_s or 60
    w0 = end - L
    if t <= w0:
        return now, effective_variance_time(t, end, L)
    n_obs = t - w0
    obs = ref[-n_obs:]
    exp = (float(np.nansum(obs)) + now * (end - t)) / L
    return exp, effective_variance_time(t, end, L)


def compute_features(s: Snapshot) -> dict:
    ref, b = s.ref, s.bars
    f: dict = {}
    k = s.t - s.start
    f["k"] = float(k)
    f["rem"] = float(s.end - s.t)
    now = float(ref[-1])
    f["ref_now"] = now
    if not np.isfinite(now):
        return f

    # --- volatility -------------------------------------------------------------------
    for h in (60, 300, 900):
        f[f"rv{h}"] = _rv(ref[-h - 1 :])
    f["rv_1h"] = s.rv_1h
    f["rv_4h"] = s.rv_4h
    sig = sigma_blend(f)
    f["sigma"] = sig
    if not (np.isfinite(sig) and sig > 0):
        return f
    f["vol_ratio_60_900"] = f["rv60"] / (f["rv900"] + EPS) if np.isfinite(f["rv900"]) else float("nan")
    f["vol_ratio_300_1h"] = f["rv300"] / (s.rv_1h + EPS) if np.isfinite(s.rv_1h) else float("nan")
    f["vol_pct_7d"] = s.vol_pct_7d
    f["sigma_rel"] = sig / now * 1e4  # bps per sqrt(s)

    # --- settlement geometry --------------------------------------------------------------
    exp_settle, vt = projected_settlement(ref, s.t, s.end, s.kind, s.lookback_s)
    f["var_time"] = vt
    f["d"] = (now - s.ptb) / (sig * math.sqrt(max(f["rem"], 1.0)))
    f["z"] = (exp_settle - s.ptb) / (sig * math.sqrt(max(vt, 1e-6)))
    f["p_gauss"] = norm_cdf(f["z"])
    pg = min(max(f["p_gauss"], 1e-4), 1 - 1e-4)
    f["lp_gauss"] = math.log(pg / (1 - pg))  # Gaussian log-odds: base margin for residual models
    # gap between the round-open reference and the price to beat (TWAP lag effect)
    i_open = len(ref) - 1 - k if 0 <= k < len(ref) else None
    if i_open is not None and k > 0:
        ref_open = float(ref[i_open])
        f["open_gap"] = (ref_open - s.ptb) / (sig * math.sqrt(60.0))
        f["ret_since_open"] = (now - ref_open) / (sig * math.sqrt(max(k, 1)))
    else:
        f["open_gap"] = (now - s.ptb) / (sig * math.sqrt(60.0))
        f["ret_since_open"] = 0.0

    # --- momentum / acceleration / mean reversion ---------------------------------------
    for h in MOM_H:
        past = ref[-1 - h] if h < len(ref) else np.nan
        f[f"m{h}"] = (now - past) / (sig * math.sqrt(h))
    prev15 = (ref[-16] - ref[-31]) / (sig * math.sqrt(15))
    f["acc15"] = f["m15"] - prev15
    alpha = 2.0 / (120 + 1)
    w = (1 - alpha) ** np.arange(len(ref))[::-1]
    r = np.nan_to_num(ref, nan=now)
    ema = float(np.sum(w * r) / np.sum(w))
    f["ema120_dev"] = (now - ema) / (sig * math.sqrt(120))
    d1 = np.diff(ref[-61:])
    nz = d1[d1 != 0]
    f["up_frac60"] = float(np.mean(nz > 0)) if len(nz) else 0.5
    f["m15_x_m120"] = f["m15"] * f["m120"]

    # --- order flow (exchange trades) ---------------------------------------------------------
    buy, sell, vol, ntr = b["buy"], b["sell"], b["vol"], b["ntr"]
    vps = float(np.sum(vol)) / len(vol) + EPS  # mean volume per second over lookback
    for h in (15, 60, 300):
        bb, ss = float(np.sum(buy[-h:])), float(np.sum(sell[-h:]))
        f[f"ofi{h}"] = (bb - ss) / (bb + ss + EPS)
        f[f"svol{h}"] = (bb - ss) / (vps * h)
    if k > 0:
        kk = min(k, len(buy))
        bb, ss = float(np.sum(buy[-kk:])), float(np.sum(sell[-kk:]))
        f["cvd_round"] = (bb - ss) / (vps * kk)
    else:
        f["cvd_round"] = 0.0
    f["intensity60"] = (float(np.sum(ntr[-60:])) / 60.0) / (float(np.sum(ntr)) / len(ntr) + EPS)
    f["vol_intensity60"] = (float(np.sum(vol[-60:])) / 60.0) / vps
    mx = b["maxsz"]
    thr = np.quantile(mx[mx > 0], 0.95) if np.any(mx > 0) else np.inf
    big = mx[-60:] >= thr
    f["large_net60"] = float(np.sum(np.sign(buy[-60:] - sell[-60:]) * big))
    pv300, v300 = float(np.sum(b["pv"][-300:])), float(np.sum(vol[-300:]))
    ex_close = float(b["close"][-1])
    if v300 > 0 and np.isfinite(ex_close):
        f["vwap300_dev"] = (ex_close - pv300 / v300) / (sig * math.sqrt(300))
    else:
        f["vwap300_dev"] = 0.0

    # --- calendar -------------------------------------------------------------------------
    hour = (s.start % 86400) / 3600.0
    f["hour_sin"] = math.sin(2 * math.pi * hour / 24)
    f["hour_cos"] = math.cos(2 * math.pi * hour / 24)
    f["weekend"] = float(((s.start // 86400) + 4) % 7 >= 5)  # 1970-01-01 was a Thursday

    f.update(s.extra)
    return f


# Features computable both historically and live (the validated set).
CORE_FEATURES = [
    "k", "rem", "z", "lp_gauss", "d", "open_gap", "ret_since_open",
    "rv60", "rv300", "rv900", "rv_1h", "rv_4h", "sigma_rel", "vol_ratio_60_900", "vol_ratio_300_1h", "vol_pct_7d",
    *[f"m{h}" for h in MOM_H], "acc15", "ema120_dev", "up_frac60", "m15_x_m120",
    "ofi15", "ofi60", "ofi300", "svol15", "svol60", "svol300", "cvd_round", "intensity60", "vol_intensity60", "large_net60", "vwap300_dev",
    "hour_sin", "hour_cos", "weekend",
]

# Feature groups for ablation tests
FEATURE_GROUPS = {
    "distance": ["z", "lp_gauss", "d", "open_gap", "ret_since_open"],
    "volatility": ["rv60", "rv300", "rv900", "rv_1h", "rv_4h", "sigma_rel", "vol_ratio_60_900", "vol_ratio_300_1h", "vol_pct_7d"],
    "momentum": [f"m{h}" for h in MOM_H] + ["acc15", "m15_x_m120", "up_frac60"],
    "mean_reversion": ["ema120_dev", "vwap300_dev"],
    "order_flow": ["ofi15", "ofi60", "ofi300", "svol15", "svol60", "svol300", "cvd_round", "large_net60"],
    "liquidity": ["intensity60", "vol_intensity60"],
    "calendar": ["hour_sin", "hour_cos", "weekend"],
    "time": ["k", "rem"],
}

# Live-only candidates (logged for future validation; NOT in the validated model).
LIVE_ONLY_FEATURES = [
    "book_imb5_cb", "book_imb5_kr", "micro_dev_cb", "xex_cb_kr", "cl_lag_gap", "fut_basis_bps", "liq_net60",
    "pm_mid", "pm_spread", "pm_book_imb",
]
