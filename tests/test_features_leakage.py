"""Point-in-time correctness: features at decision time t must not change when data at or
after t (minus the data lag) is altered."""
import numpy as np
import pandas as pd
import pytest

from btc5m.features import dataset as D
from btc5m.features.bars import BarFrame
from btc5m.features.compute import LOOKBACK, Snapshot, compute_features, projected_settlement
from btc5m.features.market_implied import fill_price, implied_at, prepare_round_trades
from btc5m.rules import TWAP60


def _synthetic_trades(t0, n_sec, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.sort(t0 + rng.uniform(0, n_sec, n_sec * 2))
    price = 60000 + np.cumsum(rng.normal(0, 3, len(ts)))
    size = rng.exponential(0.05, len(ts))
    side = rng.choice([-1, 1], len(ts))
    return ts, price, size, side


def _bars(ts, price, size, side, t0, t1):
    bf = BarFrame.from_trades(ts, price, size, side, t0, t1)
    bf.day_covered = np.ones(bf.n // 86400 + 2, dtype=bool)
    return bf


def _vc():
    idx = np.arange(0, 2_000_000_000, 10**9)
    return pd.DataFrame({"rv_1h": [5.0, 5.0], "rv_4h": [5.0, 5.0], "vol_pct_7d": [0.5, 0.5]}, index=idx)


def _round(start):
    return {"slug": f"btc-updown-5m-{start}", "start": start, "end": start + 300, "price_to_beat": 60000.0, "settlement_kind": TWAP60,
            "twap_lookback_s": 60, "outcome_up": 1, "final_price": 60010.0, "condition_id": "c", "token_up": "U", "fee_rate": 0.05, "fee_exponent": 1.0, "volume": 1.0}


@pytest.mark.parametrize("k", [0, 15, 60, 240])
def test_dataset_rows_ignore_future_trades(k):
    t0 = 1_795_000_000 - 3000
    start = t0 + 2000 - (t0 + 2000) % 300
    ts, price, size, side = _synthetic_trades(t0, 3000)
    bf = _bars(ts, price, size, side, t0, t0 + 3000)
    D._G.update({"bars": bf, "volctx": _vc(), "checkpoints": [k], "pm": {}})
    a = D._round_rows(_round(start))[0]
    # perturb every trade at or after the information horizon. Exchange features use data up
    # to t - DATA_LAG_S; the per-round basis is anchored on the official price to beat, which
    # is only observable after the round start, so k=0 is defined as "the instant the price
    # to beat is first observable" and its horizon is the round start itself.
    horizon = max(start + k - D.DATA_LAG_S, start)
    p2 = price.copy()
    p2[ts >= horizon] += 500.0
    side2 = np.where(ts >= horizon, -side, side)
    bf2 = _bars(ts, p2, size * (1 + (ts >= horizon)), side2, t0, t0 + 3000)
    D._G.update({"bars": bf2})
    b = D._round_rows(_round(start))[0]
    for key in ("z", "d", "m15", "m60", "ofi60", "svol60", "rv300", "cvd_round", "vwap300_dev"):
        assert a[key] == pytest.approx(b[key], nan_ok=True), key
    # sanity: the label-side diagnostic DOES change (uses data up to the round end)
    if k < 240:
        assert a["proxy_final_err"] != b["proxy_final_err"]


def test_projected_settlement_twap_partial_window():
    ref = np.full(LOOKBACK, 100.0)
    ref[-30:] = 110.0  # last 30 observed seconds inside the TWAP window
    t, end = 1000 + 270, 1000 + 300  # 30 s into the final 60 s window
    exp, vt = projected_settlement(ref, t, end, TWAP60, 60)
    assert exp == pytest.approx((30 * 110 + 110 * 30) / 60)
    assert vt == pytest.approx(30**3 / (3 * 60**2))


def test_compute_features_scale_free():
    rng = np.random.default_rng(1)
    ref = 60000 + np.cumsum(rng.normal(0, 4, LOOKBACK))
    bars = {"close": ref, "vol": np.ones(LOOKBACK), "buy": np.ones(LOOKBACK) * 0.6, "sell": np.ones(LOOKBACK) * 0.4,
            "ntr": np.ones(LOOKBACK), "pv": ref, "maxsz": np.ones(LOOKBACK)}
    s = Snapshot(t=10_000, start=10_000 - 30, end=10_270, ptb=float(ref[-31]), kind=TWAP60, lookback_s=60, ref=ref, bars=bars, rv_1h=4.0, rv_4h=4.0)
    f1 = compute_features(s)
    s2 = Snapshot(t=10_000, start=10_000 - 30, end=10_270, ptb=float(ref[-31]) * 2, kind=TWAP60, lookback_s=60, ref=ref * 2,
                  bars={**bars, "close": ref * 2, "pv": ref * 2}, rv_1h=8.0, rv_4h=8.0)
    f2 = compute_features(s2)
    for key in ("z", "d", "m60", "ofi60", "open_gap"):
        assert f1[key] == pytest.approx(f2[key], rel=1e-6)
    assert f1["ofi60"] == pytest.approx(0.2)


def test_market_implied_uses_only_past_trades():
    tr = pd.DataFrame({
        "token_id": ["U", "D", "U", "U"], "side": ["BUY", "BUY", "SELL", "BUY"],
        "price": [0.55, 0.47, 0.52, 0.90], "size": [10, 10, 10, 10], "ts": [100, 101, 102, 110],
    })
    rt = prepare_round_trades(tr, "U")
    m = implied_at(rt, 103)
    assert m["pm_ask"] == pytest.approx(0.55)
    assert m["pm_bid"] == pytest.approx(0.52)  # latest bid evidence (SELL Up at 0.52)
    assert m["pm_mid"] == pytest.approx(0.535)
    # the 0.90 print at t=110 is in the future and must not appear
    assert m["pm_last"] == pytest.approx(0.52)
    assert fill_price(rt, 103, True, latency=2) == pytest.approx(0.90)
    assert np.isnan(fill_price(rt, 103, False, latency=2))
