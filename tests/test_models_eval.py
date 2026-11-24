import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from btc5m.eval import metrics as M
from btc5m.eval.backtest import max_drawdown, simulate
from btc5m.eval.splits import walk_forward
from btc5m.models.baselines import GaussianRW, MonteCarloBootstrap
from btc5m.models.hmm import RegimeHMM
from btc5m.models.registry import gate
from btc5m.rules import FeeSchedule


def test_walk_forward_is_chronological_with_purge():
    starts = np.arange(0, 60 * 86400, 300)
    folds = walk_forward(starts, min_train_days=21, test_days=7, purge_s=3600)
    assert len(folds) >= 4
    prev_test_hi = -1
    for f in folds:
        assert f.train_starts.max() + 300 + 3600 <= f.val_starts.min() + 1e-9 or f.train_starts.max() < f.val_starts.min()
        assert f.val_starts.max() + 300 + 3600 <= f.test_starts.min()
        assert f.test_starts.min() > prev_test_hi
        prev_test_hi = f.test_starts.max()
        assert not set(f.train_starts) & set(f.test_starts)


def test_metrics_basic():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.8, 0.3])
    assert M.brier(np.full(4, 0.5), y) == pytest.approx(0.25)
    assert M.logloss(np.full(4, 0.5), y) == pytest.approx(np.log(2))
    assert M.accuracy(p, y) == 1.0
    assert 0 <= M.ece(p, y) <= 1
    pt, lo, hi = M.day_block_bootstrap(lambda idx: float(np.mean(p[idx])), np.array([0, 0, 1, 1]), n_boot=200)
    assert lo <= pt <= hi


def test_monte_carlo_matches_gaussian_for_gaussian_returns():
    rng = np.random.default_rng(3)
    std_ret = rng.standard_normal(200_000)
    mc = MonteCarloBootstrap(std_ret, n_paths=20000, block=1)
    df = pd.DataFrame({"rem": [240] * 5, "twap_lookback_s": [60] * 5, "z": [-1.0, -0.3, 0.0, 0.5, 1.5], "k": [60] * 5, "y": [0, 1, 0, 1, 1]})
    mc.fit(df)
    p = mc.predict(df)
    assert np.allclose(p, norm.cdf(df["z"]), atol=0.02)
    assert np.allclose(GaussianRW().predict(df), norm.cdf(df["z"]), atol=1e-3)


def test_hmm_filter_is_causal():
    rng = np.random.default_rng(4)
    r = pd.Series(np.concatenate([rng.normal(0, 1e-4, 2000), rng.normal(0, 5e-4, 2000)]))
    h = RegimeHMM(n_states=2).fit(r)
    a = h.filter(r.iloc[:3000])
    b = h.filter(r)
    pd.testing.assert_frame_equal(a, b.iloc[:3000])


def test_backtest_pnl_accounting():
    df = pd.DataFrame({"start": [0, 300], "k": [30, 30], "y": [1, 0], "fill_up_l2": [0.50, 0.50], "fill_dn_l2": [0.50, 0.50],
                       "fee_rate": [0.05, 0.05], "fee_exponent": [1.0, 1.0]})
    tr = simulate(df, np.array([0.9, 0.9]), edge=0.0, latency=2, fee=FeeSchedule(0.05, 1.0), slip=0.01)
    fee = FeeSchedule(0.05, 1.0).taker_fee_per_share(0.51)
    assert list(tr["side"]) == ["up", "up"]
    assert tr["pnl"].iloc[0] == pytest.approx(1 - 0.51 - fee)
    assert tr["pnl"].iloc[1] == pytest.approx(-0.51 - fee)
    assert max_drawdown(np.array([1.0, -2.0, 0.5])) == pytest.approx(2.0)


def test_gate_refuses_weak_model():
    rep = {"oos": {"rounds": 5000}, "metrics": {
        "vs_baseline": {str(k): {"p_lightgbm": {"d_brier": 0.001, "ci95": [-0.001, 0.003]}} for k in (0, 15, 30, 45, 60)},
        "by_checkpoint": {str(k): {"p_lightgbm": {"ece": 0.01}} for k in (0, 15, 30, 45, 60)}}}
    g = gate(rep, "lightgbm")
    assert not g["passed"]
    for k in (0, 15, 30, 45, 60):
        rep["metrics"]["vs_baseline"][str(k)]["p_lightgbm"] = {"d_brier": -0.002, "ci95": [-0.003, -0.001]}
    assert gate(rep, "lightgbm")["passed"]
