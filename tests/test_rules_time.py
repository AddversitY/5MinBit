import json
import math
from pathlib import Path

import pytest

from btc5m.history.gamma_rounds import parse_event
from btc5m.rules import SPOT, TWAP60, FeeSchedule, effective_variance_time, outcome_up, settlement_kind
from btc5m.timeutil import ROUND_S, round_start, slug_for, start_from_slug

FIX = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "price,fee_100",
    # Polymarket docs fee table (100 shares, rate 0.05): verified 2026-11-24
    [(0.01, 0.05), (0.10, 0.45), (0.30, 1.05), (0.50, 1.25), (0.70, 1.05), (0.95, 0.24), (0.99, 0.05)],
)
def test_fee_matches_documented_table(price, fee_100):
    fs = FeeSchedule(0.05, 1)
    assert round(100 * fs.taker_fee_per_share(price), 2) == pytest.approx(fee_100, abs=0.011)


def test_fee_rate_001_table():
    fs = FeeSchedule(0.01, 1)
    assert round(100 * fs.taker_fee_per_share(0.5), 2) == 0.25
    assert round(100 * fs.taker_fee_per_share(0.1), 2) == 0.09


def test_ties_resolve_up():
    assert outcome_up(100.0, 100.0) == 1
    assert outcome_up(99.99, 100.0) == 0


def test_settlement_kind():
    assert settlement_kind({"cryptoMarketConfig": {"twapEnabled": True, "twapLookbackSeconds": 60}}) == TWAP60
    assert settlement_kind({"resolutionSource": "https://data.chain.link/streams/btc-usd"}) == SPOT
    assert settlement_kind({"cryptoMarketConfig": {"twapEnabled": True, "twapLookbackSeconds": 30}}) == "twap30"


def test_effective_variance_continuity_and_limits():
    end, L = 300.0, 60
    # continuous at t = end - L
    a = effective_variance_time(end - L - 1e-9, end, L)
    b = effective_variance_time(end - L + 1e-9, end, L)
    assert a == pytest.approx(b, rel=1e-6)
    assert effective_variance_time(0, end, L) == pytest.approx(240 + 20)
    assert effective_variance_time(end, end, L) == 0
    assert effective_variance_time(0, end, 0) == 300


def test_effective_variance_matches_simulation():
    import numpy as np

    rng = np.random.default_rng(0)
    h, L = 120, 60
    paths = rng.standard_normal((20000, h)).cumsum(axis=1)
    a = paths[:, -L:].mean(axis=1)
    assert a.var() == pytest.approx(effective_variance_time(0, h, L), rel=0.05)


def test_round_math():
    assert round_start(1795497123.4) == 1795497000
    assert slug_for(1795497000) == "btc-updown-5m-1795497000"
    assert start_from_slug("btc-updown-5m-1795497000") == 1795497000
    with pytest.raises(ValueError):
        slug_for(1795497001)
    assert ROUND_S == 300


def test_parse_resolved_gamma_event():
    e = json.loads((FIX / "gamma_event_resolved.json").read_text())
    r = parse_event(e)
    assert r["start"] == 1795493700 and r["end"] == 1795494000
    assert r["settlement_kind"] == TWAP60 and r["twap_lookback_s"] == 60
    assert r["resolved"] is True
    assert r["outcome_up"] == 1
    assert r["price_to_beat"] == pytest.approx(60335.046536988404)
    assert outcome_up(r["final_price"], r["price_to_beat"]) == r["outcome_up"]
    assert r["token_up"] and r["token_down"] and r["token_up"] != r["token_down"]
    assert math.isclose(r["fee_rate"], 0.05)


def test_unresolved_prices_not_treated_as_resolved():
    e = json.loads((FIX / "gamma_event_resolved.json").read_text())
    e["markets"][0]["outcomePrices"] = '["0.9995", "0.0005"]'
    r = parse_event(e)
    assert r["resolved"] is False and r["outcome_up"] is None
