"""Settlement and fee rules for Polymarket BTC 5-minute Up/Down markets.

Facts below were verified from live Gamma API market objects and docs.polymarket.com on
2026-11-24 (see RESEARCH.md §1 for evidence):

* Outcome "Up" iff final >= price_to_beat (ties resolve Up).
* Two settlement methodologies exist in the history of the series:
    - ``spot``  : resolutionSource https://data.chain.link/streams/btc-usd (Chainlink Data
                  Streams BTC/USD spot). Observed on markets through at least 2026-07-27.
    - ``twap60``: resolutionSource https://data.chain.link/streams/btc-usd-twap-60s-streams,
                  ``cryptoMarketConfig = {twapEnabled: true, twapLookbackSeconds: 60}``.
                  Observed from at least 2026-08-26 onward. The exact switch date is detected
                  per market by the backfill, never assumed.
* Taker fee: ``fee = C * rate * (p * (1 - p)) ** exponent`` (docs: "fee = C x feeRate x p x (1-p)"
  for exponent 1). Makers pay no fee. The per-market ``feeSchedule`` changed over time
  (rate 0.25/exponent 2 seen in 2026-03, rate 0.01/exponent 1 from at least 2026-05, rate 0.05
  from ~2026-11-23). See RESEARCH.md for the caveat on the exponent-2 era formula.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SPOT = "spot"
TWAP60 = "twap60"


@dataclass(frozen=True)
class FeeSchedule:
    rate: float = 0.05
    exponent: float = 1.0
    taker_only: bool = True

    @classmethod
    def from_market(cls, fs: dict | None) -> "FeeSchedule":
        if not fs:
            return cls()
        return cls(rate=float(fs.get("rate", 0.05)), exponent=float(fs.get("exponent", 1)), taker_only=bool(fs.get("takerOnly", True)))

    def taker_fee_per_share(self, p: float) -> float:
        p = min(max(p, 0.0), 1.0)
        return round(self.rate * (p * (1.0 - p)) ** self.exponent, 5)


def settlement_kind(market: dict) -> str:
    """Classify a Gamma market/event dict into a settlement methodology."""
    cfg = market.get("cryptoMarketConfig") or {}
    src = (market.get("resolutionSource") or "").lower()
    if cfg.get("twapEnabled") or "twap" in src:
        lb = int(cfg.get("twapLookbackSeconds") or 60)
        if lb != 60:
            return f"twap{lb}"
        return TWAP60
    return SPOT


def outcome_up(final: float, price_to_beat: float) -> int:
    """1 if Up wins. Ties resolve Up ("greater than or equal to")."""
    return int(final >= price_to_beat)


def twap_window_bounds(t_end: float, lookback_s: int = 60) -> tuple[float, float]:
    return t_end - lookback_s, t_end


def effective_variance_time(t_now: float, t_end: float, lookback_s: int = 0) -> float:
    """Variance-time (in seconds) of the settlement value given information at t_now, under
    an arithmetic Brownian approximation with constant per-second variance.

    For spot settlement it is simply the time remaining. For a trailing TWAP over
    [t_end - L, t_end] the final value is A = (1/L) * integral of S_u du. Decomposing:

    * if t_now <= t_end - L:  Var(A) = sigma^2 * ((t_end - L - t_now) + L/3)
    * if t_now  > t_end - L:  part of the window is already observed; with r = t_end - t_now
      remaining seconds, the unobserved part contributes Var = sigma^2 * r^3 / (3 L^2)

    (both expressions are continuous at t_now = t_end - L).
    """
    r = max(t_end - t_now, 0.0)
    if lookback_s <= 0:
        return r
    L = float(lookback_s)
    if r >= L:
        return (r - L) + L / 3.0
    return r**3 / (3.0 * L * L)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
