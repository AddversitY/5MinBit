"""Real-time prediction engine (paper mode only).

Every second: build a point-in-time Snapshot from the live MarketState (Chainlink reference
from RTDS, Kraken trade bars, Coinbase minute candles), compute the same features as the
historical dataset, apply the production model (or the parameter-free Gaussian baseline if
no model has passed the validation gate), attach market prices and data-quality status,
and write one ledger record per (round, checkpoint).

The engine never places orders. "would_trade" fields are hypothetical paper decisions.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import norm

from btc5m.config import load_config
from btc5m.features.compute import LIVE_ONLY_FEATURES, LOOKBACK, Snapshot, compute_features, projected_settlement
from btc5m.features.dataset import DATA_LAG_S
from btc5m.live.candles import CandleCache
from btc5m.live.ledger import Ledger
from btc5m.live.state import STATE, MarketState
from btc5m.logging import get_logger
from btc5m.models.base import clip
from btc5m.models.registry import latest_report, load_production
from btc5m.rules import SPOT, TWAP60, FeeSchedule
from btc5m.timeutil import ROUND_S, round_start

log = get_logger("engine")
BASELINE_VERSION = "baseline-gaussian_rw-v1"


def _ffill_head(ref: np.ndarray, proxy: np.ndarray) -> np.ndarray:
    """Fill missing leading/inner reference values with the exchange proxy."""
    out = ref.copy()
    m = ~np.isfinite(out)
    out[m] = proxy[m]
    return out


def book_imbalance(book, levels: int = 5) -> float:
    b = sum(s for _, s in book.bids[:levels])
    a = sum(s for _, s in book.asks[:levels])
    return (b - a) / (b + a) if (a + b) > 0 else float("nan")


def microprice(book) -> float:
    if not book.bids or not book.asks:
        return float("nan")
    (pb, sb), (pa, sa) = book.bids[0], book.asks[0]
    return (pb * sa + pa * sb) / (sa + sb) if (sa + sb) > 0 else float("nan")


class Engine:
    def __init__(self, state: MarketState = STATE, ledger: Ledger | None = None):
        self.cfg = load_config()
        self.state = state
        self.ledger = ledger or Ledger.default()
        self.checkpoints = [int(x) for x in self.cfg.rounds["checkpoints_s"]]
        self.candles = CandleCache()
        self.latest: dict = {}
        self.series: deque = deque(maxlen=3600)
        self.recorded: set = set()
        self.load_model()
        self._hmm_cache: tuple[int, dict] = (-1, {})
        self.evidence = self._load_evidence()

    # ---- model / evidence ------------------------------------------------------------
    def load_model(self) -> None:
        try:
            self.version, self.family, self.model, self.hmm = load_production()
        except Exception as e:
            log.error("model_load_failed", err=repr(e))
            self.version, self.family, self.model, self.hmm = None, "gaussian_rw", None, None
        self.model_version = self.version or BASELINE_VERSION
        self.validated = self.version is not None

    def _load_evidence(self) -> dict:
        """Per-checkpoint OOS evidence (model vs market) from the latest evaluation report."""
        _, rep = latest_report()
        if not rep:
            return {}
        out = {}
        col = f"p_{self.family}"
        for k, row in (rep.get("metrics", {}).get("vs_market") or {}).items():
            r = row.get(col) or {}
            out[int(k)] = {"n": row.get("n"), "d_brier_vs_market": r.get("d_brier_vs_market"), "ci95": r.get("ci95")}
        return out

    def _predict(self, f: dict) -> tuple[float, dict]:
        """Production probability + alternative model probabilities (for the uncertainty band)."""
        alts = {"gaussian_rw": float(clip(norm.cdf(f["z"])))}
        if self.model is None:
            return alts["gaussian_rw"], alts
        X = pd.DataFrame([f])
        for c in getattr(self.model, "features", []) or []:
            if c not in X:
                X[c] = np.nan
        p = float(self.model.predict(X)[0])
        members = getattr(self.model, "members", None)
        if members:
            for m in members:
                try:
                    alts[m.name] = float(m.predict(X)[0])
                except Exception:
                    pass
        return p, alts

    def _hmm_features(self, td: int) -> dict:
        if self.hmm is None:
            return {}
        minute = td // 60
        if self._hmm_cache[0] == minute:
            return self._hmm_cache[1]
        r = self.candles.minute_returns()
        r = r[r.index + 60 <= td].iloc[-4 * 1440 :]
        if len(r) < 100:
            return {}
        reg = self.hmm.filter(r).iloc[-1].to_dict()
        self._hmm_cache = (minute, reg)
        return reg

    def explain(self, f: dict) -> list[dict]:
        """Top contributions to the current prediction."""
        m = self.model
        lgbm = None
        if m is not None and m.__class__.__name__ == "LGBM":
            lgbm = m
        elif m is not None and getattr(m, "members", None):
            lgbm = next((x for x in m.members if x.__class__.__name__ == "LGBM"), None)
        if lgbm is not None:
            X = pd.DataFrame([{c: f.get(c, np.nan) for c in lgbm.features}])
            contrib = lgbm.m.predict(X, pred_contrib=True, num_iteration=lgbm.m.best_iteration or None)[0][:-1]
            items = sorted(zip(lgbm.features, contrib), key=lambda x: -abs(x[1]))[:8]
            return [{"feature": k, "contribution_logodds": float(v), "value": float(f.get(k, np.nan))} for k, v in items]
        # Gaussian baseline: z decomposition
        return [
            {"feature": "z (vol-adjusted distance to price to beat)", "value": f.get("z"), "contribution_logodds": float(1.7 * f.get("z", 0.0))},
            {"feature": "distance d", "value": f.get("d")},
            {"feature": "sigma (USD/sqrt s)", "value": f.get("sigma")},
            {"feature": "momentum m60", "value": f.get("m60")},
            {"feature": "order flow ofi60", "value": f.get("ofi60")},
        ]

    # ---- live-only features ------------------------------------------------------------
    def live_extras(self, t: float, sigma: float, ref_now: float, pr) -> dict:
        st = self.state
        x: dict = {}
        cb, kr = st.books.get("coinbase"), st.books.get("kraken")
        if cb and cb.bids:
            x["book_imb5_cb"] = book_imbalance(cb)
            mid = 0.5 * (cb.bids[0][0] + cb.asks[0][0])
            x["micro_dev_cb"] = (microprice(cb) - mid) / sigma
        if kr and kr.bids:
            x["book_imb5_kr"] = book_imbalance(kr)
        if cb and kr and cb.bids and kr.bids:
            x["xex_cb_kr"] = (0.5 * (cb.bids[0][0] + cb.asks[0][0]) - 0.5 * (kr.bids[0][0] + kr.asks[0][0])) / sigma
            ex_mid = 0.5 * (cb.bids[0][0] + cb.asks[0][0])
            x["cl_lag_gap"] = (ex_mid - ref_now) / sigma  # exchanges lead Chainlink by ~1-2 s
        fm = st.futures_mark.last()
        if fm and kr and kr.bids:
            x["fut_basis_bps"] = (fm[2] / (0.5 * (kr.bids[0][0] + kr.asks[0][0])) - 1) * 1e4
        liq = [l for l in list(st.liquidations) if l[0] >= t - 60]
        x["liq_net60"] = float(sum(side * size for _, side, size, _ in liq))
        if pr and pr.token_up in st.pm_books:
            ub = st.pm_books[pr.token_up]
            if ub.bids and ub.asks:
                x["pm_mid"] = 0.5 * (ub.bids[0][0] + ub.asks[0][0])
                x["pm_spread"] = ub.asks[0][0] - ub.bids[0][0]
                x["pm_book_imb"] = book_imbalance(ub)
        return x

    # ---- one step ------------------------------------------------------------------------
    def step(self, now: float | None = None) -> dict:
        now = now or time.time()
        t = int(now)
        S = round_start(t)
        k = t - S
        end = S + ROUND_S
        td = t - DATA_LAG_S
        st = self.state
        reasons: list[str] = []
        pr = st.rounds.get(S)
        kind = pr.settlement_kind if pr else TWAP60
        L = (pr.twap_lookback_s if pr else 60) if kind != SPOT else 0
        if pr is None:
            reasons.append("round metadata not yet discovered (assuming TWAP-60 rules)")

        # price to beat: official = Chainlink TWAP-60 (spot for spot rounds) at the start second
        series = st.chainlink_spot if kind == SPOT else st.chainlink_twap
        hit = series.value_at_src(S)
        ptb, ptb_status = None, "unknown"
        if hit and int(hit[0]) == S and hit[1] <= now:
            ptb, ptb_status = hit[2], "observed"
        elif hit:
            ptb, ptb_status = hit[2], "estimated (latest before start)"
            reasons.append("price to beat not yet observed")
        if ptb is None:
            reasons.append("price to beat unavailable")

        # reference series: Chainlink spot as received; gaps filled from a Kraken proxy
        ref_cl = st.chainlink_spot.per_second(td, LOOKBACK, known_by=now)
        kb = st.bars["kraken"].window(td, LOOKBACK)
        kclose = kb["close"]
        ov = np.isfinite(ref_cl) & np.isfinite(kclose)
        basis_live = float(np.median(ref_cl[ov] - kclose[ov])) if ov.sum() > 10 else np.nan
        fill_frac = float(np.mean(~np.isfinite(ref_cl)))
        ref = _ffill_head(ref_cl, kclose + (basis_live if np.isfinite(basis_live) else np.nan))
        if fill_frac > 0.5:
            reasons.append(f"Chainlink history incomplete ({fill_frac:.0%} of last {LOOKBACK}s filled from exchange proxy)")

        cl_last = st.chainlink_spot.last()
        cl_age = (now - cl_last[1]) if cl_last else None
        stale = self.cfg.collector["stale_after_s"]
        if cl_age is None or cl_age > stale["chainlink"]:
            reasons.append("Chainlink feed stale")
        kr_age = now - st.bars["kraken"].last_ts if st.bars["kraken"].last_ts else None
        if kr_age is None or kr_age > 60:
            reasons.append("Kraken trade feed stale")

        self.candles.refresh()
        rv1h, rv4h, vpct = self.candles.vol_at(td)
        out: dict = {"ts": now, "round_start": S, "round_end": end, "k": k, "remaining_s": end - now, "settlement_kind": kind,
                     "twap_lookback_s": L, "price_to_beat": ptb, "price_to_beat_status": ptb_status, "model_version": self.model_version,
                     "model_validated": self.validated, "family": self.family}
        out["chainlink_spot"] = cl_last[2] if cl_last else None
        tw = st.chainlink_twap.last()
        out["chainlink_twap60"] = tw[2] if tw else None
        out["coinbase_last"] = st.bars["coinbase"].last_price if np.isfinite(st.bars["coinbase"].last_price) else None
        out["kraken_last"] = st.bars["kraken"].last_price if np.isfinite(st.bars["kraken"].last_price) else None
        out["chainlink_age_s"] = cl_age

        f: dict = {}
        if ptb is not None and np.isfinite(ref[-1]):
            snap = Snapshot(t=t, start=S, end=end, ptb=ptb, kind=kind, lookback_s=L, ref=ref, bars=kb, rv_1h=rv1h, rv_4h=rv4h, vol_pct_7d=vpct)
            f = compute_features(snap)
            f.update(self._hmm_features(td))
            sig = f.get("sigma", np.nan)
            if np.isfinite(sig) and sig > 0:
                f.update(self.live_extras(now, sig, float(ref[-1]), pr))
        p_up = None
        if f and np.isfinite(f.get("z", np.nan)):
            p_up, alts = self._predict(f)
            # proxy-reference variant (training-consistent input) for skew diagnostics
            if ptb is not None and np.isfinite(kclose[-1]):
                base_s = td - LOOKBACK  # second of kclose[0]
                i1 = S - base_s
                ka = kclose[max(i1 - L, 0) : i1] if L else kclose[i1 - 1 : i1]
                basis_proxy = ptb - float(np.nanmean(ka)) if len(ka) else np.nan
                fp = compute_features(Snapshot(t=t, start=S, end=end, ptb=ptb, kind=kind, lookback_s=L, ref=kclose + basis_proxy, bars=kb, rv_1h=rv1h, rv_4h=rv4h, vol_pct_7d=vpct))
                if np.isfinite(fp.get("z", np.nan)):
                    fp.update(self._hmm_features(td))
                    alts["proxy_ref"] = self._predict(fp)[0]
            exp_settle, vt = projected_settlement(ref, t, end, kind, L)
            sig = f["sigma"]
            half = 1.645 * sig * math.sqrt(max(vt, 0.0))
            out.update({"p_up": p_up, "p_down": 1 - p_up, "alt_probs": alts, "p_band": [min(alts.values()), max(alts.values())],
                        "projected_settlement": exp_settle, "projected_interval90": [exp_settle - half, exp_settle + half],
                        "sigma_usd_per_sqrt_s": sig, "vol_annualized_pct": sig / exp_settle * math.sqrt(365 * 86400) * 100,
                        "z": f["z"], "buy_pressure_ofi60": f.get("ofi60"), "cvd_round": f.get("cvd_round"), "vol_pct_7d": vpct})
            if abs(f["z"]) > self.cfg.live["max_abs_z"]:
                reasons.append("outside validated range (|z| too large)")
            if np.isfinite(vpct) and vpct > self.cfg.live["max_vol_percentile"]:
                reasons.append("volatility above validated range")
            if not np.isfinite(rv1h):
                reasons.append("long-horizon volatility unavailable")
        else:
            reasons.append("features unavailable")

        # market data
        mkt = {}
        if pr and pr.token_up:
            for leg, tok in (("up", pr.token_up), ("down", pr.token_down)):
                b = st.pm_books.get(tok)
                if b and (b.bids or b.asks):
                    mkt[f"{leg}_bid"] = b.bids[0][0] if b.bids else None
                    mkt[f"{leg}_ask"] = b.asks[0][0] if b.asks else None
                    mkt[f"{leg}_bid_size"] = b.bids[0][1] if b.bids else None
                    mkt[f"{leg}_ask_size"] = b.asks[0][1] if b.asks else None
                    mkt[f"{leg}_book_age_s"] = now - b.rx
            if mkt.get("up_bid") is not None and mkt.get("up_ask") is not None:
                mkt["implied_p_up_mid"] = 0.5 * (mkt["up_bid"] + mkt["up_ask"])
            if (mkt.get("up_book_age_s") or 1e9) > stale["polymarket_book"] * 4:
                reasons.append("Polymarket book stale")
        else:
            reasons.append("Polymarket market not discovered")
        out["market"] = mkt
        fee = FeeSchedule(pr.fee_rate, pr.fee_exponent) if pr else FeeSchedule()
        out["fee_schedule"] = {"rate": fee.rate, "exponent": fee.exponent}
        if p_up is not None:
            ev = {}
            slip = self.cfg.costs["tick"] * self.cfg.costs["extra_slippage_ticks"]
            if mkt.get("up_ask"):
                c = min(mkt["up_ask"] + slip, 0.999)
                ev["up"] = p_up - c - fee.taker_fee_per_share(c)
            if mkt.get("down_ask"):
                c = min(mkt["down_ask"] + slip, 0.999)
                ev["down"] = (1 - p_up) - c - fee.taker_fee_per_share(c)
            out["expected_value_per_share_after_costs"] = ev
        # evidence gate: is there validated out-of-sample evidence that this model beats the market here?
        ck = max([c for c in self.checkpoints if c <= k], default=0)
        evd = self.evidence.get(ck)
        out["evidence"] = evd
        if not self.validated:
            reasons.append("no model has passed the validation gate (showing parameter-free baseline)")
        if not evd or evd.get("ci95") is None or evd["ci95"][1] >= 0:
            reasons.append("no statistically significant out-of-sample edge vs market at this checkpoint")
        if p_up is not None and mkt.get("implied_p_up_mid") is not None:
            lo, hi = out["p_band"]
            if lo <= mkt["implied_p_up_mid"] <= hi:
                reasons.append("market price lies inside the model uncertainty band")
        out["reasons"] = reasons
        data_ok = not any(("stale" in r) or ("unavailable" in r) or ("incomplete" in r) or ("not yet observed" in r) for r in reasons)
        out["data_status"] = "OK" if data_ok else "DEGRADED"
        out["status"] = "INSUFFICIENT EVIDENCE" if reasons else "OK"
        out["explain"] = self.explain(f) if p_up is not None else []
        out["features"] = {kk: (float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else None) for kk, v in f.items()}
        out["health"] = st.health_snapshot()
        self.latest = out
        self.series.append({"ts": now, "chainlink": out["chainlink_spot"], "twap60": out["chainlink_twap60"], "coinbase": out["coinbase_last"],
                            "kraken": out["kraken_last"], "ptb": ptb, "round_start": S, "p_up": p_up, "pm_mid": mkt.get("implied_p_up_mid"),
                            "proj": out.get("projected_settlement"), "proj_lo": (out.get("projected_interval90") or [None, None])[0],
                            "proj_hi": (out.get("projected_interval90") or [None, None])[1]})
        self._maybe_record(out, f)
        return out

    def _maybe_record(self, out: dict, f: dict) -> None:
        k = out["k"]
        for c in self.checkpoints:
            key = (out["round_start"], c)
            if key in self.recorded or not (c <= k < c + 5):
                continue
            if out.get("p_up") is None:
                continue
            ev = out.get("expected_value_per_share_after_costs") or {}
            side = max(ev, key=ev.get) if ev else None
            would = side if side and ev[side] > self.cfg.live["min_edge_after_costs"] else None
            mkt = out["market"]
            rec = {
                "round_start": out["round_start"], "checkpoint": c, "k_actual": k, "pred_ts": out["ts"], "model_version": out["model_version"],
                "model_validated": out["model_validated"], "p_up": out["p_up"], "p_down": out["p_down"], "alt_probs": out["alt_probs"],
                "price_to_beat": out["price_to_beat"], "price_to_beat_status": out["price_to_beat_status"], "chainlink_spot": out["chainlink_spot"],
                "market": mkt, "fee_schedule": out["fee_schedule"], "ev_after_costs": ev, "would_trade": would,
                "would_trade_price": (min(mkt.get(f"{would}_ask", 0) + self.cfg.costs["tick"] * self.cfg.costs["extra_slippage_ticks"], 0.999) if would else None),
                "status": out["status"], "data_status": out["data_status"], "reasons": out["reasons"],
                "features": {kk: v for kk, v in out["features"].items() if kk in set(LIVE_ONLY_FEATURES) | {"z", "d", "sigma", "m60", "m15", "ofi60", "cvd_round", "vol_pct_7d", "hmm_vol"}},
                "chainlink_age_s": out["chainlink_age_s"],
            }
            try:
                self.ledger.add_prediction(rec)
            except Exception as e:
                log.error("ledger_write_failed", err=repr(e))
            self.recorded.add(key)
        if len(self.recorded) > 5000:
            cutoff = out["round_start"] - 3600
            self.recorded = {x for x in self.recorded if x[0] >= cutoff}

    async def run(self) -> None:
        await asyncio.sleep(3)
        while True:
            t0 = time.time()
            try:
                await asyncio.to_thread(self.step, t0)
            except Exception as e:
                log.error("engine_step_failed", err=repr(e))
            await asyncio.sleep(max(0.05, 1.0 - (time.time() - t0) - (time.time() % 1.0) * 0))


__all__ = ["Engine", "json"]
