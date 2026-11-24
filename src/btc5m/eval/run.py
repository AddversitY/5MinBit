"""Walk-forward evaluation of all models -> JSON report + MODEL_RESULTS.md tables.

Everything reported as "test" is strictly out-of-sample: each prediction comes from a
model fitted only on rounds that ended at least ``purge_s`` before the test block began.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from btc5m.config import load_config
from btc5m.eval import metrics as M
from btc5m.eval.backtest import simulate, summarize_trades
from btc5m.eval.splits import take, walk_forward
from btc5m.features.compute import CORE_FEATURES, FEATURE_GROUPS
from btc5m.features.dataset import load_kraken_bars
from btc5m.logging import get_logger
from btc5m.models.base import clip, logit
from btc5m.models.baselines import Coin, Constant, GaussianRW, MonteCarloBootstrap, PlattVol, standardized_returns
from btc5m.models.hmm import RegimeHMM, attach_regimes
from btc5m.models.ml import LGBM, XGB, Ensemble, LGBMResid, LogReg
from btc5m.rules import TWAP60, FeeSchedule
from btc5m.storage.store import data_dir

log = get_logger("eval")
HMM_FEATURES = ["hmm_p0", "hmm_p1", "hmm_p2", "hmm_vol"]
# Bootstrap block length: rounds are resampled in contiguous 4-hour blocks, which preserves
# short-range dependence (volatility clustering) while giving enough blocks for stable CIs.
BLOCK_S = 4 * 3600


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    path = path or data_dir() / "datasets" / "checkpoints_kraken.parquet"
    df = pd.read_parquet(path)
    df = df[np.isfinite(df["z"]) & np.isfinite(df["sigma"]) & (df["sigma"] > 0)].copy()
    if "lp_gauss" not in df:
        p = df["p_gauss"].clip(1e-4, 1 - 1e-4)
        df["lp_gauss"] = np.log(p / (1 - p))
    df["day"] = df["start"] // BLOCK_S  # bootstrap block id (4 h), see BLOCK_S
    return df.sort_values(["start", "k"]).reset_index(drop=True)


def minute_returns() -> pd.Series:
    c = pd.read_parquet(data_dir() / "hist" / "coinbase_candles_1m" / "candles.parquet").sort_values("ts").set_index("ts")["close"]
    full = c.reindex(np.arange(int(c.index.min()), int(c.index.max()) + 60, 60)).ffill()
    return np.log(full).diff().dropna()


def model_factory(std_ret: np.ndarray | None, quick: bool = False) -> dict:
    core = [f for f in CORE_FEATURES]
    core_h = core + HMM_FEATURES
    models = {
        "coin_flip": lambda: Coin(),
        "constant": lambda: Constant(),
        "gaussian_rw": lambda: GaussianRW(),
        "vol_platt": lambda: PlattVol(),
        "logreg": lambda: LogReg(core),
        "logreg_hmm": lambda: LogReg(core_h),
        "lightgbm": lambda: LGBM(core_h),
        "lgbm_resid": lambda: LGBMResid(core_h),
        "xgboost": lambda: XGB(core_h),
        # equal-weight log-odds average: stacking weights fitted on the validation block were
        # biased because LogReg refits on train+val (in-sample val predictions)
        "ensemble": lambda: Ensemble([PlattVol(), LogReg(core_h), LGBMResid(core_h)], stack=False),
    }
    if std_ret is not None:
        models["monte_carlo"] = lambda: MonteCarloBootstrap(std_ret, n_paths=5000 if quick else 20000)
    return models


def _std_returns_for(bars, t_lo: int, t_hi: int) -> np.ndarray:
    lo, hi = max(t_lo - bars.t0, 0), min(t_hi - bars.t0, bars.n)
    if hi - lo < 3600:
        return None
    return standardized_returns(bars.a["close"][lo:hi])


def run_walk_forward(df: pd.DataFrame, quick: bool = False, min_train_days: float = 21, test_days: float = 7, models: list[str] | None = None, bars=None, mret=None):
    folds = walk_forward(df["start"].to_numpy(), min_train_days=min_train_days, test_days=test_days)
    if not folds:
        raise RuntimeError("not enough data for a single walk-forward fold")
    mret = mret if mret is not None else minute_returns()
    preds, fold_info, importances, lr_coefs, train_perf = [], [], [], [], []
    for fold in folds:
        t_fit_lo, t_fit_hi = int(fold.train_starts.min()), int(fold.val_starts.max()) + 300
        # regimes: HMM fitted on training-period minutes only; filtered (causal) probs everywhere
        hmm = RegimeHMM().fit(mret[(mret.index >= t_fit_lo - 7 * 86400) & (mret.index < t_fit_hi)])
        reg = hmm.filter(mret)
        d = attach_regimes(df[df["start"] <= fold.test_starts.max()], reg)
        tr, va, te = take(d, fold.train_starts), take(d, fold.val_starts), take(d, fold.test_starts)
        std_ret = _std_returns_for(bars, t_fit_lo, t_fit_hi) if bars is not None else None
        fac = model_factory(std_ret, quick)
        names = models or list(fac)
        out = te[["start", "k", "y", "day", "vol_pct_7d", "rv_1h", "settlement_kind"] + [c for c in te.columns if c.startswith(("pm_", "fill_"))] + ["fee_rate", "fee_exponent"]].copy()
        out["fold"] = fold.i
        for name in names:
            if name not in fac:
                continue
            t1 = time.time()
            m = fac[name]().fit(tr, va)
            out[f"p_{name}"] = m.predict(te)
            if name in ("lightgbm", "lgbm_resid"):
                if name == "lightgbm":
                    importances.append(m.importance())
                ptr = m.predict(tr)
                train_perf.append({"fold": fold.i, "model": name, **{k: v for k, v in M.summary(ptr, tr["y"].to_numpy()).items() if k in ("n", "accuracy", "brier", "logloss")}})
            if name == "logreg":
                lr_coefs.append(m.coefficients())
                ptr = m.predict(tr)
                train_perf.append({"fold": fold.i, "model": name, **{k: v for k, v in M.summary(ptr, tr["y"].to_numpy()).items() if k in ("n", "accuracy", "brier", "logloss")}})
            log.info("fold_model", fold=fold.i, model=name, secs=round(time.time() - t1, 1), test_rows=len(te))
        # market-stacking: does the model add information beyond the market price?
        if "pm_mid" in te.columns and "p_lightgbm" in out.columns:
            out["p_stack_mkt"] = _stack_with_market(tr, va, te, out, fac)
        info = fold.describe()
        info["hmm"] = hmm.params()
        fold_info.append(info)
        preds.append(out)
    return pd.concat(preds, ignore_index=True), fold_info, importances, lr_coefs, train_perf


def _stack_with_market(tr, va, te, out, fac):
    """Logistic stack of [logit(pm_mid), logit(p_model)] fitted on validation rows."""
    from sklearn.linear_model import LogisticRegression

    v = va[np.isfinite(va["pm_mid"])]
    if len(v) < 500:
        return np.full(len(te), np.nan)
    base = fac["vol_platt"]().fit(tr, None)
    Xv = np.column_stack([logit(v["pm_mid"].clip(0.01, 0.99)), logit(base.predict(v)), v["k"] / 300.0])
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xv, v["y"])
    res = np.full(len(te), np.nan)
    ok = np.isfinite(te["pm_mid"]).to_numpy()
    Xt = np.column_stack([logit(te.loc[ok, "pm_mid"].clip(0.01, 0.99)), logit(base.predict(te[ok])), te.loc[ok, "k"] / 300.0])
    res[ok] = lr.predict_proba(Xt)[:, 1]
    return res


def evaluate_predictions(P: pd.DataFrame, model_cols: list[str]) -> dict:
    res: dict = {"by_checkpoint": {}, "vs_baseline": {}, "vs_market": {}, "by_vol_regime": {}, "calibration": {}, "by_month": {}}
    y_all = P["y"].to_numpy()
    for k, g in P.groupby("k"):
        y = g["y"].to_numpy()
        res["by_checkpoint"][int(k)] = {m: M.summary(g[m].to_numpy(), y) for m in model_cols}
        # Brier difference vs Gaussian random walk, with day-block bootstrap CI
        days = g["day"].to_numpy()
        base = g["p_gaussian_rw"].to_numpy()
        res["vs_baseline"][int(k)] = {}
        for m in model_cols:
            if m == "p_gaussian_rw":
                continue
            pm = g[m].to_numpy()
            pt, lo, hi = M.day_block_bootstrap(lambda idx: M.brier(pm[idx], y[idx]) - M.brier(base[idx], y[idx]), days, n_boot=400)
            res["vs_baseline"][int(k)][m] = {"d_brier": pt, "ci95": [lo, hi]}
        # versus market-implied probability where available
        if "pm_mid" in g.columns:
            h = g[np.isfinite(g["pm_mid"])]
            if len(h) > 200:
                yh = h["y"].to_numpy()
                mk = h["pm_mid"].clip(0.005, 0.995).to_numpy()
                dh = h["day"].to_numpy()
                r = {"n": int(len(h)), "market": M.summary(mk, yh)}
                for m in model_cols + (["p_stack_mkt"] if "p_stack_mkt" in h.columns else []):
                    # p_stack_mkt is undefined in folds whose validation window had too few market
                    # prices; compare it with the market on the rows where it exists.
                    ok = np.isfinite(h[m].to_numpy())
                    if ok.sum() < 200:
                        continue
                    pm, ym, mm, dm = h[m].to_numpy()[ok], yh[ok], mk[ok], dh[ok]
                    pt, lo, hi = M.day_block_bootstrap(lambda idx: M.brier(pm[idx], ym[idx]) - M.brier(mm[idx], ym[idx]), dm, n_boot=400)
                    r[m] = {**M.summary(pm, ym), "d_brier_vs_market": pt, "ci95": [lo, hi], "n_rows": int(ok.sum())}
                res["vs_market"][int(k)] = r
        for m in ("p_gaussian_rw", "p_lightgbm", "p_ensemble", "p_vol_platt"):
            if m in g.columns:
                res["calibration"].setdefault(int(k), {})[m] = M.reliability(g[m].to_numpy(), y)
    # volatility regimes from the trailing 7-day percentile (known at decision time)
    vr = pd.cut(P["vol_pct_7d"], [-0.01, 1 / 3, 2 / 3, 1.01], labels=["low", "mid", "high"])
    for (k, reg), g in P.groupby([P["k"], vr], observed=True):
        res["by_vol_regime"].setdefault(int(k), {})[str(reg)] = {m: {kk: v for kk, v in M.summary(g[m].to_numpy(), g["y"].to_numpy()).items() if kk in ("n", "accuracy", "brier", "logloss")} for m in model_cols}
    mon = pd.to_datetime(P["start"], unit="s").dt.strftime("%Y-%m")
    for (mo, k), g in P.groupby([mon, P["k"]]):
        if int(k) in (0, 30, 60):
            res["by_month"].setdefault(mo, {})[int(k)] = {m: M.brier(g[m], g["y"]) for m in model_cols} | {"n": int(len(g))}
    res["overall"] = {m: M.summary(P[m].to_numpy(), y_all) for m in model_cols}
    return res


def encompassing_test(P: pd.DataFrame, model_cols: list[str], n_boot: int = 300) -> dict:
    """Forecast-encompassing test vs the market, per checkpoint.

    Logistic regression y ~ a + b1*logit(market) + b2*(logit(model) - logit(market)).
    b2 > 0 with a CI excluding 0 means the model carries information NOT already in the
    market price (a necessary, not sufficient, condition for a trading edge). b1 ~ 1 and
    b2 ~ 0 means the market encompasses the model. CI: 4-hour-block bootstrap.
    """
    from sklearn.linear_model import LogisticRegression

    out = {}
    if "pm_mid" not in P.columns:
        return out
    for k, g in P.groupby("k"):
        h = g[np.isfinite(g["pm_mid"])]
        if len(h) < 300:
            continue
        y = h["y"].to_numpy()
        lm = logit(h["pm_mid"].clip(0.01, 0.99).to_numpy())
        blocks = h["day"].to_numpy()
        res = {"n": int(len(h))}
        for m in model_cols:
            gap = logit(h[m].to_numpy()) - lm
            X = np.column_stack([lm, gap])

            def fit(idx):
                lr = LogisticRegression(C=1e4, max_iter=2000).fit(X[idx], y[idx])
                return lr.coef_[0]

            b = fit(np.arange(len(y)))
            rng = np.random.default_rng(1)
            uniq, inv = np.unique(blocks, return_inverse=True)
            members = [np.where(inv == j)[0] for j in range(len(uniq))]
            boots = []
            for _ in range(n_boot):
                idx = np.concatenate([members[j] for j in rng.integers(0, len(uniq), len(uniq))])
                boots.append(fit(idx))
            boots = np.array(boots)
            lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)
            res[m] = {"b_market": float(b[0]), "b_market_ci": [float(lo[0]), float(hi[0])], "b_gap": float(b[1]), "b_gap_ci": [float(lo[1]), float(hi[1])],
                      "model_adds_info": bool(lo[1] > 0)}
        out[int(k)] = res
    return out


def market_calibration(P: pd.DataFrame) -> dict:
    """Is Polymarket's mid itself calibrated? Reliability by price bucket with block-bootstrap
    CIs, plus a pre-registered (fixed-threshold, exploratory) favourite-buying test."""
    out = {"reliability": [], "favourite_test": {}}
    if "pm_mid" not in P.columns:
        return out
    h = P[np.isfinite(P["pm_mid"])].reset_index(drop=True)
    edges = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001]
    b = np.digitize(h["pm_mid"], edges) - 1
    y, pm, blocks = h["y"].to_numpy(), h["pm_mid"].to_numpy(), h["day"].to_numpy()
    for i in range(len(edges) - 1):
        m = b == i
        if m.sum() < 50:
            continue
        idx = np.where(m)[0]
        pt, lo, hi = M.day_block_bootstrap(lambda j: float(np.mean(y[idx][j] - pm[idx][j])), blocks[idx], n_boot=400)
        out["reliability"].append({"bucket": f"{edges[i]:.2f}-{min(edges[i + 1], 1):.2f}", "n": int(m.sum()), "mean_mid": float(pm[m].mean()),
                                   "freq_up": float(y[m].mean()), "freq_minus_mid": pt, "ci95": [lo, hi]})
    if "fill_up_l2" in h.columns:
        cur = FeeSchedule(0.05, 1.0)
        for thr in (0.80, 0.90, 0.95):
            for k, g in h.groupby("k"):
                g = g.reset_index(drop=True)
                fav_up = g["pm_mid"].to_numpy() >= thr
                fav_dn = g["pm_mid"].to_numpy() <= 1 - thr
                sel = fav_up | fav_dn
                if not sel.any():
                    continue
                gg = g[sel].copy()
                # only the favourite leg may trade: blank the other leg's executable price
                gg.loc[~fav_up[sel], "fill_up_l2"] = np.nan
                gg.loc[~fav_dn[sel], "fill_dn_l2"] = np.nan
                p = np.where(fav_up[sel], 1.0, 0.0)
                tr = simulate(gg, p, edge=-1.0, latency=2, fee=cur)
                out["favourite_test"].setdefault(f"mid>={thr:.2f}", {})[int(k)] = summarize_trades(tr, int(g["start"].nunique()))
    return out


def trading_eval(P: pd.DataFrame, model_cols: list[str]) -> dict:
    out = {}
    if "fill_up_l2" not in P.columns:
        return out
    cur_fee = FeeSchedule(0.05, 1.0)
    for k, g in P.groupby("k"):
        g = g.reset_index(drop=True)
        n = g["start"].nunique()
        for m in model_cols + ["pm_mid_as_model"]:
            if m == "pm_mid_as_model":
                if "pm_mid" not in g.columns:
                    continue
                p = g["pm_mid"].fillna(0.5).to_numpy()
            else:
                p = g[m].to_numpy()
            for edge in (0.0, 0.02, 0.05):
                for fee_mode in ("current_0.05", "historical"):
                    tr = simulate(g, p, edge=edge, latency=2, fee=cur_fee, use_market_fee=(fee_mode == "historical"))
                    out.setdefault(int(k), {}).setdefault(m, {})[f"edge{edge}_{fee_mode}"] = summarize_trades(tr, n)
            # stress: 5 s latency, 2-tick slippage, current fee
            tr = simulate(g, p, edge=0.02, latency=5, fee=cur_fee, slip=0.02)
            out[int(k)][m]["stress_lat5_slip2_edge0.02"] = summarize_trades(tr, n)
    return out


def ablation(df: pd.DataFrame, bars=None, mret=None, n_folds: int = 4) -> dict:
    """Feature-group value measured as improvement over the Gaussian base margin.

    Uses the residual LightGBM (boosting from the Gaussian log-odds), so ``gaussian_only`` is
    the zero-feature reference, ``dist+<group>`` adds one group to the distance/time features,
    and ``drop_<group>`` removes one group from the full set. Last ``n_folds`` walk-forward folds.
    """
    folds = walk_forward(df["start"].to_numpy())[-n_folds:]
    mret = mret if mret is not None else minute_returns()
    base_feats = CORE_FEATURES + HMM_FEATURES
    groups = dict(FEATURE_GROUPS)
    groups["regime_hmm"] = HMM_FEATURES
    variants = {"all": base_feats}
    for gname, cols in groups.items():
        variants[f"drop_{gname}"] = [c for c in base_feats if c not in cols]
        if gname not in ("distance", "time"):
            variants[f"dist+{gname}"] = sorted(set(FEATURE_GROUPS["distance"] + FEATURE_GROUPS["time"] + cols))
    variants["dist_only"] = FEATURE_GROUPS["distance"] + FEATURE_GROUPS["time"]
    res = {v: [] for v in list(variants) + ["gaussian_only"]}
    for fold in folds:
        hmm = RegimeHMM().fit(mret[(mret.index >= int(fold.train_starts.min()) - 7 * 86400) & (mret.index < int(fold.val_starts.max()) + 300)])
        d = attach_regimes(df[df["start"] <= fold.test_starts.max()], hmm.filter(mret))
        tr, va, te = take(d, fold.train_starts), take(d, fold.val_starts), take(d, fold.test_starts)
        early = te["k"].to_numpy() <= 60
        y = te["y"].to_numpy()
        preds = {"gaussian_only": GaussianRW().predict(te)}
        for v, feats in variants.items():
            preds[v] = LGBMResid(feats).fit(tr, va).predict(te)
        for v, p in preds.items():
            res[v].append({"fold": fold.i, "n": len(te), "logloss": M.logloss(p, y), "brier": M.brier(p, y), "early_logloss": M.logloss(p[early], y[early])})
    summ = {}
    for v, rows in res.items():
        if rows:
            w = np.array([r["n"] for r in rows], float)
            summ[v] = {
                "logloss": float(np.average([r["logloss"] for r in rows], weights=w)),
                "brier": float(np.average([r["brier"] for r in rows], weights=w)),
                "early_logloss_k<=60": float(np.average([r["early_logloss"] for r in rows], weights=w)),
            }
    base = summ.get("all", {})
    for v in summ:
        summ[v]["d_logloss_vs_all"] = summ[v]["logloss"] - base.get("logloss", np.nan)
        summ[v]["d_early_logloss_vs_all"] = summ[v]["early_logloss_k<=60"] - base.get("early_logloss_k<=60", np.nan)
    return summ


def era_comparison(df_all: pd.DataFrame) -> dict:
    """Parameter-free Gaussian model by settlement methodology (no training involved)."""
    out = {}
    for (kind, k), g in df_all.groupby(["settlement_kind", "k"]):
        p = clip(g["p_gauss"].to_numpy())
        out.setdefault(kind, {})[int(k)] = {kk: v for kk, v in M.summary(p, g["y"].to_numpy()).items() if kk in ("n", "accuracy", "brier", "logloss", "ece")}
    return out


def proxy_quality(df_all: pd.DataFrame) -> dict:
    r = df_all.drop_duplicates("start")
    out = {}
    for kind, g in r.groupby("settlement_kind"):
        e = g["proxy_final_err"].dropna()
        out[kind] = {"rounds": int(len(g)), "outcome_agreement": float(g["proxy_outcome_agree"].mean()),
                     "final_err_mae": float(e.abs().mean()) if len(e) else None, "final_err_sd": float(e.std()) if len(e) else None,
                     "final_err_p95_abs": float(e.abs().quantile(0.95)) if len(e) else None}
    return out


def run_evaluation(quick: bool = False, dataset: Path | None = None) -> Path:
    cfg = load_config()
    t0 = time.time()
    df_all = load_dataset(dataset)
    df = df_all[df_all["settlement_kind"] == TWAP60].reset_index(drop=True)
    bars = load_kraken_bars()
    mret = minute_returns()
    min_train = 7 if quick else 21
    test_days = 3 if quick else 7
    P, folds, imps, coefs, train_perf = run_walk_forward(df, quick=quick, min_train_days=min_train, test_days=test_days, bars=bars, mret=mret)
    model_cols = [c for c in P.columns if c.startswith("p_") and c != "p_stack_mkt"]
    report = {
        "generated_at": time.time(), "quick": quick,
        "dataset": {"rows": int(len(df_all)), "rounds": int(df_all["start"].nunique()), "twap60_rounds": int(df["start"].nunique()),
                    "first": int(df_all["start"].min()), "last": int(df_all["start"].max())},
        "oos": {"rows": int(len(P)), "rounds": int(P["start"].nunique()), "first": int(P["start"].min()), "last": int(P["start"].max())},
        "folds": folds,
        "proxy_quality": proxy_quality(df_all),
        "era_gaussian": era_comparison(df_all),
        "metrics": evaluate_predictions(P, model_cols),
        "training_performance": train_perf,
        "encompassing": encompassing_test(P, [c for c in ("p_gaussian_rw", "p_vol_platt", "p_logreg", "p_lgbm_resid", "p_ensemble") if c in P.columns]),
        "market_calibration": market_calibration(P),
        "trading": trading_eval(P, ["p_gaussian_rw", "p_vol_platt", "p_logreg", "p_lightgbm", "p_lgbm_resid", "p_ensemble"]),
    }
    if imps:
        keys = set().union(*imps)
        report["lgbm_importance"] = dict(sorted({k: float(np.mean([d.get(k, 0.0) for d in imps])) for k in keys}.items(), key=lambda x: -x[1]))
    if coefs:
        report["logreg_coef_last_fold"] = {str(k): v for k, v in coefs[-1].items()}
    report["ablation"] = ablation(df, bars, mret, n_folds=2 if quick else 4)
    report["elapsed_s"] = round(time.time() - t0, 1)
    out_dir = data_dir() / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    path = out_dir / f"eval_{stamp}{'_quick' if quick else ''}.json"
    path.write_text(json.dumps(report, indent=1, default=float))
    P.to_parquet(out_dir / f"oos_predictions_{stamp}.parquet", index=False)
    log.info("evaluation_done", report=str(path), secs=report["elapsed_s"])
    return path
