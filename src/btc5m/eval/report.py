"""Render an evaluation JSON report into Markdown tables (used for MODEL_RESULTS.md)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

MODELS = ["p_coin_flip", "p_constant", "p_gaussian_rw", "p_vol_platt", "p_monte_carlo", "p_logreg", "p_logreg_hmm", "p_lightgbm", "p_lgbm_resid", "p_xgboost", "p_ensemble"]


def _d(ts) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _f(x, n=4):
    if x is None:
        return "—"
    try:
        return f"{x:.{n}f}"
    except (TypeError, ValueError):
        return str(x)


def _ci(v, n=4):
    if not v or v.get("ci95") is None:
        return "—"
    key = "d_brier" if "d_brier" in v else "d_brier_vs_market"
    lo, hi = v["ci95"]
    flag = " ✓" if hi < 0 else (" ✗" if lo > 0 else "")
    return f"{_f(v[key], n)} [{_f(lo, n)}, {_f(hi, n)}]{flag}"


def render(rep: dict) -> str:
    L: list[str] = []
    a = L.append
    ds, oos = rep["dataset"], rep["oos"]
    a(f"Report generated {_d(rep['generated_at'])} UTC{' (QUICK mode)' if rep.get('quick') else ''}.")
    a("")
    a(f"* Dataset: {ds['rounds']:,} rounds ({ds['rows']:,} checkpoint rows), {_d(ds['first'])} → {_d(ds['last'])}; TWAP-60 rounds: {ds['twap60_rounds']:,}.")
    a(f"* **Out-of-sample (walk-forward test) rounds: {oos['rounds']:,}** ({_d(oos['first'])} → {_d(oos['last'])}), {len(rep['folds'])} folds.")
    a("")
    a("### Reference-price proxy quality (Kraken-based reconstruction of the official Chainlink settlement)")
    a("")
    a("| Settlement era | Rounds | Outcome agreement | |final error| mean (USD) | sd (USD) | p95 |abs| |")
    a("|---|---:|---:|---:|---:|---:|")
    for k, v in rep["proxy_quality"].items():
        a(f"| {k} | {v['rounds']:,} | {v['outcome_agreement']:.4f} | {_f(v['final_err_mae'], 2)} | {_f(v['final_err_sd'], 2)} | {_f(v['final_err_p95_abs'], 2)} |")
    a("")
    a("### Out-of-sample metrics by checkpoint (TWAP-60 era)")
    bc = rep["metrics"]["by_checkpoint"]
    for metric, nd in (("brier", 4), ("logloss", 4), ("accuracy", 3), ("ece", 4)):
        a("")
        a(f"**{metric}** (lower is better)" if metric != "accuracy" else "**accuracy** (higher is better)")
        a("")
        cols = [m for m in MODELS if m in next(iter(bc.values()))]
        a("| k (s) | n | " + " | ".join(c[2:] for c in cols) + " |")
        a("|---|---:|" + "---:|" * len(cols))
        for k, row in bc.items():
            n = row[cols[0]]["n"]
            best = min(row[c][metric] for c in cols) if metric != "accuracy" else max(row[c][metric] for c in cols)
            cells = []
            for c in cols:
                v = row[c][metric]
                s = _f(v, nd)
                cells.append(f"**{s}**" if abs(v - best) < 1e-12 else s)
            a(f"| {k} | {n:,} | " + " | ".join(cells) + " |")
    a("")
    a("### Brier difference vs the parameter-free Gaussian random-walk model (negative = better; 4-hour-block bootstrap 95% CI; ✓ = significantly better, ✗ = significantly worse)")
    a("")
    vb = rep["metrics"]["vs_baseline"]
    cols = [m for m in MODELS if m != "p_gaussian_rw" and m in next(iter(vb.values()))]
    a("| k (s) | " + " | ".join(c[2:] for c in cols) + " |")
    a("|---|" + "---|" * len(cols))
    for k, row in vb.items():
        a(f"| {k} | " + " | ".join(_ci(row.get(c)) for c in cols) + " |")
    vm = rep["metrics"].get("vs_market") or {}
    if vm:
        a("")
        a("### Versus Polymarket's market-implied probability (rows where a market price could be reconstructed)")
        a("")
        a("Brier of each model minus Brier of the market mid on the same rows (negative = model better). 95% CI by 4-hour-block bootstrap.")
        a("")
        mcols = ["p_gaussian_rw", "p_vol_platt", "p_logreg", "p_lightgbm", "p_lgbm_resid", "p_ensemble", "p_stack_mkt"]
        a("| k (s) | n | market Brier | " + " | ".join(c[2:] for c in mcols) + " |")
        a("|---|---:|---:|" + "---|" * len(mcols))
        for k, row in vm.items():
            a(f"| {k} | {row['n']:,} | {_f(row['market']['brier'])} | " + " | ".join(_ci(row.get(c)) for c in mcols) + " |")
    vr = rep["metrics"].get("by_vol_regime") or {}
    if vr:
        a("")
        a("### Brier by volatility regime (trailing 7-day percentile of 1 h realised vol; known at decision time)")
        a("")
        a("| k (s) | regime | n | gaussian_rw | vol_platt | logreg | lightgbm | ensemble |")
        a("|---|---|---:|---:|---:|---:|---:|---:|")
        for k, regs in vr.items():
            for reg in ("low", "mid", "high"):
                if reg in regs:
                    r = regs[reg]
                    a(f"| {k} | {reg} | {r['p_gaussian_rw']['n']:,} | " + " | ".join(_f(r.get(c, {}).get("brier")) for c in ("p_gaussian_rw", "p_vol_platt", "p_logreg", "p_lightgbm", "p_ensemble")) + " |")
    tp = rep.get("training_performance") or []
    if tp:
        a("")
        a("### Training (in-sample) vs test performance — overfitting check")
        a("")
        a("| fold | model | train n | train Brier | train logloss |")
        a("|---|---|---:|---:|---:|")
        for r in tp:
            a(f"| {r['fold']} | {r['model']} | {r['n']:,} | {_f(r['brier'])} | {_f(r['logloss'])} |")
    ab = rep.get("ablation") or {}
    if ab:
        a("")
        a("### Feature-group ablation (residual LightGBM on top of the Gaussian base; last folds; Δ vs all features: positive Δ for drop_X = group X was helping; compare every row with `gaussian_only`)")
        a("")
        a("| variant | logloss | Δ logloss | early (k≤60) logloss | Δ early |")
        a("|---|---:|---:|---:|---:|")
        for v, r in sorted(ab.items(), key=lambda x: x[1]["logloss"]):
            a(f"| {v} | {_f(r['logloss'], 5)} | {_f(r['d_logloss_vs_all'], 5)} | {_f(r['early_logloss_k<=60'], 5)} | {_f(r['d_early_logloss_vs_all'], 5)} |")
    imp = rep.get("lgbm_importance") or {}
    if imp:
        a("")
        a("### LightGBM gain importance (mean over folds, top 15)")
        a("")
        a("| feature | share of gain |")
        a("|---|---:|")
        for k, v in list(imp.items())[:15]:
            a(f"| {k} | {v:.3f} |")
    tr = rep.get("trading") or {}
    if tr:
        a("")
        a("### Hypothetical taker-strategy results (PAPER ONLY — not realised profits)")
        a("")
        a("One share per round at checkpoint k when model EV after fee + 1-tick slippage exceeds the edge threshold. Fill = first observed taker print after a 2 s latency (rounds without a print are skipped). PnL in USD per share; CI = 4-hour-block bootstrap. Fee: current 0.05 schedule.")
        a("")
        a("| k (s) | model | edge ≥ | trades | hit rate | mean PnL/share [95% CI] | total | max DD |")
        a("|---|---|---:|---:|---:|---|---:|---:|")
        for k, mods in tr.items():
            for m, res in mods.items():
                for e in ("0.0", "0.02", "0.05"):
                    r = res.get(f"edge{e}_current_0.05")
                    if not r or not r.get("trades"):
                        continue
                    ci = r["mean_pnl_ci95"]
                    a(f"| {k} | {m.replace('p_', '')} | {e} | {r['trades']:,} | {r['hit_rate']:.3f} | {r['mean_pnl_per_share']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {r['total_pnl_shares']:+.1f} | {r['max_drawdown_shares']:.1f} |")
        a("")
        a("Stress test (5 s latency, 2-tick slippage, edge ≥ 0.02):")
        a("")
        a("| k (s) | model | trades | mean PnL/share [95% CI] |")
        a("|---|---|---:|---|")
        for k, mods in tr.items():
            for m, res in mods.items():
                r = res.get("stress_lat5_slip2_edge0.02")
                if r and r.get("trades"):
                    ci = r["mean_pnl_ci95"]
                    a(f"| {k} | {m.replace('p_', '')} | {r['trades']:,} | {r['mean_pnl_per_share']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    eg = rep.get("era_gaussian") or {}
    if eg:
        a("")
        a("### Parameter-free Gaussian model by settlement methodology (no training; whole dataset)")
        a("")
        a("| era | k (s) | n | accuracy | Brier | log loss |")
        a("|---|---|---:|---:|---:|---:|")
        for era, ks in eg.items():
            for k, r in ks.items():
                a(f"| {era} | {k} | {r['n']:,} | {r['accuracy']:.3f} | {r['brier']:.4f} | {r['logloss']:.4f} |")
    return "\n".join(L)


def render_file(path: Path) -> str:
    return render(json.loads(Path(path).read_text()))
