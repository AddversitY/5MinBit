"""Evaluate live paper predictions from the ledger (unseen live rounds only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from btc5m.eval import metrics as M
from btc5m.live.ledger import Ledger

TARGET_ROUNDS = 1000


def ledger_frame(ledger: Ledger | None = None) -> pd.DataFrame:
    ledger = ledger or Ledger.default()
    recs = ledger.records()
    preds = {r["hash"]: r for r in recs if r["type"] == "prediction"}
    rows = []
    for s in (r for r in recs if r["type"] == "settlement"):
        p = preds.get(s["ref_hash"])
        if not p:
            continue
        rows.append({
            "round_start": p["round_start"], "checkpoint": p["checkpoint"], "model_version": p["model_version"], "p_up": p["p_up"],
            "y": s["outcome_up"], "market_mid": (p.get("market") or {}).get("implied_p_up_mid"), "status": p["status"], "data_status": p["data_status"],
            "hyp_pnl": s.get("hyp_pnl_per_share"), "hyp_side": s.get("hyp_side"), "p_proxy": (p.get("alt_probs") or {}).get("proxy_ref"),
        })
    return pd.DataFrame(rows)


def paper_report(ledger: Ledger | None = None) -> dict:
    df = ledger_frame(ledger)
    ok, n_rec, msg = (ledger or Ledger.default()).verify()
    out = {"ledger_ok": ok, "ledger_records": n_rec, "ledger_msg": msg, "target_rounds": TARGET_ROUNDS}
    if df.empty:
        out.update({"settled_rounds": 0, "progress": 0.0})
        return out
    out["settled_rounds"] = int(df["round_start"].nunique())
    out["progress"] = out["settled_rounds"] / TARGET_ROUNDS
    out["by_checkpoint"] = {}
    for k, g in df.groupby("checkpoint"):
        y = g["y"].to_numpy()
        r = {"n": int(len(g)), "model": M.summary(g["p_up"].to_numpy(), y)}
        h = g[g["market_mid"].notna()]
        if len(h) >= 20:
            mk = h["market_mid"].clip(0.005, 0.995).to_numpy()
            r["market"] = M.summary(mk, h["y"].to_numpy())
            days = (h["round_start"] // 3600).to_numpy()  # hour blocks (live sample is short)
            pm, yh = h["p_up"].to_numpy(), h["y"].to_numpy()
            pt, lo, hi = M.day_block_bootstrap(lambda idx: M.brier(pm[idx], yh[idx]) - M.brier(mk[idx], yh[idx]), days, n_boot=500)
            r["d_brier_vs_market"] = {"point": pt, "ci95": [lo, hi]}
        t = g[g["hyp_pnl"].notna()]
        if len(t):
            r["hypothetical"] = {"trades": int(len(t)), "mean_pnl_per_share": float(t["hyp_pnl"].mean()), "total_pnl_shares": float(t["hyp_pnl"].sum()),
                                 "hit_rate": float((t["hyp_pnl"] > 0).mean()), "note": "hypothetical, not realised"}
        out["by_checkpoint"][int(k)] = r
    out["warning"] = None if out["settled_rounds"] >= TARGET_ROUNDS else f"only {out['settled_rounds']} of {TARGET_ROUNDS} target live rounds settled; results are not yet conclusive"
    return out


__all__ = ["paper_report", "ledger_frame", "np"]
