"""Attach official outcomes to ledger predictions (append-only settlement records).

Outcomes come only from Polymarket's resolved market (Gamma ``outcomePrices`` == [1,0] or
[0,1], plus ``eventMetadata.priceToBeat/finalPrice``). Hypothetical PnL is computed from the
paper decision that was frozen inside the prediction record.
"""
from __future__ import annotations

import asyncio
import time

from btc5m.config import load_config
from btc5m.history.gamma_rounds import fetch_rounds
from btc5m.http import client
from btc5m.live.ledger import Ledger
from btc5m.logging import get_logger
from btc5m.rules import FeeSchedule

log = get_logger("settle")


def settle_pending(ledger: Ledger | None = None, min_age_s: float = 20.0) -> dict:
    ledger = ledger or Ledger.default()
    now = time.time()
    pend = [p for p in ledger.pending() if now >= p["round_start"] + 300 + min_age_s]
    if not pend:
        return {"settled": 0, "pending": len(ledger.pending())}
    starts = sorted({p["round_start"] for p in pend})
    rows = {r["start"]: r for r in fetch_rounds(starts, client())}
    n = 0
    for p in pend:
        r = rows.get(p["round_start"])
        if not r or not r["resolved"]:
            continue
        y = int(r["outcome_up"])
        rec = {"outcome_up": y, "official_price_to_beat": r["price_to_beat"], "official_final_price": r["final_price"],
               "brier": (p["p_up"] - y) ** 2, "ptb_abs_error": (abs(p["price_to_beat"] - r["price_to_beat"]) if p.get("price_to_beat") and r["price_to_beat"] else None)}
        side, px = p.get("would_trade"), p.get("would_trade_price")
        if side and px:
            fs = p.get("fee_schedule") or {}
            fee = FeeSchedule(fs.get("rate", 0.05), fs.get("exponent", 1.0)).taker_fee_per_share(px)
            win = (y == 1) if side == "up" else (y == 0)
            rec.update({"hyp_side": side, "hyp_price": px, "hyp_fee": fee, "hyp_pnl_per_share": float(win) - px - fee, "hyp_note": "hypothetical paper result, not a realised trade"})
        mk = (p.get("market") or {}).get("implied_p_up_mid")
        if mk is not None:
            rec["market_brier"] = (mk - y) ** 2
        if ledger.add_settlement(p["hash"], rec):
            n += 1
    return {"settled": n, "pending": len(ledger.pending())}


async def settle_loop(ledger: Ledger, every_s: float = 30.0) -> None:
    while True:
        try:
            res = await asyncio.to_thread(settle_pending, ledger)
            if res.get("settled"):
                log.info("settled", **res)
        except Exception as e:
            log.warning("settle_failed", err=repr(e))
        await asyncio.sleep(every_s)


__all__ = ["settle_pending", "settle_loop", "load_config"]
