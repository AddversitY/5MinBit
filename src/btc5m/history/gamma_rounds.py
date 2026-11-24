"""Backfill historical BTC 5-minute rounds from the Gamma API.

Per round we keep the official ``eventMetadata.priceToBeat`` / ``finalPrice`` (the Chainlink
values Polymarket used), the resolved ``outcomePrices``, the settlement methodology, fee
schedule and CLOB token ids. Rounds are fetched by slug (``btc-updown-5m-<start>``), up to 20
slugs per request (the Gamma page size observed for multi-slug queries).

Resumable: completed (closed) rounds already on disk are skipped.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa

from btc5m.config import load_config
from btc5m.http import client, get_json
from btc5m.logging import get_logger
from btc5m.rules import settlement_kind
from btc5m.storage.store import hist_path, write_parquet_atomic
from btc5m.timeutil import ROUND_S, parse_iso, slug_for

log = get_logger("gamma_rounds")
BATCH = 20


def parse_event(e: dict) -> dict | None:
    ms = e.get("markets") or []
    if not ms:
        return None
    m = ms[0]
    start = int(e["slug"].rsplit("-", 1)[1])
    meta = e.get("eventMetadata") or {}
    try:
        op = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
    except (TypeError, ValueError):
        op = []
    outcomes = json.loads(m.get("outcomes") or "[]")
    tokens = json.loads(m.get("clobTokenIds") or "[]")
    up_idx = outcomes.index("Up") if "Up" in outcomes else 0
    dn_idx = 1 - up_idx
    resolved = bool(m.get("closed")) and len(op) == 2 and max(op) == 1.0 and min(op) == 0.0
    fs = m.get("feeSchedule") or {}
    cfg = m.get("cryptoMarketConfig") or {}
    kind = settlement_kind({**m, "resolutionSource": m.get("resolutionSource") or e.get("resolutionSource")})
    ptb = meta.get("priceToBeat")
    fin = meta.get("finalPrice")
    return {
        "slug": e["slug"],
        "start": start,
        "end": start + ROUND_S,
        "event_id": int(e["id"]),
        "market_id": int(m["id"]),
        "condition_id": m.get("conditionId"),
        "token_up": tokens[up_idx] if len(tokens) == 2 else None,
        "token_down": tokens[dn_idx] if len(tokens) == 2 else None,
        "settlement_kind": kind,
        "twap_lookback_s": int(cfg.get("twapLookbackSeconds") or 0) if cfg.get("twapEnabled") else 0,
        "config_id": m.get("cryptoMarketConfigId"),
        "resolution_source": m.get("resolutionSource") or e.get("resolutionSource"),
        "price_to_beat": float(ptb) if ptb is not None else None,
        "final_price": float(fin) if fin is not None else None,
        "outcome_up": (int(op[up_idx] > 0.5) if resolved else None),
        "resolved": resolved,
        "closed_time": parse_iso(m["closedTime"]) if m.get("closedTime") else None,
        "volume": float(m.get("volume") or 0.0) if m.get("volume") is not None else None,
        "fee_rate": float(fs["rate"]) if "rate" in fs else None,
        "fee_exponent": float(fs["exponent"]) if "exponent" in fs else None,
        "fees_enabled": bool(m.get("feesEnabled")),
        "tick_size": float(m.get("orderPriceMinTickSize") or 0.01),
        "created_at": parse_iso(m["createdAt"]) if m.get("createdAt") else None,
        "accepting_orders_ts": parse_iso(m["acceptingOrdersTimestamp"]) if m.get("acceptingOrdersTimestamp") else None,
        "uma_status": m.get("umaResolutionStatus"),
        "fetched_at": time.time(),
    }


def fetch_rounds(starts: list[int], c=None) -> list[dict]:
    cfg = load_config()
    c = c or client()
    out = []
    for i in range(0, len(starts), BATCH):
        chunk = starts[i : i + BATCH]
        params = [("slug", slug_for(s)) for s in chunk] + [("limit", str(BATCH))]
        evs = get_json(c, f"{cfg.polymarket.gamma_url}/events", params=params)
        for e in evs:
            row = parse_event(e)
            if row:
                out.append(row)
    return out


def rounds_file() -> Path:
    return hist_path("rounds") / "rounds.parquet"


def load_rounds() -> pd.DataFrame:
    f = rounds_file()
    if not f.exists():
        return pd.DataFrame()
    return pd.read_parquet(f)


def backfill(days: float, end_ts: float | None = None, chunk_rounds: int = 2000) -> pd.DataFrame:
    """Fetch every round starting in [end - days, end). Merges with existing file."""
    end_ts = end_ts or time.time()
    last = (int(end_ts) // ROUND_S) * ROUND_S - ROUND_S  # most recent fully elapsed round start
    first = last - int(days * 86400 // ROUND_S) * ROUND_S
    existing = load_rounds()
    done = set(existing.loc[existing["resolved"] == True, "start"].tolist()) if len(existing) else set()  # noqa: E712
    todo = [s for s in range(last, first - 1, -ROUND_S) if s not in done]
    log.info("rounds_backfill_start", first=first, last=last, todo=len(todo), have=len(done))
    c = client()
    frames = [existing] if len(existing) else []
    for i in range(0, len(todo), chunk_rounds):
        rows = fetch_rounds(todo[i : i + chunk_rounds], c)
        if rows:
            frames.append(pd.DataFrame(rows))
        df = _merge(frames)
        write_parquet_atomic(pa.Table.from_pandas(df, preserve_index=False), rounds_file())
        frames = [df]
        log.info("rounds_backfill_progress", fetched=i + len(todo[i : i + chunk_rounds]), total=len(todo), rows=len(df))
    return _merge(frames) if frames else pd.DataFrame()


def _merge(frames: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    df = df.sort_values(["start", "fetched_at"]).drop_duplicates("start", keep="last")
    return df.sort_values("start").reset_index(drop=True)
