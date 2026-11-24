"""Round arithmetic for Polymarket BTC 5-minute markets.

Verified against live Gamma data (2026-11-24): event slug is ``btc-updown-5m-<start_unix>``,
``eventStartTime`` == start, ``endDate`` == start + 300 s, rounds tile the clock on 5-minute
boundaries (UTC epoch multiples of 300).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

ROUND_S = 300
SLUG_PREFIX = "btc-updown-5m-"


def now() -> float:
    return time.time()


def round_start(ts: float) -> int:
    return int(ts // ROUND_S) * ROUND_S


def round_end(ts: float) -> int:
    return round_start(ts) + ROUND_S


def elapsed_in_round(ts: float) -> float:
    return ts - round_start(ts)


def remaining_in_round(ts: float) -> float:
    return round_end(ts) - ts


def slug_for(start: int) -> str:
    if start % ROUND_S:
        raise ValueError(f"{start} is not a round boundary")
    return f"{SLUG_PREFIX}{start}"


def start_from_slug(slug: str) -> int:
    if not slug.startswith(SLUG_PREFIX):
        raise ValueError(slug)
    return int(slug[len(SLUG_PREFIX):])


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso(s: str) -> float:
    s = s.strip().replace(" ", "T")
    if s.endswith("+00"):
        s = s + ":00"
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
