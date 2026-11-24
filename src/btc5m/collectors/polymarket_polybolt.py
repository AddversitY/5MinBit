"""Polymarket PolyBolt live-data socket (successor of RTDS price topics). OPTIONAL.

Protocol per docs.polymarket.com (api-reference/live-data/overview, wss/polybolt; read
2026-11-24):

* ``wss://ws-live-v2.polymarket.com/ws``; the server pings every 25 s.
* Auth frame (before subscribing):
  ``{"op":"auth","rid":"a1","auth":{"apiKey","secret","passphrase"},"chainlink_api_key":"…"}``
  -> ``{"op":"authed","rid":"a1"}``. Errors: auth_required / auth_invalid /
  auth_unavailable / chainlink_key_required (4010; connection stays open).
* Subscribe: ``{"op":"subscribe","rid":"s1","subscriptions":[{"channel":"price.crypto",
  "filter":{"symbol":"btcusd","provider":"chainlink"}}, {"channel":"price.crypto.twap",
  "filter":{"symbol":"btcusd","window_seconds":60,"provider":"chainlink"}}]}``
* Frames: ``{"v":1,"channel","seq","ts","snapshot"?,"dropped"?,"payload"}``; snapshot payload
  is ``{"symbol","source","data":[{timestamp,value,full_accuracy_value}…]}``; live payload is a
  single point with ``source``. A provider preference may be ignored, so ``source`` is checked.

Credentials come ONLY from environment variables (never a private key):
``BTC5M_CLOB_API_KEY``, ``BTC5M_CLOB_API_SECRET``, ``BTC5M_CLOB_API_PASSPHRASE``,
``BTC5M_CHAINLINK_API_KEY``. Without them this collector is not started. The protocol code
is tested against documented message shapes only; it has not been exercised live because no
credentials were available during development.
"""
from __future__ import annotations

import asyncio
import json
import os

from btc5m.collectors.base import WSCollector
from btc5m.config import load_config
from btc5m.live.state import MarketState
from btc5m.storage.store import ParquetSink

ENV = ("BTC5M_CLOB_API_KEY", "BTC5M_CLOB_API_SECRET", "BTC5M_CLOB_API_PASSPHRASE", "BTC5M_CHAINLINK_API_KEY")


def credentials_available() -> bool:
    return all(os.environ.get(k) for k in ENV)


class PolyBoltCollector(WSCollector):
    name = "polymarket_polybolt"
    recv_timeout = 60.0

    def __init__(self, state: MarketState, sink: ParquetSink | None = None):
        super().__init__()
        self.url = load_config().polymarket.polybolt_ws
        self.state = state
        self.sink = sink
        self.authed = False

    async def on_connect(self, ws) -> None:
        self.authed = False
        await ws.send(json.dumps({
            "op": "auth", "rid": "a1",
            "auth": {"apiKey": os.environ["BTC5M_CLOB_API_KEY"], "secret": os.environ["BTC5M_CLOB_API_SECRET"], "passphrase": os.environ["BTC5M_CLOB_API_PASSPHRASE"]},
            "chainlink_api_key": os.environ["BTC5M_CHAINLINK_API_KEY"],
        }))
        while True:
            j = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            if j.get("op") == "authed":
                self.authed = True
                break
            if j.get("op") == "error":
                raise RuntimeError(f"PolyBolt auth failed: {j.get('code')}")
        await ws.send(json.dumps({"op": "subscribe", "rid": "s1", "subscriptions": [
            {"channel": "price.crypto", "filter": {"symbol": "btcusd", "provider": "chainlink"}},
            {"channel": "price.crypto.twap", "filter": {"symbol": "btcusd", "window_seconds": 60, "provider": "chainlink"}},
        ]}))

    def handle(self, raw: str, rx: float) -> None:
        j = json.loads(raw)
        if j.get("op") == "error":
            self.health.last_error = f"polybolt error {j.get('code')}"
            return
        ch = j.get("channel")
        if ch not in ("price.crypto", "price.crypto.twap"):
            return
        p = j.get("payload") or {}
        if p.get("source") not in (None, "chainlink"):
            self.health.last_error = f"provider fallback to {p.get('source')}; ignored"
            return
        series = self.state.chainlink_spot if ch == "price.crypto" else self.state.chainlink_twap
        kind = "chainlink_spot" if ch == "price.crypto" else "chainlink_twap60"
        points = p.get("data") if j.get("snapshot") else [p]
        for d in points or []:
            if "value" not in d:
                continue
            ts = d["timestamp"] / 1000.0
            v = float(d.get("full_accuracy_value") or d["value"])
            series.add(ts, rx, v)
            if not j.get("snapshot") and ch == "price.crypto":
                self.health.observe_latency(ts, rx)
            if self.sink:
                self.sink.add({"rx": rx, "src_ts": ts, "kind": kind, "value": v, "full": d.get("full_accuracy_value"), "snapshot": bool(j.get("snapshot")), "via": "polybolt"})
