"""Polymarket Real-Time Data Socket (legacy RTDS): Chainlink BTC/USD spot + 60 s TWAP.

Verified 2026-11-24: public (no auth) subscriptions to ``crypto_prices_chainlink``,
``crypto_prices_twap_sixty`` (filter ``{"symbol":"btc/usd"}``) and ``crypto_prices``
(Binance relay, filter ``btcusdt``) stream ~1 update/s each. The first frame per topic is a
snapshot with ~60 s of 1-second history in ``payload.data``. Updates carry
``payload.timestamp`` (ms, the Chainlink report second) and ``payload.value``.
Text ``PING`` heartbeat every 5 s. NOTE: the filter string must be compact JSON
(``{"symbol":"btc/usd"}``); with a space after the colon the server sends no updates. RTDS price topics are deprecated (PolyBolt replaces them
behind authentication); see DATA_SOURCES.md for the fallback plan.

Measured: the official ``priceToBeat`` equals the ``crypto_prices_twap_sixty`` value at the
round start second.
"""
from __future__ import annotations

import json

from btc5m.collectors.base import WSCollector
from btc5m.config import load_config
from btc5m.live.state import MarketState
from btc5m.storage.store import ParquetSink

TOPICS = {
    "crypto_prices_chainlink": "chainlink_spot",
    "crypto_prices_twap_sixty": "chainlink_twap60",
    "crypto_prices": "binance_relay",
}


class RTDSCollector(WSCollector):
    name = "polymarket_rtds"
    heartbeat_text = "PING"
    heartbeat_every = 5.0

    def __init__(self, state: MarketState, sink: ParquetSink | None = None):
        super().__init__()
        cfg = load_config()
        self.url = cfg.polymarket.rtds_ws
        self.state = state
        self.sink = sink

    def subscribe_messages(self):
        return [
            {
                "action": "subscribe",
                "subscriptions": [
                    {"topic": "crypto_prices_chainlink", "type": "*", "filters": '{"symbol":"btc/usd"}'},
                    {"topic": "crypto_prices_twap_sixty", "type": "*", "filters": '{"symbol":"btc/usd"}'},
                    {"topic": "crypto_prices", "type": "*", "filters": "btcusdt"},
                ],
            }
        ]

    def _series(self, kind: str):
        return {"chainlink_spot": self.state.chainlink_spot, "chainlink_twap60": self.state.chainlink_twap, "binance_relay": self.state.binance_relay}[kind]

    def handle(self, raw: str, rx: float) -> None:
        j = json.loads(raw)
        kind = TOPICS.get(j.get("topic"))
        if kind is None:
            return
        p = j.get("payload") or {}
        series = self._series(kind)
        if "data" in p:  # snapshot: history; mark as snapshot so it is never used as "fresh"
            for d in p["data"]:
                ts = d["timestamp"] / 1000.0
                v = float(d["value"])
                series.add(ts, rx, v)
                if self.sink:
                    self.sink.add({"rx": rx, "src_ts": ts, "kind": kind, "value": v, "full": d.get("full_accuracy_value"), "snapshot": True})
            return
        if "value" not in p:
            return
        ts = p["timestamp"] / 1000.0
        v = float(p["value"])
        series.add(ts, rx, v)
        if kind == "chainlink_spot":
            self.health.observe_latency(ts, rx)
        if self.sink:
            self.sink.add({"rx": rx, "src_ts": ts, "kind": kind, "value": v, "full": p.get("full_accuracy_value"), "snapshot": False})
