"""Kraken Futures public WebSocket (wss://futures.kraken.com/ws/v1): PF_XBTUSD perpetual.

Verified 2026-11-24: ``trade`` feed rows carry ``type`` in {fill, liquidation, ...} and the
taker ``side``; ``ticker`` carries ``markPrice``, ``index`` (spot index), funding and open
interest. Used for liquidation pressure and futures-vs-spot basis features.
"""
from __future__ import annotations

import json

from btc5m.collectors.base import WSCollector
from btc5m.live.state import MarketState
from btc5m.storage.store import ParquetSink

URL = "wss://futures.kraken.com/ws/v1"
PRODUCT = "PF_XBTUSD"


class KrakenFuturesCollector(WSCollector):
    name = "kraken_futures"
    url = URL

    def __init__(self, state: MarketState, trade_sink: ParquetSink | None = None, ticker_sink: ParquetSink | None = None):
        super().__init__()
        self.state = state
        self.trade_sink = trade_sink
        self.ticker_sink = ticker_sink

    def subscribe_messages(self):
        return [
            {"event": "subscribe", "feed": "trade", "product_ids": [PRODUCT]},
            {"event": "subscribe", "feed": "ticker", "product_ids": [PRODUCT]},
        ]

    def handle(self, raw: str, rx: float) -> None:
        j = json.loads(raw)
        feed = j.get("feed")
        if "event" in j or "time" not in j:
            return  # info / subscribed acks / snapshots
        if feed == "trade":
            ts = j["time"] / 1000.0
            price, qty = float(j["price"]), float(j["qty"])
            taker = 1 if j.get("side") == "buy" else -1
            typ = j.get("type", "fill")
            self.state.bars["kraken_fut"].add_trade(ts, price, qty, taker)
            if typ == "liquidation":
                self.state.liquidations.append((ts, taker, qty, price))
            self.health.observe_latency(ts, rx)
            if self.trade_sink:
                self.trade_sink.add({"rx": rx, "ts": ts, "price": price, "size": qty, "taker_side": taker, "type": typ})
        elif feed == "ticker":
            ts = j["time"] / 1000.0
            mark = j.get("markPrice")
            if mark is not None:
                self.state.futures_mark.add(ts, rx, float(mark))
            if self.ticker_sink:
                self.ticker_sink.add({"rx": rx, "ts": ts, "mark": mark, "index": j.get("index"), "last": j.get("last"), "bid": j.get("bid"), "ask": j.get("ask"),
                                      "funding_rate": j.get("funding_rate"), "open_interest": j.get("openInterest")})
