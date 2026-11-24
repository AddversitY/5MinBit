"""Binance spot/USD-M futures public streams. DISABLED BY DEFAULT.

From the development environment api.binance.com and fapi.binance.com answer HTTP 451
("unavailable for legal reasons"). We do not attempt to route around that. Enable
``exchanges.binance.enabled`` only where Binance permits access from your location.
Streams: ``btcusdt@aggTrade`` (``m`` = buyer is maker => taker sell) and, on futures,
``btcusdt@forceOrder`` (liquidations).
"""
from __future__ import annotations

import json

from btc5m.collectors.base import WSCollector
from btc5m.config import load_config
from btc5m.live.state import MarketState
from btc5m.storage.store import ParquetSink


class BinanceCollector(WSCollector):
    name = "binance"

    def __init__(self, state: MarketState, trade_sink: ParquetSink | None = None, futures: bool = False):
        super().__init__()
        cfg = load_config().exchanges.binance
        sym = cfg.symbol
        base = cfg.ws_futures if futures else cfg.ws_spot
        streams = [f"{sym}@aggTrade"] + ([f"{sym}@forceOrder"] if futures else [])
        self.url = f"{base}?streams={'/'.join(streams)}"
        self.name = "binance_futures" if futures else "binance"
        self.futures = futures
        self.state = state
        self.trade_sink = trade_sink

    def handle(self, raw: str, rx: float) -> None:
        j = json.loads(raw).get("data", {})
        if j.get("e") == "aggTrade":
            ts = j["T"] / 1000.0
            price, qty = float(j["p"]), float(j["q"])
            taker = -1 if j["m"] else 1
            if not self.futures:
                self.state.bars["binance"].add_trade(ts, price, qty, taker)
            self.health.observe_latency(ts, rx)
            if self.trade_sink:
                self.trade_sink.add({"rx": rx, "ts": ts, "price": price, "size": qty, "taker_side": taker})
        elif j.get("e") == "forceOrder":
            o = j["o"]
            ts = o["T"] / 1000.0
            self.state.liquidations.append((ts, 1 if o["S"] == "BUY" else -1, float(o["q"]), float(o["p"])))
