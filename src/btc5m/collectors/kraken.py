"""Kraken WebSocket v2 public feeds: BTC/USD ``trade`` and ``book`` (depth 10).

Verified 2026-11-24 (wss://ws.kraken.com/v2). Trade messages carry the taker side directly.
"""
from __future__ import annotations

import json

from btc5m.collectors.base import WSCollector
from btc5m.collectors.coinbase import _book_row
from btc5m.config import load_config
from btc5m.live.state import Book, MarketState
from btc5m.storage.store import ParquetSink
from btc5m.timeutil import parse_iso


class KrakenCollector(WSCollector):
    name = "kraken"

    def __init__(self, state: MarketState, trade_sink: ParquetSink | None = None, book_sink: ParquetSink | None = None, snap_ms: int = 1000):
        super().__init__()
        cfg = load_config()
        self.url = cfg.exchanges.kraken.ws
        self.symbol = cfg.exchanges.kraken.symbol
        self.state = state
        self.trade_sink = trade_sink
        self.book_sink = book_sink
        self.snap_every = snap_ms / 1000.0
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._last_snap = 0.0

    def subscribe_messages(self):
        self._bids.clear()
        self._asks.clear()
        return [
            {"method": "subscribe", "params": {"channel": "trade", "symbol": [self.symbol]}},
            {"method": "subscribe", "params": {"channel": "book", "symbol": [self.symbol], "depth": 10}},
        ]

    def handle(self, raw: str, rx: float) -> None:
        j = json.loads(raw)
        ch = j.get("channel")
        if ch == "trade":
            for d in j.get("data", []):
                ts = parse_iso(d["timestamp"])
                price, size = float(d["price"]), float(d["qty"])
                taker = 1 if d["side"] == "buy" else -1
                self.state.bars["kraken"].add_trade(ts, price, size, taker)
                self.health.observe_latency(ts, rx)
                if self.trade_sink and j.get("type") == "update":
                    self.trade_sink.add({"rx": rx, "ts": ts, "price": price, "size": size, "taker_side": taker, "trade_id": d.get("trade_id")})
        elif ch == "book":
            for d in j.get("data", []):
                if j.get("type") == "snapshot":
                    self._bids = {float(x["price"]): float(x["qty"]) for x in d.get("bids", [])}
                    self._asks = {float(x["price"]): float(x["qty"]) for x in d.get("asks", [])}
                else:
                    for side, book in (("bids", self._bids), ("asks", self._asks)):
                        for x in d.get(side, []):
                            p, q = float(x["price"]), float(x["qty"])
                            if q == 0:
                                book.pop(p, None)
                            else:
                                book[p] = q
                    # depth-10 subscription: trim to 10 levels as Kraken does
                    for book, rev in ((self._bids, True), (self._asks, False)):
                        if len(book) > 10:
                            for p in sorted(book, reverse=not rev)[: len(book) - 10]:
                                book.pop(p, None)
                ts = parse_iso(d["timestamp"]) if d.get("timestamp") else rx
                if rx - self._last_snap >= self.snap_every:
                    bids = sorted(self._bids.items(), key=lambda x: -x[0])[:10]
                    asks = sorted(self._asks.items(), key=lambda x: x[0])[:10]
                    self.state.books["kraken"] = Book(bids=bids, asks=asks, ts=ts, rx=rx)
                    self._last_snap = rx
                    if self.book_sink and bids and asks:
                        self.book_sink.add(_book_row(rx, ts, bids, asks))
