"""Coinbase Exchange public WebSocket: BTC-USD trades (``matches``), ticker, L2 (``level2_batch``).

Verified 2026-11-24 (wss://ws-feed.exchange.coinbase.com): subscribing to ``level2_batch`` is
served as ``level2_50`` (top-50 snapshot then ``l2update`` deltas). For ``match`` messages the
``side`` field is the MAKER side, so taker (aggressor) side is the opposite.
"""
from __future__ import annotations

import json
import time

from btc5m.collectors.base import WSCollector
from btc5m.config import load_config
from btc5m.live.state import Book, MarketState
from btc5m.storage.store import ParquetSink
from btc5m.timeutil import parse_iso


class CoinbaseCollector(WSCollector):
    name = "coinbase"

    def __init__(self, state: MarketState, trade_sink: ParquetSink | None = None, book_sink: ParquetSink | None = None, depth: int = 10, snap_ms: int = 1000):
        super().__init__()
        cfg = load_config()
        self.url = cfg.exchanges.coinbase.ws
        self.product = cfg.exchanges.coinbase.product
        self.state = state
        self.trade_sink = trade_sink
        self.book_sink = book_sink
        self.depth = depth
        self.snap_every = snap_ms / 1000.0
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._last_snap = 0.0

    def subscribe_messages(self):
        self._bids.clear()
        self._asks.clear()
        return [{"type": "subscribe", "product_ids": [self.product], "channels": ["matches", "level2_batch", "heartbeat"]}]

    def handle(self, raw: str, rx: float) -> None:
        j = json.loads(raw)
        t = j.get("type")
        if t in ("match", "last_match"):
            ts = parse_iso(j["time"])
            price, size = float(j["price"]), float(j["size"])
            taker = -1 if j["side"] == "buy" else 1  # maker buy => taker sell
            self.state.bars["coinbase"].add_trade(ts, price, size, taker)
            self.health.observe_latency(ts, rx)
            if self.trade_sink and t == "match":
                self.trade_sink.add({"rx": rx, "ts": ts, "price": price, "size": size, "taker_side": taker, "trade_id": j.get("trade_id")})
        elif t == "snapshot":
            self._bids = {float(p): float(s) for p, s in j.get("bids", [])}
            self._asks = {float(p): float(s) for p, s in j.get("asks", [])}
            self._emit_book(rx, rx)
        elif t == "l2update":
            for side, p, s in j.get("changes", []):
                book = self._bids if side == "buy" else self._asks
                p, s = float(p), float(s)
                if s == 0:
                    book.pop(p, None)
                else:
                    book[p] = s
            ts = parse_iso(j["time"]) if j.get("time") else rx
            if rx - self._last_snap >= self.snap_every:
                self._emit_book(ts, rx)

    def _emit_book(self, ts: float, rx: float) -> None:
        bids = sorted(self._bids.items(), key=lambda x: -x[0])[: self.depth]
        asks = sorted(self._asks.items(), key=lambda x: x[0])[: self.depth]
        self.state.books["coinbase"] = Book(bids=bids, asks=asks, ts=ts, rx=rx)
        self._last_snap = rx
        if self.book_sink and bids and asks:
            self.book_sink.add(_book_row(rx, ts, bids, asks))


def _book_row(rx, ts, bids, asks) -> dict:
    row = {"rx": rx, "ts": ts}
    for i in range(10):
        row[f"bp{i}"] = bids[i][0] if i < len(bids) else None
        row[f"bs{i}"] = bids[i][1] if i < len(bids) else None
        row[f"ap{i}"] = asks[i][0] if i < len(asks) else None
        row[f"as{i}"] = asks[i][1] if i < len(asks) else None
    return row


__all__ = ["CoinbaseCollector", "_book_row", "time"]
