"""Polymarket CLOB market WebSocket + round discovery.

Endpoint (docs.polymarket.com, verified 2026-11-24): wss://ws-subscriptions-clob.polymarket.com/ws/market
Subscribe with ``{"assets_ids": [...], "type": "market", "custom_feature_enabled": true}``;
add/remove with ``{"assets_ids": [...], "operation": "subscribe"|"unsubscribe"}``.
Events: ``book`` (snapshot), ``price_change`` (level updates + best bid/ask), ``last_trade_price``,
``best_bid_ask``, ``tick_size_change``, ``market_resolved``. Text ``PING`` every 10 s.

Round discovery uses Gamma ``/events?slug=btc-updown-5m-<start>``.
"""
from __future__ import annotations

import asyncio
import json
import time

from btc5m.collectors.base import WSCollector
from btc5m.config import load_config
from btc5m.history.gamma_rounds import parse_event
from btc5m.http import async_client
from btc5m.live.state import Book, MarketState, PolyRound
from btc5m.storage.store import ParquetSink
from btc5m.timeutil import ROUND_S, round_start, slug_for


class PolymarketMarketCollector(WSCollector):
    name = "polymarket_clob"
    heartbeat_text = "PING"
    heartbeat_every = 10.0

    def __init__(self, state: MarketState, trade_sink: ParquetSink | None = None, book_sink: ParquetSink | None = None, round_sink: ParquetSink | None = None):
        super().__init__()
        self.cfg = load_config()
        self.url = self.cfg.polymarket.market_ws
        self.state = state
        self.trade_sink = trade_sink
        self.book_sink = book_sink
        self.round_sink = round_sink
        self.subscribed: set[str] = set()
        self.token_round: dict[str, tuple[int, str]] = {}  # token -> (start, "up"/"down")
        self._books: dict[str, dict[str, dict[float, float]]] = {}
        self._last_snap: dict[str, float] = {}

    # ---- round discovery -------------------------------------------------------------
    async def discover(self, start: int) -> PolyRound | None:
        if start in self.state.rounds and self.state.rounds[start].token_up:
            return self.state.rounds[start]
        async with async_client() as c:
            r = await c.get(f"{self.cfg.polymarket.gamma_url}/events", params={"slug": slug_for(start)})
            if r.status_code != 200 or not r.json():
                return None
            row = parse_event(r.json()[0])
        if not row or not row.get("token_up"):
            return None
        pr = PolyRound(
            start=start, slug=row["slug"], condition_id=row["condition_id"], token_up=row["token_up"], token_down=row["token_down"],
            settlement_kind=row["settlement_kind"], twap_lookback_s=row["twap_lookback_s"] or 0,
            fee_rate=row["fee_rate"] if row["fee_rate"] is not None else 0.05, fee_exponent=row["fee_exponent"] or 1.0,
            tick_size=row["tick_size"] or 0.01,
        )
        self.state.rounds[start] = pr
        self.token_round[pr.token_up] = (start, "up")
        self.token_round[pr.token_down] = (start, "down")
        if self.round_sink:
            self.round_sink.add({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()})
        return pr

    async def round_manager(self) -> None:
        """Keep the current and next round subscribed; drop rounds that ended > 60 s ago."""
        while not self._stop.is_set():
            try:
                now = time.time()
                cur = round_start(now)
                wanted = [cur] + ([cur + ROUND_S] if now - cur > ROUND_S - 90 else [])
                tokens = set()
                for s in wanted:
                    pr = await self.discover(s)
                    if pr:
                        tokens |= {pr.token_up, pr.token_down}
                keep = {t for t, (s, _) in self.token_round.items() if s >= cur - ROUND_S and now - (s + ROUND_S) < 60}
                add = tokens - self.subscribed
                drop = self.subscribed - tokens - keep
                if add and self._ws is not None:
                    await self.send({"assets_ids": sorted(add), "operation": "subscribe", "custom_feature_enabled": True})
                    self.subscribed |= add
                if drop and self._ws is not None:
                    await self.send({"assets_ids": sorted(drop), "operation": "unsubscribe"})
                    self.subscribed -= drop
                    for t in drop:
                        self._books.pop(t, None)
                for s in list(self.state.rounds):
                    if s < cur - 6 * ROUND_S:
                        self.state.rounds.pop(s, None)
            except Exception as e:
                self.log.warning("round_manager_error", err=repr(e))
            await asyncio.sleep(1.0)

    async def on_connect(self, ws) -> None:
        self._books.clear()
        cur = round_start(time.time())
        tokens = []
        for s in (cur, cur + ROUND_S):
            pr = await self.discover(s)
            if pr:
                tokens += [pr.token_up, pr.token_down]
        self.subscribed = set(tokens)
        await ws.send(json.dumps({"assets_ids": tokens, "type": "market", "custom_feature_enabled": True}))

    async def run(self) -> None:
        mgr = asyncio.create_task(self.round_manager())
        try:
            await super().run()
        finally:
            mgr.cancel()

    # ---- message handling ------------------------------------------------------------
    def handle(self, raw: str, rx: float) -> None:
        j = json.loads(raw)
        for ev in j if isinstance(j, list) else [j]:
            self._handle_one(ev, rx)

    def _handle_one(self, ev: dict, rx: float) -> None:
        et = ev.get("event_type")
        ts = float(ev.get("timestamp") or 0) / 1000.0 or rx
        if et == "book":
            tok = ev["asset_id"]
            self._books[tok] = {
                "bids": {float(x["price"]): float(x["size"]) for x in ev.get("bids", [])},
                "asks": {float(x["price"]): float(x["size"]) for x in ev.get("asks", [])},
            }
            self._emit(tok, ts, rx, force=True)
        elif et == "price_change":
            for pc in ev.get("price_changes", []):
                tok = pc["asset_id"]
                b = self._books.setdefault(tok, {"bids": {}, "asks": {}})
                side = b["bids"] if pc["side"] == "BUY" else b["asks"]
                p, s = float(pc["price"]), float(pc["size"])
                if s == 0:
                    side.pop(p, None)
                else:
                    side[p] = s
                self._emit(tok, ts, rx)
        elif et == "last_trade_price":
            tok = ev["asset_id"]
            price, size = float(ev["price"]), float(ev["size"])
            self.state.pm_trades.append((ts, tok, ev.get("side"), price, size))
            self.health.observe_latency(ts, rx)
            if self.trade_sink:
                s, leg = self.token_round.get(tok, (None, None))
                self.trade_sink.add({"rx": rx, "ts": ts, "token": tok, "round_start": s, "leg": leg, "side": ev.get("side"), "price": price, "size": size, "fee_rate_bps": ev.get("fee_rate_bps")})
        elif et == "market_resolved":
            if self.round_sink:
                self.round_sink.add({"event": "market_resolved", "rx": rx, "raw": json.dumps(ev)})

    def _emit(self, tok: str, ts: float, rx: float, force: bool = False) -> None:
        if not force and rx - self._last_snap.get(tok, 0) < 0.5:
            return
        b = self._books.get(tok)
        if b is None:
            return
        bids = sorted(b["bids"].items(), key=lambda x: -x[0])[:10]
        asks = sorted(b["asks"].items(), key=lambda x: x[0])[:10]
        self.state.pm_books[tok] = Book(bids=bids, asks=asks, ts=ts, rx=rx)
        self._last_snap[tok] = rx
        if self.book_sink:
            s, leg = self.token_round.get(tok, (None, None))
            row = {"rx": rx, "ts": ts, "token": tok, "round_start": s, "leg": leg}
            for i in range(10):
                row[f"bp{i}"] = bids[i][0] if i < len(bids) else None
                row[f"bs{i}"] = bids[i][1] if i < len(bids) else None
                row[f"ap{i}"] = asks[i][0] if i < len(asks) else None
                row[f"as{i}"] = asks[i][1] if i < len(asks) else None
            row["bid_depth"] = float(sum(s for _, s in b["bids"].items()))
            row["ask_depth"] = float(sum(s for _, s in b["asks"].items()))
            self.book_sink.add(row)
