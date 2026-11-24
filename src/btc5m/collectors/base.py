"""Reconnecting WebSocket client base with health tracking.

Every collector records, per stream: messages received, last receive time, reconnect count,
last error, and (where the venue provides a source timestamp) the receive-minus-source
latency. The live engine uses these to mark feeds STALE and to gate predictions.
"""
from __future__ import annotations

import asyncio
import json
import random
import ssl
import time
from collections import deque
from dataclasses import dataclass, field

import websockets

from btc5m.config import ca_bundle
from btc5m.logging import get_logger


@dataclass
class FeedHealth:
    name: str
    connected: bool = False
    msgs: int = 0
    reconnects: int = 0
    last_rx: float = 0.0
    last_error: str | None = None
    latencies: deque = field(default_factory=lambda: deque(maxlen=500))

    def observe_latency(self, src_ts: float, rx_ts: float) -> None:
        if src_ts > 0:
            self.latencies.append(rx_ts - src_ts)

    def snapshot(self, now: float | None = None) -> dict:
        now = now or time.time()
        lat = sorted(self.latencies)
        med = lat[len(lat) // 2] if lat else None
        p95 = lat[int(len(lat) * 0.95)] if lat else None
        return {
            "name": self.name,
            "connected": self.connected,
            "msgs": self.msgs,
            "reconnects": self.reconnects,
            "age_s": (now - self.last_rx) if self.last_rx else None,
            "latency_median_s": med,
            "latency_p95_s": p95,
            "last_error": self.last_error,
        }


def ssl_context() -> ssl.SSLContext:
    ca = ca_bundle()
    return ssl.create_default_context(cafile=ca) if isinstance(ca, str) else ssl.create_default_context()


class WSCollector:
    """Subclasses implement ``subscribe_messages()`` and ``handle(msg, rx_ts)``."""

    name = "ws"
    url = ""
    heartbeat_text: str | None = None  # application-level text ping (e.g. "PING")
    heartbeat_every: float = 10.0
    recv_timeout: float = 30.0

    def __init__(self):
        self.health = FeedHealth(self.name)
        self.log = get_logger(f"collector.{self.name}").bind(feed=self.name)
        self._stop = asyncio.Event()
        self._ws = None

    def subscribe_messages(self) -> list:
        return []

    async def on_connect(self, ws) -> None:
        for m in self.subscribe_messages():
            await ws.send(m if isinstance(m, str) else json.dumps(m))

    def handle(self, raw: str, rx_ts: float) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    async def send(self, msg) -> None:
        if self._ws is not None:
            await self._ws.send(msg if isinstance(msg, str) else json.dumps(msg))

    def stop(self) -> None:
        self._stop.set()

    async def _heartbeat(self, ws) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_every)
            try:
                await ws.send(self.heartbeat_text)
            except Exception:
                return

    async def run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            hb = None
            try:
                async with websockets.connect(
                    self.url, ssl=ssl_context(), open_timeout=20, ping_interval=20, ping_timeout=20, max_size=2**25
                ) as ws:
                    self._ws = ws
                    self.health.connected = True
                    await self.on_connect(ws)
                    self.log.info("connected", url=self.url)
                    backoff = 1.0
                    if self.heartbeat_text:
                        hb = asyncio.create_task(self._heartbeat(ws))
                    while not self._stop.is_set():
                        raw = await asyncio.wait_for(ws.recv(), timeout=self.recv_timeout)
                        rx = time.time()
                        self.health.msgs += 1
                        self.health.last_rx = rx
                        if raw in ("PONG", "pong", ""):
                            continue
                        try:
                            self.handle(raw, rx)
                        except Exception as e:  # never let one bad message kill the feed
                            self.health.last_error = f"handle: {e!r}"
                            self.log.warning("handle_error", err=repr(e), raw=str(raw)[:300])
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.health.last_error = repr(e)
                self.log.warning("disconnected", err=repr(e), backoff=backoff)
            finally:
                self.health.connected = False
                self._ws = None
                if hb:
                    hb.cancel()
            if self._stop.is_set():
                break
            self.health.reconnects += 1
            await asyncio.sleep(backoff + random.random())
            backoff = min(backoff * 2, 60.0)
