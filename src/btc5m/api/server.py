"""FastAPI backend: live collectors + engine + settlement in one process, REST + WebSocket.

Endpoints
    GET  /api/state         latest engine snapshot (model estimates + verified market data)
    GET  /api/series        per-second chart series (last N seconds)
    GET  /api/performance   walk-forward OOS summary + live paper-test metrics
    GET  /api/ledger        recent ledger predictions with settlements
    GET  /api/health        feed health + drift monitor
    WS   /ws                pushes the snapshot every second
    GET  /                  dashboard (dashboard/dist)
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from btc5m.collectors.runner import run_collectors
from btc5m.config import ROOT
from btc5m.live.engine import Engine
from btc5m.live.ledger import Ledger
from btc5m.live.monitor import DriftMonitor
from btc5m.live.paper_report import paper_report
from btc5m.live.settle import settle_loop
from btc5m.live.state import STATE
from btc5m.logging import get_logger
from btc5m.models.registry import latest_report, read_registry

log = get_logger("api")
ENGINE: Engine | None = None
MONITOR: DriftMonitor | None = None
_tasks: list = []


def _clean(o):
    """JSON-safe: NaN/inf -> None."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE, MONITOR
    ledger = Ledger.default()
    ENGINE = Engine(STATE, ledger)
    MONITOR = DriftMonitor()
    persist = os.environ.get("BTC5M_PERSIST", "1") == "1"
    _tasks.append(asyncio.create_task(run_collectors(STATE, persist=persist)))
    _tasks.append(asyncio.create_task(ENGINE.run()))
    _tasks.append(asyncio.create_task(settle_loop(ledger)))

    async def feed_monitor():
        while True:
            await asyncio.sleep(5)
            if ENGINE.latest.get("features"):
                MONITOR.observe(ENGINE.latest["features"])

    _tasks.append(asyncio.create_task(feed_monitor()))

    async def hourly_compaction():
        from btc5m.storage.store import compact

        while True:
            await asyncio.sleep(3600)
            try:
                await asyncio.to_thread(compact, 3600.0)
            except Exception as e:
                log.warning("compaction_failed", err=repr(e))

    if persist:
        _tasks.append(asyncio.create_task(hourly_compaction()))
    yield
    for t in _tasks:
        t.cancel()


app = FastAPI(title="btc5m paper-testing engine", lifespan=lifespan)


@app.get("/api/state")
def state():
    return JSONResponse(_clean(ENGINE.latest if ENGINE else {}))


@app.get("/api/series")
def series(seconds: int = 900):
    pts = list(ENGINE.series)[-seconds:] if ENGINE else []
    return JSONResponse(_clean(pts))


_perf_cache: dict = {"t": 0, "v": None}


@app.get("/api/performance")
def performance():
    if time.time() - _perf_cache["t"] < 30 and _perf_cache["v"]:
        return JSONResponse(_perf_cache["v"])
    path, rep = latest_report()
    summary = None
    if rep:
        bc = rep["metrics"]["by_checkpoint"]
        vm = rep["metrics"].get("vs_market", {})
        summary = {
            "report": Path(path).name, "oos_rounds": rep["oos"]["rounds"], "oos_range": [rep["oos"]["first"], rep["oos"]["last"]],
            "by_checkpoint": {k: {m: {x: v[m][x] for x in ("n", "accuracy", "brier", "logloss", "ece")} for m in v if m in ("p_gaussian_rw", "p_vol_platt", "p_logreg", "p_lightgbm", "p_ensemble")} for k, v in bc.items()},
            "vs_market": {k: {"n": v["n"], "market_brier": v["market"]["brier"], **{m: {"brier": v[m]["brier"], "d_vs_market": v[m]["d_brier_vs_market"], "ci95": v[m]["ci95"]} for m in v if m.startswith("p_")}} for k, v in vm.items()},
            "proxy_quality": rep.get("proxy_quality"),
        }
    out = _clean({"walk_forward": summary, "paper": paper_report(), "registry": read_registry()})
    _perf_cache.update(t=time.time(), v=out)
    return JSONResponse(out)


@app.get("/api/ledger")
def ledger(limit: int = 100):
    recs = Ledger.default().records()
    preds = [r for r in recs if r["type"] == "prediction"][-limit:]
    sets = {r["ref_hash"]: r for r in recs if r["type"] == "settlement"}
    rows = []
    for p in reversed(preds):
        s = sets.get(p["hash"], {})
        rows.append({"round_start": p["round_start"], "checkpoint": p["checkpoint"], "p_up": p["p_up"], "market_mid": (p.get("market") or {}).get("implied_p_up_mid"),
                     "status": p["status"], "would_trade": p.get("would_trade"), "outcome_up": s.get("outcome_up"), "hyp_pnl": s.get("hyp_pnl_per_share"),
                     "model_version": p["model_version"], "hash": p["hash"][:12]})
    return JSONResponse(_clean(rows))


@app.get("/api/health")
def health():
    return JSONResponse(_clean({"feeds": STATE.health_snapshot(), "drift": MONITOR.report() if MONITOR else {}, "ledger": Ledger.default().verify()}))


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_text(json.dumps(_clean(ENGINE.latest if ENGINE else {})))
            await asyncio.sleep(1.0)
    except (WebSocketDisconnect, RuntimeError):
        return


DIST = ROOT / "dashboard" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(DIST / "index.html")
