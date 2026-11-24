"""Command-line entry point: ``btc5m <command>`` (or ``python -m btc5m.cli``)."""
from __future__ import annotations

import argparse
import sys

from btc5m.config import load_config
from btc5m.logging import get_logger, setup_logging


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    cfg = load_config()
    p = argparse.ArgumentParser(prog="btc5m", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("backfill-rounds", help="Gamma: historical rounds, price-to-beat, outcomes")
    s.add_argument("--days", type=float, default=cfg.backfill["rounds_days"])
    s = sub.add_parser("backfill-kraken", help="Kraken XBT/USD tick trades")
    s.add_argument("--days", type=float, default=cfg.backfill["kraken_days"])
    s = sub.add_parser("backfill-candles", help="Coinbase BTC-USD 1m candles")
    s.add_argument("--days", type=float, default=cfg.backfill["coinbase_candle_days"])
    s = sub.add_parser("backfill-pm-trades", help="Polymarket taker trades per round")
    s.add_argument("--days", type=float, default=cfg.backfill["pm_trades_days"])

    sub.add_parser("collect", help="run all live collectors (writes Parquet)")
    s = sub.add_parser("compact", help="merge small live Parquet files per stream/day")
    s.add_argument("--min-age", type=float, default=3600.0)
    s = sub.add_parser("build-dataset", help="build point-in-time checkpoint dataset from history")
    s.add_argument("--source", default="kraken", choices=["kraken"])
    s.add_argument("--workers", type=int, default=None)
    s = sub.add_parser("evaluate", help="walk-forward evaluation of all models -> reports")
    s.add_argument("--quick", action="store_true")
    s = sub.add_parser("train-production", help="fit candidate model on all data; register as candidate")
    s.add_argument("--model", default="lgbm")
    s = sub.add_parser("promote", help="promote a validated candidate (requires passing gate)")
    s.add_argument("version")
    s.add_argument("--force", action="store_true", help="override a failed gate (recorded in registry history)")
    s = sub.add_parser("serve", help="run live engine + API + dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s = sub.add_parser("report", help="render an evaluation JSON into Markdown")
    s.add_argument("path", nargs="?")
    sub.add_parser("verify-ledger", help="verify the paper-trading ledger hash chain")
    sub.add_parser("settle-ledger", help="attach official outcomes to ledger predictions")
    sub.add_parser("paper-report", help="evaluate live paper predictions")

    a = p.parse_args(argv)
    log = get_logger("cli")
    log.info("cmd", cmd=a.cmd)

    if a.cmd == "backfill-rounds":
        from btc5m.history.gamma_rounds import backfill

        df = backfill(a.days)
        print(f"rounds on disk: {len(df)}")
    elif a.cmd == "backfill-kraken":
        from btc5m.history.kraken_trades import backfill

        backfill(a.days)
    elif a.cmd == "backfill-candles":
        from btc5m.history.coinbase_candles import backfill

        backfill(a.days)
    elif a.cmd == "backfill-pm-trades":
        from btc5m.history.pm_trades import backfill

        backfill(a.days)
    elif a.cmd == "collect":
        import asyncio

        from btc5m.collectors.runner import run_collectors

        asyncio.run(run_collectors())
    elif a.cmd == "compact":
        from btc5m.storage.store import compact

        res = compact(a.min_age)
        print(f"compacted {len(res)} stream-days: {res}")
    elif a.cmd == "build-dataset":
        from btc5m.features.dataset import build_dataset

        build_dataset(source=a.source, workers=a.workers)
    elif a.cmd == "evaluate":
        from btc5m.eval.run import run_evaluation

        run_evaluation(quick=a.quick)
    elif a.cmd == "train-production":
        from btc5m.models.registry import train_candidate

        train_candidate(a.model)
    elif a.cmd == "promote":
        from btc5m.models.registry import promote

        promote(a.version, force=a.force)
    elif a.cmd == "serve":
        import uvicorn

        uvicorn.run("btc5m.api.server:app", host=a.host, port=a.port, log_level="info")
    elif a.cmd == "report":
        from btc5m.eval.report import render_file
        from btc5m.models.registry import latest_report

        print(render_file(a.path or latest_report()[0]))
    elif a.cmd == "verify-ledger":
        from btc5m.live.ledger import Ledger

        ok, n, msg = Ledger.default().verify()
        print(f"ledger ok={ok} records={n} {msg}")
        return 0 if ok else 1
    elif a.cmd == "settle-ledger":
        from btc5m.live.settle import settle_pending

        print(settle_pending())
    elif a.cmd == "paper-report":
        from btc5m.live.paper_report import paper_report

        print(paper_report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
