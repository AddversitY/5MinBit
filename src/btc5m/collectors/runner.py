"""Wire collectors to the shared MarketState and Parquet sinks, and run them concurrently."""
from __future__ import annotations

import asyncio

from btc5m.collectors.binance import BinanceCollector
from btc5m.collectors.coinbase import CoinbaseCollector
from btc5m.collectors.kraken import KrakenCollector
from btc5m.collectors.kraken_futures import KrakenFuturesCollector
from btc5m.collectors.polymarket_clob import PolymarketMarketCollector
from btc5m.collectors.polymarket_polybolt import PolyBoltCollector, credentials_available
from btc5m.collectors.polymarket_rtds import RTDSCollector
from btc5m.config import load_config
from btc5m.live.state import STATE, MarketState
from btc5m.logging import get_logger
from btc5m.storage.store import ParquetSink

log = get_logger("collectors")


def build_collectors(state: MarketState = STATE, persist: bool = True):
    cfg = load_config()
    fl = cfg.collector["flush_seconds"]

    def sink(name):
        return ParquetSink(name, fl) if persist else None

    sinks = {}

    def s(name):
        sinks[name] = sink(name)
        return sinks[name]

    cols = [
        RTDSCollector(state, s("chainlink")),
        PolymarketMarketCollector(state, s("pm_trades"), s("pm_book"), s("pm_rounds")),
    ]
    if cfg.polymarket["polybolt_enabled"]:
        if credentials_available():
            cols.append(PolyBoltCollector(state, s("chainlink_polybolt")))  # redundant Chainlink source
        else:
            log.warning("polybolt_enabled_but_no_credentials", hint="set BTC5M_CLOB_API_* and BTC5M_CHAINLINK_API_KEY")
    if cfg.exchanges.coinbase["enabled"]:
        cols.append(CoinbaseCollector(state, s("coinbase_trades"), s("coinbase_book")))
    if cfg.exchanges.kraken["enabled"]:
        cols.append(KrakenCollector(state, s("kraken_trades"), s("kraken_book")))
    cols.append(KrakenFuturesCollector(state, s("kraken_fut_trades"), s("kraken_fut_ticker")))
    if cfg.exchanges.binance["enabled"]:
        cols.append(BinanceCollector(state, s("binance_trades")))
        cols.append(BinanceCollector(state, s("binance_fut_trades"), futures=True))
    for c in cols:
        state.register_health(c.name, c.health)
    return cols, {k: v for k, v in sinks.items() if v is not None}


async def flusher(sinks: dict, every: float = 60.0):
    while True:
        await asyncio.sleep(every)
        for sk in sinks.values():
            try:
                sk.flush()
            except Exception as e:  # disk issues must not kill collection
                log.error("flush_error", sink=sk.stream, err=repr(e))


async def run_collectors(state: MarketState = STATE, persist: bool = True):
    cols, sinks = build_collectors(state, persist)
    tasks = [asyncio.create_task(c.run(), name=c.name) for c in cols]
    if sinks:
        tasks.append(asyncio.create_task(flusher(sinks)))
    try:
        await asyncio.gather(*tasks)
    finally:
        for sk in sinks.values():
            sk.flush()
