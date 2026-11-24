"""Collector message parsing against message shapes captured from the live feeds (2026-11-24)."""
import json

import numpy as np

from btc5m.collectors.coinbase import CoinbaseCollector
from btc5m.collectors.kraken import KrakenCollector
from btc5m.collectors.polymarket_clob import PolymarketMarketCollector
from btc5m.collectors.polymarket_rtds import RTDSCollector
from btc5m.live.state import MarketState, SecondBars


def test_rtds_snapshot_and_update():
    st = MarketState()
    c = RTDSCollector(st, None)
    snap = {"payload": {"data": [{"timestamp": 1795497207000, "value": 60500.7}, {"timestamp": 1795497208000, "value": 60501.2}]}, "topic": "crypto_prices_twap_sixty"}
    upd = {"payload": {"symbol": "btc/usd", "timestamp": 1795497268000, "value": 60485.73}, "timestamp": 1795497268786, "topic": "crypto_prices_chainlink", "type": "update"}
    c.handle(json.dumps(snap), 1795497209.0)
    c.handle(json.dumps(upd), 1795497269.5)
    assert len(st.chainlink_twap.q) == 2
    assert st.chainlink_spot.last()[2] == 60485.73
    assert st.chainlink_twap.value_at_src(1795497208)[2] == 60501.2
    assert c.subscribe_messages()[0]["subscriptions"][0]["filters"] == '{"symbol":"btc/usd"}'


def test_coinbase_match_taker_side_is_opposite_of_maker():
    st = MarketState()
    c = CoinbaseCollector(st)
    c.handle(json.dumps({"type": "match", "side": "buy", "size": "0.5", "price": "60000.0", "time": "2026-11-24T05:24:58.432299Z", "trade_id": 1}), 0)
    w = st.bars["coinbase"].window(int(st.bars["coinbase"].last_ts) + 1, 1)
    assert w["sell"][0] == 0.5 and w["buy"][0] == 0.0  # maker bought -> taker sold


def test_kraken_trade_and_book():
    st = MarketState()
    c = KrakenCollector(st)
    c.handle(json.dumps({"channel": "trade", "type": "update", "data": [{"symbol": "BTC/USD", "side": "buy", "price": 60001.0, "qty": 0.1, "ord_type": "market", "trade_id": 9, "timestamp": "2026-11-24T05:24:57.013018Z"}]}), 0)
    assert st.bars["kraken"].last_price == 60001.0
    c.handle(json.dumps({"channel": "book", "type": "snapshot", "data": [{"symbol": "BTC/USD", "bids": [{"price": 60000.0, "qty": 1.0}], "asks": [{"price": 60001.0, "qty": 2.0}], "timestamp": "2026-11-24T05:24:57.1Z"}]}), 1.0)
    b = st.books["kraken"]
    assert b.bids[0] == (60000.0, 1.0) and b.asks[0] == (60001.0, 2.0)


def test_polymarket_book_and_price_change():
    st = MarketState()
    c = PolymarketMarketCollector(st)
    c.handle(json.dumps([{"event_type": "book", "asset_id": "U", "market": "m", "timestamp": "1782753357257",
                          "bids": [{"price": "0.48", "size": "100"}], "asks": [{"price": "0.50", "size": "50"}]}]), 10.0)
    c.handle(json.dumps({"event_type": "price_change", "market": "m", "timestamp": "1782753358257",
                         "price_changes": [{"asset_id": "U", "price": "0.49", "size": "20", "side": "BUY", "best_bid": "0.49", "best_ask": "0.50"}]}), 11.0)
    b = st.pm_books["U"]
    assert b.bids[0] == (0.49, 20.0) and b.asks[0] == (0.50, 50.0)
    c.handle(json.dumps({"event_type": "last_trade_price", "asset_id": "U", "market": "m", "price": "0.5", "size": "10", "side": "BUY", "timestamp": "1782753359257"}), 12.0)
    assert st.pm_trades[-1][3] == 0.5


def test_second_bars_window_excludes_current_second():
    sb = SecondBars(ring_s=100)
    sb.add_trade(1000.2, 10.0, 1.0, 1)
    sb.add_trade(1001.5, 11.0, 2.0, -1)
    w = sb.window(1001, 3)  # seconds 998, 999, 1000
    assert np.isnan(w["close"][0]) and w["close"][-1] == 10.0
    assert w["sell"].sum() == 0.0
