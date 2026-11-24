# DATA_SOURCES

Every source below was exercised from this codebase on 2026-11-24. "Verified" means the
endpoint, parameters and message shapes were observed directly, not taken from memory.
The system is **read-only**: it never authenticates a wallet, signs, or sends orders.

## Polymarket

| Purpose | Endpoint | Auth | Notes (verified) | Code |
|---|---|---|---|---|
| Round discovery, metadata, official results | `GET https://gamma-api.polymarket.com/events?slug=btc-updown-5m-<start>` (repeatable `slug`, ≤ 20 per call) | none | `eventMetadata.priceToBeat/finalPrice` only after resolution; `outcomePrices` reaches `["1","0"]`/`["0","1"]` when final (intermediate values like `0.9995` are **not** final) | `history/gamma_rounds.py`, `collectors/polymarket_clob.py` |
| Live order book and trades | `wss://ws-subscriptions-clob.polymarket.com/ws/market`, `{"assets_ids":[…],"type":"market","custom_feature_enabled":true}`, text `PING` every 10 s | none | `book`, `price_change` (level size + best bid/ask), `last_trade_price` (taker side, ms timestamp), `best_bid_ask`, `market_resolved` | `collectors/polymarket_clob.py` |
| Chainlink BTC/USD spot + 60 s TWAP (live) | `wss://ws-live-data.polymarket.com` (legacy RTDS) topics `crypto_prices_chainlink`, `crypto_prices_twap_sixty` (filter exactly `{"symbol":"btc/usd"}`), `crypto_prices` (Binance relay, `btcusdt`); text `PING` every 5 s | none | **Deprecated** in favour of PolyBolt; still serving on 2026-11-24. ~1 update/s; ~1.5 s latency; snapshot of ~60 s on subscribe | `collectors/polymarket_rtds.py` |
| Same, new protocol | `wss://ws-live-v2.polymarket.com/ws` (PolyBolt) channels `price.crypto`, `price.crypto.twap` | **CLOB API credentials + Chainlink API key** | Unauthenticated subscribe returns `{"op":"error","code":"auth_required"}`. Optional, disabled by default (`polymarket.polybolt_enabled`) | not wired (see ROADMAP) |
| Historical taker trades | `GET https://data-api.polymarket.com/v2/trades?condition=<≤20 ids>&limit=1000` then `cursor` | none | One row per fill on the taker side; `timestamp` = block time, a median **2.2 s after** the match | `history/pm_trades.py` |
| Price history | `GET https://data-api.polymarket.com/v2/prices-history` / CLOB `/prices-history` | none | 1-minute grain retained ≥ 7 days only, so not usable for intra-round history | — |
| Fees | per-market `feeSchedule {rate, exponent, takerOnly}` (Gamma); docs: `fee = C × rate × p × (1−p)` | — | 0.05 for BTC 5m/15m since 2026-11-20 | `rules.py` |

**PolyBolt fallback plan.** RTDS price topics are planned for removal. When that happens:
(1) supply credentials via environment variables to enable PolyBolt; this needs an API key
derived *by you* outside this tool, and this project will never ask for a private key; or
(2) run on the exchange-based reference proxy, which the engine already uses to fill gaps and
logs for every prediction (`alt_probs.proxy_ref`), at the cost of the ~1% outcome
disagreement measured in RESEARCH.md §3. The dashboard shows INSUFFICIENT EVIDENCE whenever
the Chainlink feed is stale.

## Chainlink

* Settlement source: Chainlink **Data Streams** BTC/USD TWAP-60
  (`https://data.chain.link/streams/btc-usd-twap-60s-streams`). Direct access to Data Streams
  needs a Chainlink API key (and for history, typically an enterprise agreement).
  `data.chain.link` answered HTTP 429 to automated requests. Not scraped.
* These are **not** the on-chain Chainlink Data Feeds (push oracles with deviation/heartbeat
  updates). Do not substitute them; they are a different product with different update times.

## Exchanges

| Venue | Live | History | Notes |
|---|---|---|---|
| **Coinbase Exchange** (US-regulated) | `wss://ws-feed.exchange.coinbase.com`: `matches` (maker side → taker is opposite), `level2_batch` (served as `level2_50`), `heartbeat` | `GET /products/BTC-USD/candles?granularity=60` (≤ 300/request); `/trades` (≈1.7M trades/day, too slow to backfill) | Primary live book and trades; 1-minute volatility context |
| **Kraken** (US-regulated) | `wss://ws.kraken.com/v2`: `trade` (taker side), `book` depth 10 | `GET /0/public/Trades?pair=XBTUSD&since=<ns>` (1000 trades/call, ~1 call/s) | Primary historical tick source (≈ 86–230k trades/day) |
| **Kraken Futures** | `wss://futures.kraken.com/ws/v1`: `trade` (`type` = fill / liquidation), `ticker` (`markPrice`, `index`, funding, OI) for `PF_XBTUSD` | — | Liquidations and futures–spot basis (live-only features). Kraken Futures is not offered to all jurisdictions; only public market data is read |
| **Binance** spot / USD-M | `stream.binance.com` / `fstream.binance.com` | — | **Disabled.** `api.binance.com` and `fapi.binance.com` returned HTTP **451** from the development environment, and we do not route around that. Enable `exchanges.binance.enabled` only where Binance permits access |
| Bybit | — | — | HTTP 403 from the development environment; not used |
| OKX | reachable | — | Not used (restricted for some jurisdictions; adds little beyond Coinbase and Kraken) |

## Third-party datasets (not used, documented for follow-up)

* `polydata5/polymarket-updown-5m-ws` (Hugging Face, CC-BY-4.0): raw WebSocket capture of the
  5-minute up/down markets (CLOB books and trades plus RTDS Chainlink) from 2026-08-21. This
  would allow historical evaluation of order-book and Chainlink-lag features. **Gated:**
  needs a Hugging Face login and accepting its terms.
* Envio Polymarket fills export, Polymarket-v1 Database (arXiv:2606.04217), Jia et al.
  dataset suite (arXiv:2604.20421).

## Storage

| Dataset | Path | Produced by |
|---|---|---|
| Rounds | `data/hist/rounds/rounds.parquet` | `btc5m backfill-rounds` |
| Kraken trades | `data/hist/kraken_trades/date=YYYY-MM-DD/trades.parquet` | `btc5m backfill-kraken` |
| Coinbase 1m candles | `data/hist/coinbase_candles_1m/candles.parquet` | `btc5m backfill-candles` |
| Polymarket taker trades | `data/hist/pm_trades/batch-<start>.parquet` | `btc5m backfill-pm-trades` |
| Live streams | `data/raw/live/<stream>/date=YYYY-MM-DD/*.parquet` (chainlink, pm_book, pm_trades, pm_rounds, coinbase_trades, coinbase_book, kraken_trades, kraken_book, kraken_fut_trades, kraken_fut_ticker) | `btc5m serve` / `btc5m collect` |
| Checkpoint dataset | `data/datasets/checkpoints_kraken.parquet` (+ `.meta.json` with code hash) | `btc5m build-dataset` |
| Reports | `data/reports/eval_*.json`, `oos_predictions_*.parquet` | `btc5m evaluate` |
| Models | `data/models/<version>/{model.pkl,meta.json}`, `registry.json` | `btc5m train-production` |
| Paper ledger | `data/ledger/predictions.jsonl` (append-only, hash-chained) | `btc5m serve` |

DuckDB queries the Parquet globs directly, e.g.
`select * from read_parquet('data/raw/live/chainlink/date=*/*.parquet')`.
