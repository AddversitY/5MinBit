# ARCHITECTURE

```
                ┌──────────────────────── live (btc5m serve) ─────────────────────────┐
 Polymarket RTDS ─┐                                                                    │
 (Chainlink spot, │   collectors/*  ──►  live/state.MarketState  ──►  live/engine ──► ledger (hash chain)
  TWAP-60)        │   (reconnecting      (1 s trade bars, books,      (features/compute   │        ▲
 Polymarket CLOB ─┤    WebSockets,        Chainlink series,            same code as        │   live/settle
 Coinbase ────────┤    health stats)      PM books)                    backtests)          │   (Gamma outcomes)
 Kraken ──────────┤        │                                               │               │
 Kraken Futures ──┘        ▼                                               ▼               │
                    storage ParquetSink                           api/server (FastAPI)     │
                    data/raw/live/…                               REST + WS ──► dashboard  │
                └────────────────────────────────────────────────────────────────────────┘

                ┌──────────────────────── research (offline) ────────────────────────────┐
 history/* backfills ──► data/hist/* ──► features/dataset (point-in-time rows per round×k)
                                              │
                                              ▼
                     eval/run: walk-forward (eval/splits) × models (models/*)
                               metrics + bootstrap CIs (eval/metrics), backtest (eval/backtest)
                                              │
                                              ▼
                     data/reports/eval_*.json ──► eval/report ──► MODEL_RESULTS.md
                                              │
                                              ▼
                     models/registry: train candidate ─► gate ─► (manual) promote ─► engine
                └────────────────────────────────────────────────────────────────────────┘
```

## Modules

| Module | Responsibility |
|---|---|
| `config.py`, `config/default.yaml` | All tunables; `mode: paper` is enforced (any other value raises) |
| `rules.py` | Settlement methodology, tie rule, fee formula, TWAP variance geometry |
| `timeutil.py` | Round arithmetic and slugs |
| `collectors/base.py` | WebSocket loop: exponential backoff + jitter, heartbeats, per-feed health (msgs, age, latency, reconnects), never dies on a bad message |
| `collectors/polymarket_rtds.py` | Chainlink spot / TWAP-60 / Binance relay from RTDS |
| `collectors/polymarket_clob.py` | Round discovery (Gamma), rolling subscription to current + next round tokens, local order books, trades |
| `collectors/coinbase.py`, `kraken.py`, `kraken_futures.py`, `binance.py` | Exchange trades, books, liquidations, futures ticker (Binance disabled) |
| `storage/store.py` | Atomic Parquet writes (temp + rename), DuckDB helpers |
| `history/*` | Resumable backfills (rounds, Kraken trades, Coinbase candles, Polymarket trades) |
| `features/bars.py` | Dense 1 s bars with the same `window()` API as the live ring buffer |
| `features/compute.py` | **Single** feature implementation (backtest and live) on arrays that end strictly before the decision time |
| `features/market_implied.py` | Market-implied probability and executable price from taker fills (mirrored-book logic) |
| `features/dataset.py` | Builds the checkpoint dataset in parallel; reference proxy anchored on official prices; code hash in metadata |
| `models/*` | Baselines (coin, constant, Gaussian RW, Platt vol, Monte Carlo bootstrap), LR, LightGBM, XGBoost, causal HMM regimes, stacking ensemble; common `fit/predict` interface |
| `eval/*` | Walk-forward splits with purge, metrics, block bootstrap, cost-aware backtest, ablations, report rendering |
| `models/registry.py` | Versioned artefacts (SHA-256 checked), promotion gate, manual promotion only |
| `live/engine.py` | Per-second prediction, uncertainty band, status gating (INSUFFICIENT EVIDENCE), checkpoint ledger writes |
| `live/ledger.py` | Append-only JSONL with SHA-256 chain; write-time integrity rules; full verification |
| `live/settle.py`, `live/paper_report.py`, `live/monitor.py` | Official outcomes, live evaluation, drift / decay monitoring |
| `api/server.py` | FastAPI app hosting collectors + engine + settlement; REST + WebSocket; serves the dashboard |
| `dashboard/` | React + TypeScript (Vite) + uPlot |

## Key design decisions

1. **One feature implementation.** `compute_features()` receives arrays that end at the data
   horizon. Backtests slice historical bars; live slices ring buffers. This removes the most
   common source of train/serve skew. Unit tests perturb all data after the horizon and
   assert the features do not change.
2. **Settlement-exact target.** Labels are Polymarket's official outcomes, never
   reconstructed ones. The reference proxy is only an input, and its error is measured per
   round.
3. **Benchmark = the market.** Every model is scored against Polymarket's own implied
   probability on the same rows, with block-bootstrap confidence intervals.
4. **Conservative execution.** Historical fills require an observed taker print after the
   latency; otherwise the round is skipped. Fees use the current (highest) schedule by default.
5. **Gated deployment.** A model reaches the live engine only through `promote`, which
   enforces the validation gate. Without a validated model the engine runs the
   parameter-free baseline and says so.
6. **Tamper-evident paper ledger.** Predictions are frozen before the round ends. Outcomes
   are appended as separate records, and any rewrite breaks the chain.
7. **Read-only.** There is no order-placement code path, no wallet and no key handling.
   `mode` must be `paper`.

## Running

See [README.md](README.md).
