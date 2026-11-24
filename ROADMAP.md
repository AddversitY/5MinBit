# ROADMAP

Status key: ✅ done · 🟡 in progress / partial · ⬜ not started

## Phase 1 — Research, API verification, feasibility ✅
- ✅ Verify market rules, settlement (TWAP-60 at start/end second), tie rule, methodology eras, fee history (RESEARCH.md §1)
- ✅ Verify Polymarket Gamma, CLOB WS, RTDS, PolyBolt (auth), Data API v2 (block-time offset measured)
- ✅ Literature review with verified citations
- ✅ Inventory of historical datasets; legal/accessibility notes (Binance 451 respected)

## Phase 2 — Data infrastructure ✅ (backfills continue in background)
- ✅ Live collectors: RTDS Chainlink, CLOB books/trades, Coinbase, Kraken, Kraken Futures (Binance optional/disabled)
- ✅ Reconnection, heartbeats, feed health, Parquet persistence, DuckDB access
- ✅ Backfills: Gamma rounds (80k), Coinbase 1m candles (280 d)
- 🟡 Kraken tick backfill (newest-first, ~3.6 min/day of history; extends while the process runs)
- 🟡 Polymarket taker trades (45 days)
- ⬜ PolyBolt client (authenticated; needs user-supplied CLOB API creds + Chainlink key). Required before RTDS price topics are removed
- ⬜ Optional importer for the gated `polydata5/polymarket-updown-5m-ws` dataset (user HF token) to evaluate book and Chainlink-lag features historically

## Phase 3 — Features & baselines ✅
- ✅ Point-in-time feature builder shared with live engine; leakage tests
- ✅ Coin flip, climatology, Gaussian random walk (TWAP-aware), Platt-calibrated vol model, Monte Carlo bootstrap

## Phase 4 — Advanced models ✅
- ✅ Logistic regression (per checkpoint), LightGBM, XGBoost, causal HMM regimes, stacking ensemble, market-stacking test
- ⬜ LSTM/GRU: **not justified so far.** Gradient boosting does not beat the parameter-free model, so the extra non-linear capacity of a sequence net is not supported by the evidence (see MODEL_RESULTS.md). Revisit only if live-only order-book features show signal

## Phase 5 — Walk-forward & calibration ✅
- ✅ Expanding-window walk-forward with purge; validation-only calibration; block-bootstrap CIs; vol-regime breakdown; ablations; cost-aware hypothetical backtest; stress tests
- 🟡 Re-run on the full TWAP-60 era once the Kraken backfill completes

## Phase 6 — Dashboard ✅
- ✅ FastAPI + WebSocket; React/TS dashboard with verified vs model labelling, INSUFFICIENT EVIDENCE gating, live chart, feed health, influential variables, performance panels

## Phase 7 — Live paper testing 🟡
- ✅ Hash-chained ledger, automatic settlement from official outcomes, paper report, drift monitor
- 🟡 Accumulate ≥ 1,000 unseen live rounds (~3.5 days of continuous running). **Started 2026-11-24**; needs a persistent host (the development container is ephemeral)
- ⬜ Build a live-capture dataset (`raw/live/*`) and evaluate live-only features (book imbalance, micro-price, exchange−Chainlink lag gap, futures basis, liquidations, Polymarket book) with walk-forward once ≥ 2,000 rounds exist

## Phase 8 — Review & reliability ⬜
- ⬜ Run the server under a supervisor (systemd/docker) with log rotation and disk quotas
- ⬜ Anchor the ledger head hash externally (e.g. periodic signed commit) for third-party verifiability
- ⬜ Maker-side (liquidity provision) study: makers pay no fee and earn rebates. This is structurally more promising than taking, but needs queue-position and adverse-selection modelling from live book data
- ⬜ Monthly re-validation; promote only through the gate

## Decision rules (pre-registered)
- A model is promoted only if the gate in `models/registry.py` passes (≥ 2,000 OOS rounds, significant Brier improvement vs the Gaussian baseline at ≥ 3 early checkpoints, ECE ≤ 0.03).
- A "trading edge" is claimed only if the live paper test, on ≥ 1,000 unseen rounds, shows a Brier improvement over the market mid with a 95% CI excluding zero **and** a positive hypothetical PnL after current fees with a CI excluding zero. Neither condition is met so far.
