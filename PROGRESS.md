# PROGRESS

Chronological log of work, findings and decisions. Times are UTC.

## 2026-11-24

### Environment and access (05:10)
* Repository was empty (README only). Python 3.13, Node 22, 4 CPUs, 15 GB RAM.
* Reachability: Polymarket Gamma/CLOB/Data API ✅, Coinbase ✅, Kraken ✅, Kraken Futures ✅,
  OKX ✅, **Binance API 451** (legal block, respected; connector disabled), Bybit 403,
  data.chain.link 429 (not scraped).

### Phase 1: research and verification (05:10–05:55)
* Live Gamma market object: current BTC 5m markets settle on the **Chainlink TWAP-60** stream
  (`twapLookbackSeconds: 60`); taker fee schedule rate 0.05; ties resolve Up.
* Docs: fee formula `C × rate × p(1−p)`; RTDS deprecated in favour of PolyBolt, which needs
  auth (verified: `auth_required`); Data API v1 retired, v2 cursor-paginated.
* Recorded RTDS Chainlink spot and TWAP-60 and compared with official results:
  **priceToBeat = TWAP-60 at the start second, finalPrice = TWAP-60 at the end second**,
  exact to the last decimal. The TWAP ≈ mean of 1 s spot over `[t−60, t)` (MAE $0.52).
* Found and fixed: an RTDS filter with a space after the colon silently yields no updates.
* Backfilled all 80,603 rounds. Methodology eras: spot (to 2026-08-06), TWAP-30
  (08-07…08-13), TWAP-60 (from 08-14). Fee eras: 0.25/exp2 → 0.01 (03-30) → **0.05
  (11-20 16:15)**. Outcomes match `final ≥ ptb` in 69,336/69,336 rounds. Up rate 49.6%
  (TWAP-60 era).
* Literature checked by search: Cont–Kukanov–Stoikov OFI, Stoikov micro-price, Kolm et al.
  deep OFI, Makarov–Schoar, Jaquart et al. (predictable but unprofitable after costs),
  Polymarket microstructure papers (Dubach; Qin & Yang; Jia et al.), and the Stanford/SMU
  settlement-manipulation study that preceded the TWAP switch.
* Found the `polydata5/polymarket-updown-5m-ws` dataset (order books and Chainlink from
  2026-08-21). It is gated, so it was not accessed; documented as a follow-up.

### Phase 2: data infrastructure (05:20–05:45)
* Collectors (RTDS, CLOB, Coinbase, Kraken, Kraken Futures; Binance disabled) with
  reconnection, heartbeats and health stats; Parquet sinks; running continuously since 05:30.
* Backfills: Coinbase 1 m candles (280 d, done), Polymarket taker trades (45 d, running),
  Kraken ticks (running newest-first at ~3.6 min per day of history; the bottleneck).
* Decision: no full Coinbase tick backfill (≈1.7M trades/day, too slow). Kraken is the
  historical tick source; Coinbase supplies 1-minute volatility context.

### Phase 3–5: features, models, evaluation (05:35–06:05)
* Single point-in-time feature implementation shared by the backtest and live engine.
  Leakage tests perturb all post-horizon data.
* Reference proxy: Kraken + per-round basis anchored on the official price to beat.
  Reproduces official outcomes in **98.7–99.0%** of rounds.
* Measured: Data API trade timestamps lag WS match time by a median 2.2 s, so market lookups
  are shifted by +2 s (errs in the market's favour).
* Defined **k = 0 as the instant the price to beat is observable** (~1.6 s after the open)
  and encoded it in tests, engine and replay.
* Models: coin flip, climatology, Gaussian RW (TWAP-aware), Platt vol, Monte Carlo bootstrap,
  per-checkpoint LR, LR+HMM, LightGBM, residual LightGBM (boosts from the Gaussian log-odds),
  XGBoost, equal-weight ensemble, market stack.
* Bugs found by early evaluations and fixed:
  - Day-block bootstrap gave degenerate CIs with few test days → 4-hour blocks.
  - Stacked ensemble weights were fitted on in-sample validation predictions (LogReg refits
    on train+val) → equal-weight log-odds average.
  - Plain LightGBM approximates the smooth Φ(z) poorly → added the residual LightGBM for a
    fair "do features add information?" test.
* Early quick evaluations (≤ 3 test days) all point the same way: the parameter-free
  Gaussian model is the best of our models, and **Polymarket's mid is better still** at
  every checkpoint, significantly so at k = 0 and 15 s. Not yet conclusive; the full run is
  pending more Kraken history.

### Phase 6–7: live engine, ledger, dashboard (05:45–06:00)
* Engine: per-second predictions, model-disagreement band, TWAP projection interval, status
  gating, and live-only features (book imbalance, micro-price, exchange−Chainlink gap,
  futures basis, liquidations, Polymarket book).
* Ledger: append-only, SHA-256 chained, write-time rules (prediction before round end;
  settlement after, once). Settlement from official Gamma results. Live paper test started
  at 05:50 (first round with an observed price to beat).
* Dashboard: React/TS + uPlot, light/dark, verified vs model labelling, INSUFFICIENT
  EVIDENCE banner, chart of BTC vs the price to beat, model vs market probability chart.
* Live lead–lag (26 min sample): Chainlink spot moves with Coinbase at lag 0 (ρ = 0.71) and
  +1 s (ρ = 0.50), then arrives 1.5 s later. The Coinbase−Chainlink gap predicts the next 5 s
  of Chainlink (ρ = 0.21) but not 30 s (ρ = 0.06), so it is a near-expiry effect only.

### Phase 8: deployment and reliability (07:30–09:30)
* CI now runs (the repository was made public, which gives it Actions minutes; before that
  every job was refused before starting). Jobs: Linux pytest, dashboard build, and a
  **Windows** job that runs the Windows paper-test runner exactly as a user would, the test
  suite, a server start-up with `/api/health`, and the one-writer-per-ledger check.
* Windows runner (`scripts/windows/`): per-user `uv`, Python 3.13, checksum-verified portable
  Node.js for the dashboard, keep-awake while running, restart loop, optional Startup-folder
  autostart. No admin rights, keys or wallet. The owner started the long paper test on their
  desktop with it on 2026-11-24.
* `btc5m serve` takes an OS-level lock next to the ledger; a second server on the same data
  directory exits (code 3) instead of forking the hash chain.
* Live compaction no longer risks duplicates or lost rows if a file is locked (Windows).
* Dataset build memory: ~20M Polymarket trade rows held as text ids needed ~9 GB and
  threatened the live server; the loader now reads one batch at a time and keeps only
  booleans and numbers (~1 GB peak). Exact duplicate rows are now judged including the
  transaction hash (there were none: 19,893,249 rows before and after).
* Settlement re-verified on ~4 h of recorded streams: price to beat = TWAP-60 at the start
  second on 45/45 rounds, final = TWAP-60 at the end second on 44/44.

### Phase 5: full walk-forward run (09:18–10:05)
* Backfills done: Kraken ticks 2026-08-21 → 11-23 (95 days, 11.9M trades; still extending
  backwards), Polymarket taker trades for 45 days (19.9M rows, 12,928 markets).
* Dataset: 26,412 TWAP-60 rounds × 9 checkpoints (237,708 rows) in 2.5 min. Walk-forward:
  11 folds, 20,891 out-of-sample rounds, 36 min.
* Result (MODEL_RESULTS.md): H1 yes, H2 barely, H3 no, H4 no. Market mid more accurate than
  every model at every checkpoint; all 189 hypothetical taker cells lose after costs; no model
  passes the promotion gate (best: ensemble, 2 of the 3 required early checkpoints).
* Reporting bug found and fixed: the market-stack model was skipped in the market comparison
  when undefined in some folds; metrics recomputed from the saved predictions (unchanged).

### Open items
* 1,000-round live paper milestone (~3.5 days): running on the owner's desktop.
* Extend history over the spot and TWAP-30 eras; maker-side (liquidity provision) study.
