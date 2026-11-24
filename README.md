# 5MinBit — Polymarket BTC 5-minute Up/Down prediction research engine

A research and **paper-testing-only** system for testing whether there is a statistically
significant, cost-surviving predictive edge in Polymarket's Bitcoin 5-minute Up/Down markets,
especially in the first 15–60 seconds of each round.

> **Read-only by design.** No wallet, no private keys, no order placement. `mode: paper` is
> enforced in code. All trading figures are hypothetical. Nothing here is investment advice.

**Current conclusion (details in [MODEL_RESULTS.md](MODEL_RESULTS.md)):** see the "Bottom
line" section of MODEL_RESULTS.md. In short, early-round outcomes are predictable from public
information (mostly the TWAP geometry), but Polymarket prices already reflect it. No model
has passed the validation gate, and no edge after costs has been demonstrated.

## Documents

| File | Contents |
|---|---|
| [RESEARCH.md](RESEARCH.md) | Verified market rules (TWAP-60 settlement measured exactly), methodology/fee history, API verification, literature review, hypotheses |
| [DATA_SOURCES.md](DATA_SOURCES.md) | Every endpoint used, auth, limits, legal notes, storage layout |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design and module map |
| [MODEL_RESULTS.md](MODEL_RESULTS.md) | Out-of-sample results: by checkpoint, vs market, by volatility regime, ablations, hypothetical costs |
| [ROADMAP.md](ROADMAP.md) | Phases, status, pre-registered decision rules |
| [PROGRESS.md](PROGRESS.md) | Chronological work log and decisions |

## Quick start

Requirements: Python ≥ 3.11, Node ≥ 18 (dashboard build only).

```bash
python -m venv .venv && . .venv/bin/activate     # or: uv venv && uv pip install -e ".[dev]"
pip install -e ".[dev]"
pytest -q                                         # 41 tests: rules, leakage, ledger, models, parsers

# 1. history (resumable; Kraken is the slow one at ~3–4 min per day of history)
btc5m backfill-rounds --days 280       # Gamma: every round, official price to beat/final, outcome, fees, method
btc5m backfill-candles --days 280      # Coinbase 1-minute candles
btc5m backfill-kraken --days 100       # Kraken tick trades (newest first)
btc5m backfill-pm-trades --days 45     # Polymarket taker trades (market-implied prices)

# 2. research
btc5m build-dataset                    # point-in-time checkpoint rows -> data/datasets/
btc5m evaluate                         # walk-forward OOS evaluation -> data/reports/eval_*.json
btc5m report > /tmp/results.md         # render the latest report as Markdown

# 3. models (never auto-promoted)
btc5m train-production --model lightgbm   # registers a candidate + its validation gate result
btc5m promote <version>                   # refused unless the gate passed

# 4. live paper testing + dashboard
(cd dashboard && npm ci && npm run build)
btc5m serve --port 8000                # collectors + engine + ledger + API; open http://127.0.0.1:8000
btc5m verify-ledger                    # recompute the hash chain
btc5m paper-report                     # live (unseen) round evaluation
python -m btc5m.live.replay            # replay recorded live streams -> live-only feature dataset
python scripts/verify_settlement.py    # re-check price to beat / final vs the Chainlink TWAP stream
```

Configuration: `config/default.yaml`. Override with `config/local.yaml` or
`BTC5M_CONFIG=/path.yaml`. Set `BTC5M_PERSIST=0` to run the server without writing raw data.

### Long-running paper test

The paper test needs a machine that stays on for days (≥ 1,000 rounds ≈ 3.5 days).

* **Linux/macOS/any Docker host:** `docker compose up -d --build` (dashboard on
  `127.0.0.1:8000`), or the systemd unit in `deploy/btc5m.service`.
* **Windows (no admin rights, no Docker):** double-click `scripts\windows\start-paper-test.cmd`,
  or run `powershell -ExecutionPolicy Bypass -File scripts\windows\run-paper-test.ps1 -InstallAutostart`.
  The script installs `uv` per-user, creates a Python 3.13 environment, builds the dashboard with
  a checksum-verified portable Node.js if none is installed, keeps Windows awake while running,
  restarts the server if it stops, and (with `-InstallAutostart`) starts minimized at log-on.
  Logs: `data\logs\serve.log`.

Only one server may write a given ledger; a second `btc5m serve` on the same data directory
exits with code 3 instead of forking the hash chain.

## What the dashboard shows

* **Verified data** (blue tags): Chainlink spot and TWAP-60 (via Polymarket RTDS), official
  price to beat, Coinbase/Kraken prices, Polymarket bid/ask, feed health.
* **Model estimates** (violet tags): P(Up)/P(Down), the disagreement band across models,
  the projected TWAP settlement with a 90% interval, and the most influential variables.
* **Hypothetical** (dashed tags): expected value after fee and slippage, live paper PnL.
* **INSUFFICIENT EVIDENCE** banner whenever any of these hold: a feed is stale; the price to
  beat is not yet observed; inputs are outside the validated range; no model has passed the
  validation gate; there is no statistically significant out-of-sample edge versus the market
  at the current checkpoint; or the market price lies inside the model's uncertainty band.
  Given the current evidence, this banner is expected to show essentially all the time. That
  is the honest state of the evidence.

## Repository layout

```
config/            default.yaml
src/btc5m/         collectors/ history/ features/ models/ eval/ live/ api/ storage/ rules.py cli.py
dashboard/         React + TypeScript (Vite, uPlot)
scripts/           verify_settlement.py
tests/             pytest suite
data/              (git-ignored) raw, hist, datasets, reports, models, ledger
```
