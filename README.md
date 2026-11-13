# ETI — Evolutionary Trading Intelligence

ETI is a research-first quantitative and AI trading intelligence project. The v0.2 architecture provides reproducible experiments, deterministic risk controls, transparent strategy comparisons, and a mobile-friendly research dashboard.

## Safety status

- Research-only: no live order placement routes or authenticated exchange actions exist.
- Training is not automatic: model fitting is an explicit CLI action and is not part of startup or CI.
- No profitability claim: backtests do not establish future performance.
- Independent risk: model decisions must pass deterministic risk and portfolio checks; defaults disable shorts and leverage above 1x.
- No API secrets: market discovery uses public CCXT endpoints only.

## Architecture implemented

| Layer | Current state |
|---|---|
| Core | Modular configuration, provider interface and documented service boundaries |
| Market data | Public OHLCV for Binance/MEXC, completed-candle filtering, CSV input |
| Market catalog | Canonical venue-independent instrument specifications and filtered instrument catalog; current live discovery is crypto spot only |
| Adapter registry | Venue adapters advertise asset-class/market-type capabilities; unsupported combinations fail closed |
| Quantitative | Causal technical features, MA and momentum baselines |
| AI | From-scratch HistGradientBoosting baseline; nested CLI evaluation compares it with a majority-class baseline; no model trained on startup or in CI |
| Evolution | Constrained candidate search on caller-supplied validation data |
| Risk | Independent trade risk gate, directional stops, leverage, drawdown and exposure checks |
| Portfolio | Gross, single-asset, correlated-group and position-count limits; now mandatory in combined decision gate |
| Backtesting | Bar-based simulator with strict index/input validation, fees, slippage, optional funding cost and equity accounting |
| Evaluation | Purged nested walk-forward, validation-only candidate selection, untouched test predictions, classification diagnostics, cost-aware spot and hypothetical long/short simulation, drawdown/Sharpe/Sortino/trade statistics, buy-and-hold and no-trade baselines |
| Decision | Probability-to-action conversion followed by independent trade-risk AND portfolio approval |
| API | Research FastAPI endpoints for risk, portfolio, combined decision and provider capability status; no order route |
| UI | Responsive PWA dashboard with combined decision gate simulator and provider capability display |
| Mobile | Browser-installable PWA foundation; native Android and on-device inference remain later milestones |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
uvicorn eti.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`; health check is `/health`.

Start nested from-scratch research evaluation only when you intentionally run the command (this fits models inside each fold):

```bash
python -m eti.pipeline --exchange binance --symbols BTC/USDT,ETH/USDT --timeframe 1h --limit 3000 --evaluation nested --output artifacts/eti_nested_walk_forward.json
```

The earlier single-symbol chronological holdout mode remains available:

```bash
python -m eti.pipeline --exchange binance --symbol BTC/USDT --timeframe 1h --limit 3000
```

There is no API endpoint to start training or place live orders. Nested reports include out-of-sample trading-performance diagnostics; default illustrative costs are 6 bps fee and 2 bps slippage per unit turnover, with zero funding cost unless explicitly changed in code. These are not exchange fee quotes. Spot metrics are long-only; the separate long/short simulation is hypothetical and omits borrow, margin and liquidation mechanics.

## Instrument-agnostic architecture

ETI is organized around asset classes and instrument types rather than broker names. The shared intelligence core is reusable; specialist market heads are introduced only when out-of-sample evidence supports them; broker-specific adapters normalize data and contract metadata. Adding a broker does not automatically require retraining the shared core. See [ARCHITECTURE.md](ARCHITECTURE.md) for the canonical instrument and adapter contracts.

A tokenized gold spot pair, an XAU/USD CFD and a COMEX gold future are distinct instruments with different pricing, contract and risk mechanics. The instrument catalog records those differences. Current Binance/MEXC connectivity remains public crypto spot only; the new registry is an extension point, not evidence that a particular gold or futures product is currently listed or accessible.

## Provider and device scope

- Binance/MEXC public connectivity is crypto spot only. Other labelled asset classes can currently be evaluated from validated local CSV OHLCV files; no live FX, equity, index or commodity feed is claimed.
- `eti/device_benchmark.py` measures inference latency and Python-traced allocations for an already-fitted model. Run it on the actual Android target; it is not a full native Android RAM/battery benchmark.
- `eti/model_benchmark.py` compares already-fitted models and does not train them.
- Nested evaluation selects candidates on validation windows and reports untouched outer test folds. Each outer test fold now includes classification metrics (per-class precision/recall/F1 and confusion matrix) plus cost-aware strategy metrics (net return, drawdown, annualized Sharpe/Sortino, exposure, turnover, trade count, win rate and profit factor). The spot result is long-only; a separate hypothetical long/short result is shown for research only. Buy-and-hold and no-trade baselines use the same test window and cost assumptions. Fold bootstrap intervals are descriptive and do not establish statistical significance or future profitability.

## Important limitations

1. Venue discovery currently covers crypto spot on Binance and MEXC only. Forex, equities, commodities, derivatives and additional brokers require separately validated adapters and instrument-specific metadata.
2. The baseline classifier is tabular ML, not a deep neural network or chart-vision model.
3. Evolution is a small transparent candidate selector, not an unconstrained self-modifying agent.
4. Backtesting is bar-based, not exchange-grade; it does not fully model queue position, partial fills, liquidation engines, borrow, contract multipliers or venue-specific funding.
5. Risk and portfolio modules are deterministic research gates, not a streaming production portfolio service with broker-side enforcement.
6. Multi-symbol nested evaluation is available for supported public crypto spot symbols; independent CSV datasets support other asset classes. Live non-crypto feeds and pooled portfolio simulation remain future work.
7. The PWA is an initial dashboard, not a finished native Android application. On-device inference requires target-device memory, latency and battery benchmarks.

See [ARCHITECTURE.md](ARCHITECTURE.md), [PLATFORM_ARCHITECTURE.md](PLATFORM_ARCHITECTURE.md), [ROADMAP.md](ROADMAP.md) and [KAGGLE.md](KAGGLE.md).


## Shared platform foundation

The shared-platform target contracts are documented in [PLATFORM_ARCHITECTURE.md](PLATFORM_ARCHITECTURE.md). A first persistent storage primitive is now available in `eti.storage.MemoryStore`:

- File-backed SQLite workspaces and JSON memory items, scoped by an owner identifier.
- Owner checks are applied in workspace and memory queries; cross-owner workspace writes fail closed.
- Append-only audit event insertion and owner-scoped audit reads.
- JSON payload validation, finite-number enforcement and a 64 KiB item limit.
- The store is not exposed through HTTP yet. The owner identifier must come from a verified authentication principal; clients must never be allowed to choose it.
- SQLite is a local/single-instance foundation. Multi-instance production deployment requires a PostgreSQL-backed implementation/migration before shared concurrent writes.
- Authentication, authenticated API routes, agent orchestration and model registry are not yet implemented.
