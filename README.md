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
| Market catalog | Public crypto spot market discovery and normalized instrument descriptors |
| Quantitative | Causal technical features, MA and momentum baselines |
| AI | From-scratch HistGradientBoosting baseline; nested CLI evaluation compares it with a majority-class baseline; no model trained on startup or in CI |
| Evolution | Constrained candidate search on caller-supplied validation data |
| Risk | Independent trade risk gate, directional stops, leverage, drawdown and exposure checks |
| Portfolio | Gross, single-asset, correlated-group and position-count limits; now mandatory in combined decision gate |
| Backtesting | Bar-based simulator with strict index/input validation, fees, slippage, optional funding cost and equity accounting |
| Evaluation | Purged walk-forward, nested candidate selection on validation windows, untouched outer test reporting, multi-market runner and descriptive fold bootstrap intervals |
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

There is no API endpoint to start training or place live orders.

## Provider and device scope\n\n- Binance/MEXC public connectivity is crypto spot only. Other labelled asset classes can currently be evaluated from validated local CSV OHLCV files; no live FX, equity, index or commodity feed is claimed.\n- `eti/device_benchmark.py` measures inference latency and Python-traced allocations for an already-fitted model. Run it on the actual Android target; it is not a full native Android RAM/battery benchmark.\n- `eti/model_benchmark.py` compares already-fitted models and does not train them.\n- The walk-forward runner is an execution harness, not nested model selection or proof of statistical significance.\n\n## Important limitations

1. Venue discovery currently covers crypto spot on Binance and MEXC only. Forex, equities, commodities, derivatives and additional brokers require separately validated adapters and instrument-specific metadata.
2. The baseline classifier is tabular ML, not a deep neural network or chart-vision model.
3. Evolution is a small transparent candidate selector, not an unconstrained self-modifying agent.
4. Backtesting is bar-based, not exchange-grade; it does not fully model queue position, partial fills, liquidation engines, borrow, contract multipliers or venue-specific funding.
5. Risk and portfolio modules are deterministic building blocks, not yet a streaming production portfolio service with broker-side enforcement.
6. Walk-forward utilities create purged chronological folds; full multi-market walk-forward experiment orchestration and statistical significance reporting remain future work.
7. The PWA is an initial dashboard, not a finished native Android application. On-device inference requires target-device memory, latency and battery benchmarks.

See [ROADMAP.md](ROADMAP.md) and [KAGGLE.md](KAGGLE.md).
