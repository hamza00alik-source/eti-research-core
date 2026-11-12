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
| AI | From-scratch HistGradientBoosting baseline; no model trained by default |
| Evolution | Constrained candidate search on caller-supplied validation data |
| Risk | Independent trade risk gate, directional stops, leverage, drawdown and exposure checks |
| Portfolio | Gross, single-asset, correlated-group and position-count limits |
| Backtesting | Next-bar bar-based simulator, fees, slippage, optional funding-cost input, short opt-in |
| Evaluation | Purged expanding-window walk-forward split utility and baseline comparison |
| Decision | Probability-to-action conversion followed by risk approval |
| API | Read-only FastAPI service and risk-assessment endpoint |
| UI | Responsive PWA shell with service-worker caching |
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

Start baseline training only when the development/evaluation phase is complete and training is explicitly approved:

```bash
python -m eti.pipeline --exchange binance --symbol BTC/USDT --timeframe 1h --limit 3000
```

There is no API endpoint to start training or place live orders.

## Important limitations

1. Venue discovery currently covers crypto spot on Binance and MEXC only. Forex, equities, commodities, derivatives and additional brokers require separately validated adapters and instrument-specific metadata.
2. The baseline classifier is tabular ML, not a deep neural network or chart-vision model.
3. Evolution is a small transparent candidate selector, not an unconstrained self-modifying agent.
4. Backtesting is bar-based, not exchange-grade; it does not fully model queue position, partial fills, liquidation engines, borrow, contract multipliers or venue-specific funding.
5. Risk and portfolio modules are deterministic building blocks, not yet a streaming production portfolio service with broker-side enforcement.
6. Walk-forward utilities create purged chronological folds; full multi-market walk-forward experiment orchestration and statistical significance reporting remain future work.
7. The PWA is an initial dashboard, not a finished native Android application. On-device inference requires target-device memory, latency and battery benchmarks.

See [ROADMAP.md](ROADMAP.md) and [KAGGLE.md](KAGGLE.md).
