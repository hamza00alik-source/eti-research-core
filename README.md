# ETI — Evolutionary Trading Intelligence

ETI is a research-first quantitative and AI trading intelligence project. The current v0.2 architecture is designed for reproducible experiments, deterministic risk checks, transparent strategy comparisons, and a mobile-friendly research dashboard.

## Safety status

- **Research-only:** no live order placement routes or authenticated exchange actions exist.
- **Training is not started automatically:** data download, feature generation, model fitting and evaluation are explicit commands.
- **No profitability claim:** backtests and model metrics are experimental and do not establish future performance.
- **Risk is independent:** model decisions must pass the deterministic risk gate; risk defaults disable shorts and leverage above 1x.
- **No API secrets:** public market discovery uses CCXT public endpoints only.

## Architecture

| Layer | Current implementation |
|---|---|
| Market data | CCXT OHLCV for Binance/MEXC, closed-candle filtering, CSV input |
| Market catalog | Public spot-market normalization for Binance/MEXC |
| Quantitative features | Causal technical feature set |
| Baseline ML | HistGradientBoostingClassifier, trained from scratch when explicitly run |
| Strategy library | Buy-and-hold, moving average and momentum baselines |
| Backtesting | Next-bar execution, fees, slippage, optional funding and explicit short opt-in |
| Evolution | Small transparent parameter search on a caller-supplied validation slice |
| Risk | Independent deterministic position, exposure, drawdown and leverage checks |
| Decision | Probability-to-action mapping followed by mandatory risk approval |
| API | Read-only research endpoints and risk-assessment endpoint |
| Interface | Responsive PWA shell with service-worker caching |
| Mobile | PWA works in mobile browsers; native Android/local inference is a later milestone |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

Run the research API and PWA:

```bash
uvicorn eti.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. API health: `/health`.

Run the baseline training pipeline only when intentionally starting the training phase:

```bash
python -m eti.pipeline --exchange binance --symbol BTC/USDT --timeframe 1h --limit 3000
```

The API does not expose a training or live execution endpoint.

## Current limitations

1. Market discovery currently covers crypto spot on Binance and MEXC only. Forex, equities, commodities, derivatives and additional brokers need separate vetted adapters and instrument-specific contract metadata.
2. The ML baseline is a conventional tabular classifier, not a deep neural network or chart-vision model.
3. The evolutionary search is a small transparent validation-slice selector, not an unconstrained self-modifying agent.
4. The backtester is bar-based and not exchange-grade. It does not model queue position, partial fills, liquidation engine details, borrow availability or all venue-specific funding rules.
5. The risk gate is a deterministic pre-trade calculator; portfolio-wide streaming controls and broker-side enforcement are future work.
6. The PWA is an initial dashboard shell, not a finished native Android application. Local inference requires model format, latency, memory and battery benchmarks on target devices.

See [ROADMAP.md](ROADMAP.md) and [KAGGLE.md](KAGGLE.md).
