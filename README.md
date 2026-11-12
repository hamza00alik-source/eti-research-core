# ETI Research Core

**ETI (Evolutionary Trading Intelligence)** is an experimental, research-first trading intelligence project.

## Current status

The repository is being built in stages. The first milestone is a reproducible, runnable research baseline for public crypto market data, causal features, chronological model evaluation, and conservative paper simulation.

- Research and paper trading only.
- No exchange order execution or withdrawal functionality.
- No claim of profitability. Backtests and model scores are not evidence of future returns.
- Risk controls must remain independent of learned models.

## Planned first milestone

1. Fetch public OHLCV candles from supported exchanges through CCXT.
2. Validate and normalize timestamps and candle data.
3. Generate causal technical and market features.
4. Create volatility-scaled forward-return labels.
5. Train a small baseline classifier using chronological splits.
6. Evaluate out-of-sample predictions with fees and slippage in a deliberately simple paper simulator.
7. Add smoke tests and a Kaggle-friendly run path.

## Development principles

- Avoid look-ahead leakage: features at time *t* may use only information available at or before *t*.
- Split data chronologically; never randomly shuffle time series for evaluation.
- Include trading costs in every simulation.
- Keep model training, decision logic, risk controls, and execution boundaries separate.
- Prefer reproducible experiments, saved configurations, and explicit limitations.

## Intended architecture

`eti/data.py` — public market data collection and validation  
`eti/features.py` — causal feature generation and labels  
`eti/model.py` — baseline model training and prediction  
`eti/simulator.py` — conservative research-only paper simulation  
`eti/pipeline.py` — end-to-end experiment orchestration

## Quick start

The runnable commands and dependency list will be added with the initial pipeline implementation. Python 3.10+ is the target.

## Safety

Do not connect live trading keys to this research baseline. When execution is eventually considered, it must be a separate, disabled-by-default component with strict independent risk limits, explicit user authorization, and keys that cannot withdraw or transfer funds.
