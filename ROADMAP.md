# ETI roadmap

## v0.1 — research baseline (current)
- [x] Public OHLCV fetch for Binance and MEXC through CCXT
- [x] OHLCV validation and causal technical features
- [x] Volatility-scaled three-class forward-return labels
- [x] Chronological train/validation/test workflow
- [x] Small baseline classifier and long/flat cost-aware paper simulation
- [x] Unit smoke tests and GitHub Actions workflow
- [x] Kaggle run instructions

## v0.2 — evaluation integrity
- [ ] Purged/embargoed walk-forward validation
- [ ] Multi-symbol and multiple market-regime evaluation
- [ ] Funding, spread, fees, and liquidity-aware cost model
- [ ] Better fill assumptions and explicit position sizing
- [ ] Experiment configuration and reproducible artifact registry

## v0.3 — specialist model research
- [ ] Train-from-scratch MLP and 1D-CNN baselines
- [ ] GRU/compact attention experiments only if they beat simpler baselines out of sample
- [ ] Regime and volatility specialist models
- [ ] Calibration, uncertainty, and no-trade thresholds

## v0.4 — evolutionary research
- [ ] Feature subset and feature-generation search
- [ ] Strategy parameter evolution and multi-objective selection
- [ ] Diversity preservation, robust out-of-sample selection, and experiment lineage

## Later — mobile and visual intelligence
- [ ] Benchmark quantized local inference on target Android hardware
- [ ] Export eligible models to ONNX or LiteRT
- [ ] Separate chart-image dataset, labeling, and vision model
- [ ] Dashboard/PWA and mobile research interface

Live execution is intentionally not part of these milestones. Any future execution layer must be separate from model training and protected by independent, non-learned risk controls.
