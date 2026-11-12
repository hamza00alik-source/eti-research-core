# ETI roadmap

## v0.2 — architecture foundation
- [x] Modular configuration and public market-data provider interface
- [x] Binance/MEXC public crypto spot discovery
- [x] OHLCV ingestion, validation and closed-candle filtering
- [x] Causal feature and label pipeline
- [x] Baseline classifier (training remains opt-in)
- [x] Transparent MA, momentum and buy-and-hold strategies
- [x] Bar-based backtest with explicit costs and optional funding-cost input
- [x] Independent deterministic trade risk gate
- [x] Portfolio gross/single-asset/correlation/position-count checks
- [x] Risk-gated research decision orchestration
- [x] Candidate strategy search and purged walk-forward split utility
- [x] Read-only FastAPI service and responsive PWA shell
- [x] Expanded automated tests and CI workflow
- [ ] Confirm latest CI run is green and merge reviewed pull request

## v0.3 — evaluation integrity
- Full walk-forward experiment runner with nested selection and untouched test folds
- Multi-symbol and multi-regime evaluation
- Benchmark against buy-and-hold and naive baselines on identical samples
- Stress fees, slippage, funding, gaps and missing data
- Bootstrap confidence intervals, calibration and experiment registry
- No model or strategy promotion from a single holdout

## v0.4 — specialist intelligence
- Train-from-scratch specialist numerical models after the evaluation harness is complete
- Regime, trend, mean-reversion, volatility and risk specialists
- Probability calibration and uncertainty estimation
- Controlled evolutionary candidate generation with immutable risk constraints

## v0.5 — portfolio and venue expansion
- Forex, equities, commodities and derivatives provider adapters
- Venue-specific contract metadata, funding, margin and liquidation simulation
- Portfolio correlation estimation, concentration and dynamic exposure controls
- Broker/exchange eligibility and capability matrix

## v0.6 — mobile and visual intelligence
- Native Android client and device inference benchmark on target hardware
- Export only supported models to mobile runtime after latency/RAM/battery tests
- Separate chart-image model research; numerical validation remains authoritative
- Complete PWA install/offline behavior and accessibility

## v1.0 — guarded autonomous paper research
- Autonomous market/instrument/strategy selection in paper mode
- Monitoring, audit log, kill switch and reproducible decision trace
- Independent risk approval for every simulated decision
- Live execution remains a separate explicitly reviewed milestone and disabled by default
