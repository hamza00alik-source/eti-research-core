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
- [x] Confirm CI green on merged main commit and merge reviewed pull request

## v0.3 — evaluation integrity
- [x] Executable purged expanding-window runner with caller-supplied fit/predict callback
- [x] Test folds reported separately; test labels are never supplied to the validation prediction call
- [x] Identical-sample strategy baselines and hardened cost-aware backtest
- [x] Nested candidate selection using validation-only scores and untouched outer test folds
- [x] Multi-symbol nested research pipeline for supported crypto spot feeds
- [x] Descriptive bootstrap confidence intervals across fold-level metrics
- [x] Untouched outer-test predictions retained for trading diagnostics
- [x] Per-class precision/recall/F1 and confusion-matrix reporting
- [x] Cost-aware long-only spot simulation and separate hypothetical long/short results
- [x] Net return, drawdown, Sharpe/Sortino, exposure, turnover, trade count, win rate and profit factor
- [x] Same-window buy-and-hold and no-trade baselines
- [ ] Market-regime stratification, experiment registry and stronger dependence-aware statistical inference

## v0.4 — specialist intelligence
- Train-from-scratch specialist numerical models after the evaluation harness is complete
- Regime, trend, mean-reversion, volatility and risk specialists
- Probability calibration and uncertainty estimation
- Controlled evolutionary candidate generation with immutable risk constraints

## v0.5 — portfolio and venue expansion
- [x] Extensible asset-class provider protocol, registry and CSV history adapter for crypto/forex/equities/commodities/indices
- [x] Canonical instrument specifications and venue-independent adapter capability registry
- [x] Explicit shared-core / specialist-head / broker-adapter architecture contract
- [ ] Live forex, equities, commodities and derivatives provider adapters
- Venue-specific contract metadata, funding, margin and liquidation simulation
- Portfolio correlation estimation, concentration and dynamic exposure controls
- Broker/exchange eligibility and capability matrix

## v0.6 — mobile and visual intelligence
- [x] Inference-only latency and Python-allocation benchmark utilities for execution on target devices
- [ ] Native Android client and device inference benchmark on target hardware
- Export only supported models to mobile runtime after latency/RAM/battery tests
- Separate chart-image model research; numerical validation remains authoritative
- Complete PWA install/offline behavior and accessibility

## v1.0 — guarded autonomous paper research
- Autonomous market/instrument/strategy selection in paper mode
- Monitoring, audit log, kill switch and reproducible decision trace
- Independent risk approval for every simulated decision
- Live execution remains a separate explicitly reviewed milestone and disabled by default
