# ETI roadmap

## v0.2 — architecture foundation (in progress)
- [x] Public OHLCV data ingestion and closed-candle filtering
- [x] Causal feature and label pipeline
- [x] Baseline classifier and chronological holdout
- [x] Transparent strategy baselines and backtest costs
- [x] Independent deterministic risk gate
- [x] Research decision orchestration
- [x] Initial strategy candidate search
- [x] Read-only FastAPI service and PWA shell
- [x] Mobile-browser dashboard foundation
- [ ] CI green on the complete architecture branch
- [ ] Integrate walk-forward folds, realistic funding/contract models and robust benchmark reporting

## v0.3 — evaluation integrity
- Nested walk-forward validation and embargo/purge controls
- Multiple symbols and market regimes
- Benchmark against buy-and-hold and naive baselines
- Stress fees, slippage, funding, gaps and missing data
- Robustness tests, bootstrap confidence intervals and experiment registry
- No model or strategy promotion based on a single holdout

## v0.4 — specialist intelligence
- Train-from-scratch specialist numerical models only after evaluation harness is complete
- Regime detection, trend, mean-reversion, volatility and risk specialists
- Probability calibration and uncertainty estimation
- Controlled evolutionary candidate generation with immutable risk constraints

## v0.5 — portfolio and venue expansion
- Provider interfaces for forex, equities, commodities and derivatives
- Venue-specific contract metadata, funding, margin and liquidation simulation
- Portfolio-level correlation, concentration and exposure controls
- Broker/exchange eligibility and capability matrix

## v0.6 — mobile and visual intelligence
- Benchmark model size, latency, RAM and battery on target Android devices
- Export only supported models to a mobile runtime after benchmark
- Separate chart-image model research; never replace numerical validation with screenshot interpretation
- Complete PWA install/offline behavior and accessibility

## v1.0 — guarded autonomous paper research
- Autonomous market/instrument/strategy selection in paper mode
- Monitoring, audit log, kill switch and reproducible decision trace
- Independent risk approval for every simulated decision
- Live execution remains a separate explicitly reviewed project milestone and is disabled by default
