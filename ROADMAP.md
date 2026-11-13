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


## v0.7 — shared multi-client platform foundation
- [x] Define one backend as the source of truth for Android and web/PWA
- [x] Specify account-scoped shared memory and typed agent-to-agent task/message flow
- [x] Specify central model registry, immutable versions and gated promotion lifecycle
- [x] Define stable client-neutral API families and backend-only market plug-in contract
- [x] Define authentication, authorization, encryption, audit and credential-handling requirements
- [x] Add file-backed SQLite workspace/memory store with owner-scoped queries and isolation tests
- [x] Add authenticated API access for account-scoped workspaces, memory, agent ledgers, audit and model metadata
- [ ] Add explicit data export, retention and deletion policy
- [ ] Replace SQLite with production PostgreSQL before multi-instance deployment
- [x] Implement invite-only authenticated sessions, signed short-lived tokens, persistent login/signup throttles and scoped authorization
- [x] Implement durable SQLite agent task/message queue, atomic leases, lease-token completion, bounded retries and audit events
- [x] Implement admin-governed immutable model metadata registry with independent approval and retirement
- [x] Implement server-owned content-addressed model artifact storage and authenticated inference for independently approved versions
- [ ] Add inference runtime monitoring, drift detection and model health alerts
- [x] Migrate PWA account, workspace, memory and agent task screens to authenticated APIs
- [ ] Migrate market, instrument, provider, risk and portfolio resources to stable versioned APIs
- [ ] Add encrypted provider credential vault and read-only connection verification
- [ ] Build native Android client against the same API contract


## v0.8 — durable worker and registry services
- [x] Add a separate worker process; API startup does not start workers implicitly
- [x] Add an allowlisted public crypto spot market-snapshot capability (Binance/MEXC)
- [x] Add crash recovery through expiring leases, random lease tokens and bounded retry attempts
- [x] Prevent clients from forging task completion; users may cancel queued tasks only
- [x] Expose approved model metadata to signed-in accounts and restrict registry changes to server-listed administrators
- [x] Add explicit opt-in final-model artifact export to the training CLI (exports remain uncalibrated)
- [ ] Implement time-series probability calibration and calibration report validation before model approval
- [ ] Add general multi-agent planning/delegation and versioned capability permissions
- [x] Add model artifact persistence, checksum verification and approved-version inference
- [ ] Add PostgreSQL queue/storage backend and worker health monitoring
- [ ] Deploy HTTPS API + worker + PWA and verify end-to-end Android browser installation
