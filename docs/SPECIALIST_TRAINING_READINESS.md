# ETI specialist model training and readiness plan

## Design rule

ETI uses a shared data, feature, evidence and lifecycle foundation, with
separate task-specific models only where learned predictions add value. A
single universal model is not used for every department. Deterministic controls
remain authoritative where they are safer and more auditable than a prediction.

## Department readiness

| Department | Active foundation | Learned-model status | Additional evidence required |
|---|---|---|---|
| Market data | Instrument discovery, provenance, candle quality, session-aware validation | No ML model required for core controls | More verified venue/session calendars and licensed data adapters |
| Market intelligence | Causal regime diagnostics and derivative context; research-only training/evaluation pipeline for two distinct roles | `regime_classifier` and `volatility_estimator` have candidate training/evaluation pipelines; no promoted or inference-enabled models | Multi-market replication, robust per-class coverage, calibration/uncertainty review, baseline superiority and independent approval |
| Prediction | BTC spot direction classifier, immutable artifact lineage and scoped inference contract | One research candidate; not economically viable in the first recovered run | Fresh post-lineage run, multi-instrument evidence, positive cost-adjusted OOS results and independent review |
| Strategy | Validation-only strategy selection and evidence-gated paper portfolio allocation | No separately trained strategy selector yet | Multiple independent instruments, untouched tests, stable performance after costs and regime stratification |
| Risk | Independent deterministic risk gate | Optional risk forecast is not required and cannot override the gate | Only consider a forecast model after enough independently observed risk/outcome samples |
| Macro | Point-in-time release/vintage selection with license checks | `macro_regime` and `event_impact` not trained | Licensed provider adapter, historical releases and revisions with publication timestamps |
| Execution | Evidence-backed cost estimation; no order submission | Fill-probability and slippage ML models not trained | Timestamped paper/realized fills, quotes, order-book depth, fee tiers and instrument-specific execution records |
| Learning | Economic disposition, evidence reports, immutable model versions and Queen readiness telemetry | Drift detector remains optional | Stable champion history, time-indexed predictions/outcomes and minimum sample thresholds |

## Why some departments do not need a model

- Market-data validation must remain deterministic. A learned model cannot make
  missing, stale, unlicensed or malformed source data valid.
- Risk limits must remain independent and fail closed. Optional forecasts may
  inform sizing research but cannot override hard limits.
- Execution permissions and order validation must remain deterministic.
  Predictive fill/slippage models may be added only after enough quality
  execution observations exist.
- Learning and promotion are evidence-governance responsibilities; the Queen
  must not self-approve models or increase financial authority.

## Training and promotion rules

1. Train each learned role as a separate candidate with an exact department,
   role, asset class, market type and timeframe scope.
2. Share causal feature contracts, not model weights or assumptions that all
   instruments behave alike.
3. Keep label horizons, purge gaps, cost assumptions and data cutoffs in
   immutable experiment lineage.
4. Compare against simple baselines on chronological validation and untouched
   outer-test windows. Include 2x cost stress and regime-stratified results.
5. Store candidates with content-addressed artifact versions and code,
   configuration and dataset hashes.
6. A candidate is not a champion. Require independent evidence and human
   approval before approved inference. No result in this plan enables live
   trading.

## Market-intelligence specialist pipeline

PR #203 adds a research-only training/evaluation pipeline for `regime_classifier`
and `volatility_estimator`. It constructs causal features, forward regime and
forward realized-volatility targets, expanding chronological outer folds with
horizon-sized purge gaps, and naive baselines. The fit-call cap is checked
before fitting, and reports include code/config/data lineage. The pipeline
returns metrics only: it does not serialize or register models, promote
candidates, enable inference, or grant Queen authority. Adjacent expanding
folds may overlap and are not independent replications. Single-instrument
results are insufficient for promotion; class coverage and calibration /
uncertainty must be reviewed across multiple eligible instruments and regimes.

## Current first-run evidence

The recovered 2026-11-16 BTC/USD 1h run completed training and calibration but
had negative mean net return, zero positive outer-test folds, and worse mean
net return under 2x illustrative costs. It remains `no_go`. Its artifact
predates the immutable experiment-lineage changes. The Kaggle notebook now
asserts department, role, exact scope, content-addressed version and lineage
metadata; a new intentional run is required to produce a post-change artifact.

## External blockers

- A fresh Kaggle run requires explicit manual workflow confirmation and a
  review of Kaggle's own billing display. ETI's configured limits are USD 10
  per day and USD 100 per month, but notebook code cannot read or enforce
  Kaggle billing.
- Macro ML cannot be trained honestly until a licensed historical provider with
  release and revision timestamps is integrated.
- Execution ML cannot be trained honestly without sufficiently rich,
  timestamped order/quote/fill observations.
- No live execution is enabled. These prerequisites are not bypassed by
  synthetic fixtures or fabricated training metrics.
