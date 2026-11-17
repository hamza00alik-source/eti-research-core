# Multi-market market-intelligence validation

The research-only multi-market validator is exposed as
`eti.multi_market_intelligence.train_multi_market_intelligence_specialists`.
It accepts validated `MarketDataSnapshot` objects and an explicit
`independence_group` for each underlying.

Each instrument receives its own feature/target construction, chronological
purged folds, classifier, volatility estimator, naive baselines and data/code
lineage. No observations or labels are pooled across instruments. Duplicate
instrument identities and repeated underlying groups are rejected. All
snapshots must use the same timeframe.

The total fit-call budget is preflighted across every market before the first
estimator fit. Per-market results retain their own fold metrics, class support,
coverage, baseline comparisons, source provenance and quality metadata.
Cross-market summaries are descriptive medians of per-market metrics; folds
are not pooled, and no confidence interval is calculated from overlapping
folds. A caller-declared underlying group is the replication unit. The default
minimum is three independent groups; below that threshold the report states
`insufficient_independent_markets`.

This is not a model-selection or promotion gate. Three markets are a minimum
for describing cross-market replication, not proof of statistical power.
Class calibration/uncertainty, broader regime coverage, independent economic
evaluation after realistic fees/slippage, multiple-testing controls and
independent approval remain outstanding. The module never serializes models,
updates the registry, activates inference, grants Queen authority or enables
order execution.
