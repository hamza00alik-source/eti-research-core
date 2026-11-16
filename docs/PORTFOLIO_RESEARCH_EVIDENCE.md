# Synchronized Portfolio Research Evidence

The per-instrument nested research runner now preserves each untouched outer-test
bar's UTC timestamp, net strategy return and position. The
`eti.portfolio_quality` module combines those streams using exact
`venue:market_type:symbol` keys and fixed operator-supplied capital weights.

Example:

```bash
python -m eti.portfolio_quality artifacts/multi-market-research-result.json \
  --weights '{"coinbase:spot:BTC/USD":0.40,"kraken:spot:ETH/USD":0.40}' \
  --output artifacts/portfolio-quality-report.json \
  --minimum-folds 5
```

The remaining 20% is held as cash in this example. Weights are never
automatically normalized to 100%. For timestamps when a venue has no bar,
that instrument contributes zero incremental return while its last known
position is carried forward; weights are never silently reallocated.

## Reported evidence

- Per-fold compounded portfolio return and maximum drawdown.
- Annualized Sharpe using the configured bars-per-year assumption.
- Mean and maximum gross exposure, mean net exposure.
- Active-weight concentration (HHI), cross-instrument correlation matrix,
  and covariance-based variance risk contributions.
- Exact instrument weights, residual cash, and timestamp coverage.

## Limitations and authority

This is a synchronized outer-test research combination, not a production
portfolio simulator. It does not model shared capital constraints, order
queueing, partial fills, liquidity/market impact, margin, borrow, cross-currency
conversion, venue outages, or dynamic rebalancing. Independently trained
strategies may have been selected on separate data and are not a jointly
optimized portfolio. Correlations and risk contributions are descriptive;
few or overlapping folds are not independent statistical trials.

A report with enough folds is marked `human_review_required`, never approved.
It cannot promote models, change budgets, or enable execution. Legacy research
artifacts without timestamped bar returns and positions are rejected rather
than approximated from per-fold aggregate returns.
