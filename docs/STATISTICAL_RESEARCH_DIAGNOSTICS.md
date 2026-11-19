# Statistical research diagnostics

ETI provides four complementary diagnostics for a declared family of aligned
strategy return series. They are research outputs only: they do not promote a
model, allocate capital, or authorize orders.

## Input contract

Inputs are net or differential returns aligned on identical timestamps, one row
per tested strategy/configuration and one column per observation. Do not drop
missing observations independently by strategy. Build a common timestamp
intersection upstream and disclose lost observations. Include every strategy,
instrument, timeframe and material parameter trial; undisclosed trials make
multiple-testing adjustments optimistic.

## Diagnostics

- **Deflated Sharpe Ratio / Probabilistic Sharpe Ratio:** estimates the
  probability that the selected strategy's Sharpe exceeds a hurdle based on
  the expected maximum Sharpe across the declared trial family. It uses sample
  skewness and Pearson kurtosis and depends on distributional assumptions.
- **PBO / CSCV:** divides history into contiguous blocks, evaluates every
  half-in-sample/half-out-of-sample split, selects the best in-sample mean
  return and measures how often its out-of-sample rank is at or below median.
  It does not replace chronological walk-forward testing.
- **White Reality Check:** circular moving-block bootstrap of the maximum
  studentized strategy-minus-benchmark mean under a joint null.
- **Hansen SPA-style test:** consistent null recentering removes clearly
  inferior alternatives before the maximum studentized block bootstrap.

## Interpretation and guardrails

No single threshold establishes a durable edge. Report raw net returns,
chronological fold results, drawdown, turnover, capacity, liquidity, benchmark
comparisons and cost assumptions beside these diagnostics. Freeze the trial
family before examining the holdout; repeatedly reusing a holdout invalidates
its confirmatory role. Bootstrap conclusions depend on block length and
stationarity.

These functions are not wired into model approval or capital allocation in
this change. A human reviewer must inspect raw data and assumptions. Live
execution remains disabled.
