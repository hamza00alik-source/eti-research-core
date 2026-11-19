# ETI statistical validation (Phase 2)

## Purpose

Phase 2 adds multiple-testing-aware diagnostics to the research plane. These methods are evidence filters, not return forecasts, guarantees or order authorization. They must be applied to aligned, point-in-time, net-of-realistic-cost returns and to a fully disclosed family of strategies/parameters that were actually explored.

## 1. Deflated Sharpe Ratio (DSR)

Call `deflated_sharpe_ratio(returns, trials=N)` with one-dimensional per-period arithmetic returns and the total number of attempted strategy/parameter/model trials. The function reports a non-annualized sample Sharpe, sample skewness and Pearson kurtosis, the expected maximum Sharpe benchmark for the declared trial family, and the probability that the observed Sharpe exceeds that benchmark.

The trial count is a governance input. Omitting abandoned or failed experiments makes the result optimistic. The implementation assumes the supplied observations are suitable for the standard Sharpe sampling approximation; serial dependence, nonstationarity and unreported searches are not repaired by the function. Use block-aware inference and chronological out-of-sample evidence alongside it.

## 2. Probability of Backtest Overfitting (PBO / CSCV)

Call `probability_of_backtest_overfitting(returns_matrix, n_splits=8)` with a matrix shaped observations by strategies. The function partitions the timeline into equal contiguous blocks, enumerates half-block in-sample / complementary out-of-sample combinations, selects the in-sample winner, ranks that winner out of sample and calculates the fraction of logit ranks at or below zero.

CSCV requires equal-sized blocks and aligned return series. The default split cap prevents accidental combinatorial explosion. If the selected split count does not divide the observation count evenly, the integrated family assessment marks PBO as not computed and still returns White Reality Check and SPA; it does not discard otherwise usable family diagnostics. PBO measures how often the selected in-sample winner falls below the OOS median across the symmetric partitions; it is not a probability that a live strategy will fail.

## 3. White Reality Check and Hansen SPA

Call `white_reality_check(differentials)` and `hansen_spa_test(differentials)` with a matrix of aligned, per-period, net-of-cost strategy return differentials against the same benchmark. Both use a circular moving-block bootstrap to preserve local serial dependence and cross-strategy dependence. The block length must be chosen before inspecting the result and should be justified for the data frequency and dependence structure.

- White's Reality Check tests the maximum mean benchmark-relative differential across the supplied family. It can be conservative when many alternatives are poor.
- Hansen SPA uses studentized statistics and a consistent lower-tail recentering step to reduce the influence of clearly poor alternatives. The implementation is a transparent SPA-style bootstrap diagnostic; it is not claimed to provide exact finite-sample inference.

A small p-value is evidence against the joint no-positive-performance null under the supplied benchmark, family and bootstrap design. It does not establish that costs, data or family declarations are correct.

## Reproducibility and review

All bootstrap routines accept an explicit seed and return it with the report. The p-value uses the finite-sample correction (exceedances + 1) / (repetitions + 1), so it is never reported as exactly zero. Increase repetitions for final research review and record the selected block length before comparing candidates.

## Integrated family assessment

Call `assess_research_trial_family(family_snapshot, selected_trial_id=...)` with the object returned by the ledger matrix accessor. It applies the DSR estimator to the selected trial's net benchmark-relative differential (therefore an information-ratio basis) using the full attempted-trial count, and runs PBO, White Reality Check and SPA across all completed aligned columns. It preserves the family benchmark and dataset/cost hashes in the report. If the selected trial has zero volatility, the family report marks its DSR as undefined rather than blocking PBO and bootstrap diagnostics; zero-return cash comparators are valid for family PBO and benchmark-relative tests.

## Integration boundary

The integrated family assessment is now wired to the SQLite experiment ledger, the authenticated workspace-scoped diagnostics API and the approved research runner. The runner records candidate-specific outer-test return differentials against a frozen no-trade benchmark; it does not derive the matrix from fold-level summary metrics. Registered-but-unstarted hypotheses remain visible but are excluded from attempted-trial counts, while every started attempt—including failed attempts—is counted for DSR.

These utilities remain separate from model promotion and order execution. They do not authenticate external source data, independently verify cost assumptions, prove that every external experiment was registered, or authorize promotion. SQLite is a single-instance store; multi-replica deployment requires a PostgreSQL migration.

## Safety

- These functions do not submit, simulate or authorize orders.
- No automatic model promotion is added.
- Live order execution remains hard-disabled.
- A candidate must continue to pass chronological walk-forward, leakage, calibration, economic and human-governance checks independently.
