# Research Quality Evidence Report

eti.research_quality converts an already completed
eti.multi_market_research_result.v1 artifact into a conservative,
instrument-by-instrument summary of nested outer-test evidence.

Run:

```bash
python -m eti.research_quality artifacts/multi-market-research-result.json \
  --output artifacts/research-quality-report.json \
  --minimum-folds 5
```

The report includes mean/median net return, positive-return folds, benchmark
comparisons, worst and mean drawdown, annualized Sharpe, balanced accuracy,
and the 2x fee/slippage sensitivity result. Failed instruments are reported
as failed; missing or malformed evidence is rejected.

## Interpretation and authority

- Fold-level results are descriptive; overlapping chronological folds are
  dependent and must not be treated as independent trials.
- A positive 2x-cost summary only earns the status human_review_required,
  never an automatic approval.
- Fewer than the configured minimum folds is insufficient_folds.
- Non-positive mean net return at 2x costs is not_robust_at_2x_costs.
- The 2x stress scales the modeled fee and slippage assumptions. It does not
  prove actual fill quality or model all market impact.
- These are independent instrument studies, not a portfolio backtest.
- This report cannot promote models, change budgets, enable execution, or
  establish future profitability.

The Queen's operational task-reliability intelligence remains a separate
evidence stream and must never be mixed with this research-quality report.


## Probability-calibration evidence

When a completed research artifact includes fold-local calibration diagnostics,
the quality report also validates the aggregate calibration summary against
the per-fold records. It reports how many outer-test folds were assessed and
how many independently confirmed calibration, alongside raw and calibrated
NLL and Brier-score means. An inconsistent or fabricated aggregate is rejected.

Legacy research artifacts without this evidence are explicitly marked
`not_available`; they are not silently treated as calibrated. Calibration
status is descriptive and does not change the trading evidence status or
grant model-promotion, budget, or execution authority.
