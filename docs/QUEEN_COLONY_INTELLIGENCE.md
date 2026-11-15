# Queen Colony Intelligence

The Queen now has a bounded, read-only operational evidence report built from
the current owner's workspace task history.

## What it measures

- Task counts by lifecycle status.
- Capability-level completed/failed counts and success rate.
- Wilson 95% interval for the observed success rate.
- Retry counts and median end-to-end task age when timestamps are valid.
- A minimum sample threshold before a capability is marked eligible for
  reliability-informed routing.

## What it does not measure

This is not model quality, strategy quality, predictive calibration, net
profitability, or an individual worker score. Task age includes queue waiting
time. Worker identity is not currently persisted, so per-worker comparisons
would be fabricated. Samples below the configured threshold are descriptive
only.

The report is observational and has no authority to approve models, increase
research spend, change risk limits, or enable order execution. Model promotion
continues through the independent model quality gate; financial risk and
execution permissions remain separate.

The report reads at most 500 recent owner/workspace tasks. If the history
reaches that cap, it marks the report as potentially truncated.
