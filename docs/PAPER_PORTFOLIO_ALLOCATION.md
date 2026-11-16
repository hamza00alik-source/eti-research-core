# Evidence-gated paper portfolio allocation

ETI exposes `portfolio.allocate` as an allowlisted Strategy department
capability. It creates a paper proposal only. It does not submit orders, approve
models, promote a candidate, or enable execution.

## Input contract

The JSON payload contains:

- `equity`: positive account equity in the quote currency used by all notionals.
- `candidates`: up to 200 distinct candidate symbols. Each candidate includes
  `strategy`, an operator-reviewed `correlation_group`, `mean_max_drawdown`,
  `stop_distance_fraction`, and the original `research_report` produced by
  ETI's multi-market research runner.
- `current_positions`: optional map of symbol to existing positive notional.
- `correlated_group_exposure`: optional map of correlation-group labels to
  already occupied notional.
- Optional risk limits: cash buffer (default 20%), gross exposure (25%), one
  asset (10%), one correlation group (15%), risk per trade (0.5%), and maximum
  positions (5).

The planner recomputes the economic disposition from each supplied research
report. A caller-supplied cached `economic_assessment` is ignored. Reports must
contain completed training evidence, matching fold counts, positive outer-test
returns at base and 2x costs, and every fold must beat no-trade and remain
positive under stress. At least two outer-test folds are required.

This is a conservative evidence screen, not a cryptographic authenticity
system. The calling workflow must verify report storage/provenance and bind the
report to its exact instrument, venue, market type, timeframe, model and
transaction-cost evidence before sending it to the planner.

## Allocation method

Eligible candidates are ranked by the lower of base-cost and 2x-cost mean net
return divided by mean maximum drawdown. Proposed notionals are capped by
stop-distance risk, per-asset exposure, correlation-group exposure, gross
exposure, cash reserve and position count. Limits are never relaxed to use all
available capital; unused budget remains cash.

Correlation groups are operator-supplied labels, not covariance estimates.
This planner does not estimate portfolio correlation, hedge exposures, rebalance
live holdings, or calculate realized execution costs.

## Authority and status

Every output includes `paper_only=true`, `orders_submitted=false`,
`owner_review_required=true`, and `execution_enabled=false`. Even
`proposals_ready` means only that a bounded paper proposal was produced. It is
not an instruction to trade or evidence of future profitability.


## Provenance and independent risk gate

Rule-strategy candidates must carry a research report whose venue, asset class,
market type, symbol and timeframe identify the same instrument as the candidate.
The report's latest price and data-end timestamp bind an owner-defined stop
record to that exact research snapshot. The allocator computes stop distance
from entry and stop prices; it does not trust a caller-supplied distance fraction.

Before serializing each proposed notional, ETI's independent deterministic
RiskEngine rechecks the request against account equity, stop distance, gross
exposure, position size and the configured per-trade risk limit. This remains
a paper proposal only. The current correlation-group label is still an
operator-supplied classification, not a covariance estimate, and must be
reviewed before any future portfolio automation.
