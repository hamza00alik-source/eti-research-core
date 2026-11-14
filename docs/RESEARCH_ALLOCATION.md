# ETI research allocation

The research allocator converts a completed `eti.market_universe.v1` scan into a
bounded, deterministic research-priority plan. It does not estimate returns,
generate trading signals, fit models, or submit orders.

## What the ranking means

Candidates are ranked only on:
- validated history coverage (up to 40 points),
- last-candle freshness (up to 30 points),
- continuity / estimated missing intervals (up to 30 points).

Stale data, excessive gaps, invalid OHLCV quality, or missing provenance block a
candidate from the selected research queue. Ties are resolved deterministically
by asset class, venue, and symbol. A score is a data-readiness score, not alpha.

Transaction-cost eligibility is separate. Unless a candidate has an explicitly
validated cost model, it is not eligible for economic evaluation. Do not treat
unknown fees, spreads, funding, borrow, roll, or market-impact costs as zero.
Yahoo reference data remains reference-only and is not execution-grade.

## Create a plan

First create a market-universe scan using the command in
[Multi-market data](MULTI_MARKET_DATA.md), then allocate research capacity:

```bash
python -m eti.research_allocator artifacts/market-universe.json \
  --max-model-fit-calls 32 \
  --fits-per-candidate 2 \
  --max-candidates 8 \
  --estimated-run-cost-usd 2.00 \
  --daily-spend-to-date-usd 0.00 \
  --monthly-spend-to-date-usd 10.00 \
  --cost-models artifacts/validated-cost-models.json \
  --output artifacts/research-allocation.json
```

The cost-model JSON is a mapping from `venue:symbol` to the literal status
`validated`, for example:

```json
{
  "kraken:BTC/USD": "validated"
}
```

The cost-model file must be backed by measured or documented, instrument- and
venue-specific fees, spread, slippage, funding/borrow where applicable, contract
multipliers, and market-session assumptions. The allocator does not validate
the underlying economics merely because the status string is supplied; that
review is an upstream governance responsibility.

## Hard limits and fail-closed behavior

- Maximum estimated run cost: USD 10 per day.
- Maximum cumulative monthly spend: USD 100.
- All three USD values (estimated run cost, daily spend to date, monthly spend to
  date) are required before the USD budget is marked authorized.
- Missing USD inputs leave authorization false.
- Fit calls are capped independently. Candidate count is bounded by both the
  candidate limit and integer fit capacity.
- Even an authorized plan is still plan-only: `training_authorized=false`.
  A separate reviewed training workflow must enforce the cost and fit budget
  before launching any training job.

Example output records the ranking, quality evidence, cost-model status,
allocated fit-call proxy, and reasons for any exclusion. No model training or
execution occurs.
