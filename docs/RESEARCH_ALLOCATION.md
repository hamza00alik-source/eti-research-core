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

The cost-model JSON is keyed by exact `venue:market_type:symbol` identity.
Each value must be a complete `eti.transaction_cost_model.v1` object, not a
status string. It must explicitly state fee, spread, slippage, market impact,
funding and borrow costs, contract multiplier, quote/settlement currencies,
retrieval time, validity window, and at least one evidence record with an HTTPS
source URI, observation time, and declared `open` or `licensed` status.
Unknown/restricted license declarations, missing values, expired/future-dated
records, and identity mismatches fail closed.

Example structure (illustrative values only; not a validated fee schedule):

```json
{
  "kraken:spot:BTC/USD": {
    "schema_version": "eti.transaction_cost_model.v1",
    "instrument_identity": "kraken:spot:BTC/USD",
    "quote_currency": "USD",
    "settlement_currency": "USD",
    "fee_bps": 6,
    "spread_bps": 2,
    "slippage_bps": 1,
    "market_impact_bps": 1,
    "funding_bps_per_bar": 0,
    "borrow_bps_per_bar": 0,
    "contract_multiplier": 1,
    "retrieved_at": "2026-01-01T00:00:00Z",
    "valid_from": "2026-01-01T00:00:00Z",
    "valid_until": "2030-01-01T00:00:00Z",
    "evidence": [{
      "source": "Published venue fee schedule",
      "uri": "https://example.com/fees",
      "license_status": "open",
      "observed_at": "2026-01-01T00:00:00Z"
    }]
  }
}
```

The validator checks schema, exact identity, explicit numeric assumptions,
currency fields, timestamps, validity window, HTTPS evidence references, and
declared license status. The runner also requires approval assumptions to match
the validated model. Half-spread, slippage, and market impact are included in
the evaluator's slippage input. Nonzero borrow cost is rejected until the
evaluator supports it.

Important: ETI does not independently authenticate a URL, prove that a cited
document contains the supplied values, determine whether a license actually
covers this use, or measure live spread/slippage. Evidence must be reviewed by
the operator; the schema is a fail-closed evidence record, not proof of truth.

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
