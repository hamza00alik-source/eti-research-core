# Approved multi-market research runner

The research allocator remains plan-only. It ranks data readiness and marks
candidates eligible for economic evaluation only when a venue:symbol cost-model
status is explicitly validated. It never authorizes training.

The separate runner in `eti.research_runner` requires a reviewed local approval
artifact. Approval is explicit, scoped to exact `venue:market_type:symbol` keys, and must
include:
- `approved=true` and `authorize_research_fits=true`,
- `execution_enabled=false`,
- per-instrument fee, slippage, funding-per-bar, and short-capability assumptions,
- estimated run cost and daily/monthly spend-to-date,
- an overall model-fit-call cap no greater than 32.

The runner validates the scan and allocation schemas, rejects candidates that
are not marked `economic_evaluation_eligible`, fetches fresh validated history
through the market-data hub, and computes the total fit-call requirement for
all approved instruments before starting any model fit. If the total exceeds
the cap, no model fitting begins. Each instrument is evaluated independently
using ETI's existing nested purged walk-forward implementation; price series
are never pooled.

Example approval shape (fill with reviewed values; this is not a recommendation
to authorize a run):

```json
{
  "schema_version": "eti.research_approval.v1",
  "approved": true,
  "authorize_research_fits": true,
  "execution_enabled": false,
  "max_model_fit_calls": 32,
  "usd_budget": {
    "estimated_run_cost_usd": 2.0,
    "daily_spend_to_date_usd": 0.0,
    "monthly_spend_to_date_usd": 10.0
  },
  "evaluation": {
    "horizon": 6,
    "min_train": 400,
    "validation_size": 100,
    "test_size": 100,
    "step_size": 200
  },
  "instruments": {
    "kraken:spot:BTC/USD": {
      "fee_bps": 6.0,
      "slippage_bps": 2.0,
      "funding_bps_per_bar": 0.0,
      "allow_short": false
    }
  }
}
```

Instrument identity includes venue, market type, and symbol to prevent spot/derivative collisions. Duplicate identities and scan/allocation metadata mismatches are rejected. Contract multipliers are validated rather than silently replacing falsey values.\n\nThis artifact is an operator governance input, not a cryptographic signature
or proof of independent review. USD figures are operator estimates, not
measured Kaggle billing. Fit-call limits are compute proxies, not dollar costs.
The runner does not submit orders and cannot enable execution.

Yahoo non-crypto reference data, session-based markets without exchange
calendars, corporate actions, futures rolls, and macro release vintages remain
blocked or require dedicated validation. The current runner does not turn
reference OHLCV into execution-grade data.
