# ETI Point-in-Time Macroeconomic Data

## Purpose and boundary

Macroeconomic observations are not OHLCV candles. They have a reference period,
a publication/release time, and potentially multiple revised vintages. ETI keeps
this domain in a separate module and does not yet feed macro observations into
the market-feature or model-training pipeline.

## Observation contract

Each immutable \`MacroObservation\` carries:

- \`series_id\`: stable provider/series identifier;
- \`period\`: the economic reference period;
- \`value\` and \`unit\`;
- \`released_at\`: when the value became public;
- \`vintage_at\`: when this specific value/revision became available;
- \`source\`, \`provenance\`, and \`license_status\`.

All timestamps must be timezone-aware. Values must be finite. A vintage cannot
precede its release timestamp. Provider integrations must supply actual release
and vintage metadata; ETI does not infer them from download time or observation
period.

## Point-in-time selection

\`MacroDataHub.fetch_as_of(provider, series_id, decision_time)\` selects the latest
vintage for each reference period for which both release and vintage timestamps
are no later than the decision time. Later revisions remain invisible to earlier
decisions. Duplicate identities and ambiguous same-time vintages fail closed.
A provider is selected explicitly; different providers are never silently merged.

The default license policy accepts only \`open\` and \`licensed\`. Restricted or
unknown status is rejected. A status label is a governance input, not legal
advice or proof that a dataset's terms permit the intended use. Store source
terms, attribution, access limits, retention and redistribution rules alongside
provider integration evidence before production use.

## Current implementation status

- Implemented: typed observation/vintage contract, in-memory provider protocol,
  as-of selector, duplicate and timestamp validation, license gate, tests.
- Not implemented: live FRED/central-bank/statistical-agency API adapters,
  provider authentication, revision-history ingestion, persistent storage,
  macro feature engineering, model integration, or backtest integration.
- No provider reachability, completeness, licensing, or trading-edge claim is
  made by this module.

## Integration requirements

Before adding a real provider, verify its official API documentation, API terms,
series metadata, release calendar, revision/vintage endpoint, timezone semantics,
rate limits, historical depth and permitted use. Preserve the raw provider
payload or a content hash and retrieval metadata for audit. Do not backfill a
latest revised value across earlier decision dates. For real-time systems, model
the provider's actual publication timestamp and any feed delay separately.

No order execution is provided or enabled by this module.
