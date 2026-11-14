# ETI Solana research adapter (paper-only)

## Purpose

Add a Solana/Pump.fun research lane to ETI without replacing ETI's shared intelligence core or multi-market roadmap. This lane discovers new/migrated tokens, gathers market and chain evidence, computes risk and opportunity diagnostics, and records paper-only hypothetical opportunities. It does not connect to FOMO, sign transactions, copy wallets, or place orders.

## Pipeline

```text
PumpPortal event stream
        |
        v
Event normalizer + dedupe + durable event log
        |
        v
Scanner: pool selection, age, liquidity, turnover, concentration hints
        |
        v
Auditor: mint/freeze authorities, supply, holder distribution,
         creator evidence, LP/lock evidence, insider graph, delegates
        |
        v
Analyst: causal features + approved ETI model + optional external comparator
        |
        v
ETI deterministic risk gate + portfolio limits
        |
        v
Paper position simulator + monitoring + immutable decision/audit events
        |
        v
ETI API / PWA / Android client / optional notifications
```

The five article roles map to ETI capabilities, not independent unbounded LLMs:

- Scanner: event ingestion and market-data enrichment.
- Auditor: evidence collection and risk classification.
- Analyst: feature/model inference and walk-forward-calibrated probability.
- Executor: renamed **Paper Sizer**; creates a hypothetical paper intent only.
- Risk: portfolio-aware paper position monitor and exit simulation.

## Provider contracts

### PumpPortal

One long-lived WebSocket connection per process. Subscribe to documented new-token and migration streams. Use reconnect with bounded exponential backoff, event signature/mint deduplication, UTC timestamps, and a raw event log. Do not assume delivery is exactly-once; reconcile with chain/indexer history where available. Keep API key server-side if a key is required.

### DexScreener

Use documented token/pair lookup endpoints. The multi-token endpoint accepts up to 30 token addresses per request. Select the correct pool using chain, base/quote mint, pair age, liquidity and source identity. Store retrieval timestamp and pair address. Apply endpoint-specific rate limits and bounded backoff on 429/5xx. A snapshot is not a candle history; derive candles only from stored timestamped observations and label them as sampled marks, not exchange OHLCV.

### Solana RPC

Use JSON-RPC POST requests. Use getAccountInfo/getTokenSupply/getTokenLargestAccounts/getTokenAccountsByOwner/getProgramAccounts as required by a layer. Test both SPL Token and Token-2022. Use configured RPC provider(s); the public shared endpoint is for development and can rate-limit or block traffic. Bound getProgramAccounts by data filters and response size; never assume all RPC vendors support identical enhanced options.

### RugCheck

Optional enrichment. Store raw report and insider graph with retrieval time and provider metadata. Validate the actual response schema at runtime. Missing data is unknown. Graph connections are evidence for review, not proof of shared ownership.

## Canonical evidence record

Every observation includes:
- chain = solana
- mint and pool address
- provider and endpoint identifier
- fetched_at and source timestamp/block/slot when available
- schema version and raw-response SHA-256
- normalized field value, unit and coverage status
- stale/partial/error state and retry count

Do not merge data from different pools or quote assets without explicit conversion and provenance.

## Risk semantics

Each audit layer returns one of:
- PASS: evidence present and the specific test passed.
- FAIL: evidence present and the specific test failed.
- UNKNOWN: missing, stale, unsupported or inconsistent evidence.

UNKNOWN must never be converted to PASS. A token may be blocked from paper entry on UNKNOWN critical evidence, but the UI must say "unknown/provider unavailable", not "scam". A total score must not hide any critical FAIL/UNKNOWN layer.

Risk features should include holder concentration aggregated by beneficial owner where resolvable, pool-specific liquidity/LP ownership and unlock schedule, mint/freeze authorities, creator/insider holdings with historical coverage, Token-2022 extensions, delegate authority/amount, transfer restrictions/taxes where measurable, pool age, volume quality and data freshness.

## Analyst and labels

The Analyst must not ask a model to assert future facts as if they were observed. For example, "will still be actively traded six hours from now" is a forecast target, not an audit fact. Persist snapshots at decision time and create labels only after the horizon passes. Define target precisely (e.g. executable exit liquidity, survival, max adverse/favourable excursion, net return after realistic costs). Train and evaluate with chronological purging and nested walk-forward, and compare with no-trade, simple momentum and liquidity baselines.

External Jev can be an optional benchmark if a server-side key is configured. Pin the exact model version, record response and usage, enforce a timeout, and never use it to bypass ETI risk controls.

## Paper simulation

The Paper Sizer calculates a target notional only after risk/portfolio checks. Risk budget is based on stop distance and conservative liquidity/impact estimates, not a fixed percentage of stack alone. The simulator must include spread, pool curve price impact, platform/protocol fees, priority fees, latency, missed events, failed transactions, partial fills and stop gaps. If a cost component cannot be estimated, report it as unknown and do not promote the result as profitability evidence.

## Explicit non-goals

- No FOMO browser session or private API.
- No Privy token, password, wallet seed, private key or transaction signing.
- No live buy/sell endpoint.
- No copy-trading execution.
- No "safe token" certification.
- No performance or six-figure claim without reproducible independently reviewed records.


## Durable discovery ingestion (implemented in ETI branch)

The branch now includes:
- `eti/solana_ingest.py`: a separate process that subscribes to PumpPortal new-token and migration feeds, normalizes only recognized event types, and writes events to ETI storage.
- `solana_research_events`: owner/workspace-scoped SQLite ledger with a unique event key, raw JSON payload, SHA-256 payload digest, source timestamp and received timestamp.
- Authenticated endpoints:
  - `POST /v1/workspaces/{workspace_id}/solana/events`
  - `GET /v1/workspaces/{workspace_id}/solana/events`
- Duplicate delivery returns the original event; reuse of an event key with a different payload is surfaced as `payload_conflict`, not silently overwritten.

The new-token and migration subscriptions are free according to PumpPortal's current Data API documentation. Token/account trade streams require an API key and are metered; the initial ETI ingestion process does not subscribe to those streams. Run the ingestor separately from the API and worker process so WebSocket outages do not take down authenticated application requests.

The public PumpPortal docs describe the subscription methods but do not provide a complete stable JSON schema for every migration payload. ETI therefore only accepts the normalized `txType` values explicitly recognized by its adapter; unrecognized messages are dropped rather than guessed. Before relying on migration capture, validate actual live sample payloads and add schema fixtures.


## Automated research task chain

The first accepted PumpPortal event now creates its `solana.market.scan` task in the
same SQLite transaction as the durable event row. Replayed event keys return the
original event and do not enqueue another scan. The event record includes the
created scan task ID for observability. This avoids a crash window between event
persistence and task scheduling.

The worker executes the allowlisted DexScreener market scan. An explicit
`REJECT` verdict ends the chain. `PASS`, `WAIT`, or `UNKNOWN` outcomes enqueue
a read-only `solana.token.audit` follow-up, carrying the source scan and event
IDs. The audit remains conservative: missing critical evidence blocks a pass.
No wallet, signing, swap, copy-trading, or live Solana execution capability is
registered. The scan completion and audit-task insertion commit in one SQLite transaction,
so a completed scan cannot be left without its eligible audit task due to a
worker process crash between those operations.


## Point-in-time research snapshots (implemented)

Every completed `solana.market.scan` and `solana.token.audit` task now
persists an immutable snapshot in `solana_research_snapshots` within the same
SQLite transaction as task completion. Each record includes owner/workspace,
source task and capability, mint, pool address when known, schema version,
normalized result payload, SHA-256 digest and fetched timestamp. The task ID is
unique per owner/workspace, so a task cannot create multiple snapshots.

Authenticated retrieval is available at
`GET /v1/workspaces/{workspace_id}/solana/snapshots`, with optional `mint`
filter and bounded limit. These snapshots are point-in-time observations, not
OHLCV candles or proof of executable liquidity. Future labels must be generated
only after their prediction horizon has elapsed, and sampled price marks must
remain clearly distinguished from exchange or pool candle data.


## Recurring observations and delayed outcome labels (implemented)

A successful scan that identifies a pool creates an owner/workspace-scoped refresh
schedule. The worker checks due schedules and enqueues repeat
`solana.market.scan` tasks on a configurable 5-minute to 24-hour cadence
(default 5 minutes). Scheduling and enqueueing are transactional and repeated
worker polls do not duplicate a due scan. Each completed scan remains an
immutable point-in-time snapshot. Refreshes are not backfills: ETI does not
fabricate observations for periods when the worker or provider was unavailable.

The scanner now records the DexScreener sampled `priceUsd`, USD liquidity,
reported rolling 24-hour volume, 1-hour buy/sell counts, DEX identifier and
quote-token address when supplied. Missing fields remain null. These are
provider-reported sampled marks; they are not a continuous trade tape, OHLCV
candles, order-book depth, or executable quotes. Pool identity is preserved so
observations from different pools are never compared as if they were one market.

The delayed label module supports 5-minute, 15-minute, 1-hour, 6-hour and
24-hour horizons. A label is created only when (a) its horizon has elapsed,
(b) a later observation exists at or beyond that horizon, and (c) the baseline
and target have the same mint and non-empty pool address. The first qualifying
future observation is used. A unique baseline-snapshot/horizon key makes label
persistence idempotent. Missing future observations do not produce a label. The
label generator skips malformed stored JSON and snapshot schema versions it does
not explicitly support, and reports the skipped count instead of crashing or
silently interpreting unknown payload formats.
Authenticated endpoints:
- `POST /v1/workspaces/{workspace_id}/solana/labels/generate`
- `GET /v1/workspaces/{workspace_id}/solana/labels`

Labels expose sampled price return, liquidity change and change in the
provider's rolling 24-hour volume field. That last metric is explicitly not
volume traded during the selected horizon. No future values are read from the
baseline feature record. Evaluation must split by time and group by mint across all its pools;
overlapping horizons from the same launch must not be randomly scattered across
train and test sets. Use chronological walk-forward splits with purging/embargo
at least as long as the maximum label horizon, and fit all transforms only on
the training fold. Keep a launch/pool group entirely within one fold when
reporting generalization to unseen launches.

## Execution-cost and failed-exit evidence boundary

The current public DexScreener adapter does not provide validated executable
quotes or transaction-level fills. Therefore label fields for exitability,
realized slippage and failed exit are deliberately `UNKNOWN` / null. ETI must
not substitute a fixed slippage guess or infer a successful exit from displayed
liquidity. A future cost model must be pool-specific and validated against
actual swap simulations or observed transactions, including curve impact,
fees, priority fees, latency, liquidity depletion and failed/partial exits.
Until then, the research pipeline reports price/liquidity deterioration as
observed risk proxies, not executable P&L.

Model training remains paused for this Solana label lane until sufficient
time-stamped observations and independently reviewed labels accumulate.
Nothing in this pipeline adds wallet access, transaction signing, swaps,
copy-trading, or live execution.


A reusable `chronological_group_split` helper is now included for downstream
evaluation. It keeps every pool belonging to the same mint wholly within one fold, requires
training label outcomes to end before the validation boundary, applies an
explicit embargo (default 24 hours), and purges groups crossing either
boundary. It fails closed if there are too few distinct pools or if purging
leaves an empty fold. This is a validation primitive, not evidence that a model
is ready to train; minimum sample size, class balance, launch-regime coverage,
and independent test periods still need to be assessed before training.


A new-token event that arrives before DexScreener has indexed a pair is not
dropped from longitudinal collection. ETI creates a pending-pool refresh record
(`pool_address=__pending__`) and rechecks the mint every 5 minutes. When a
pair appears, the pending schedule is disabled and replaced by the discovered
pool schedule. If a scan returns a definitive REJECT, active refresh schedules
for that mint are disabled. This prevents a missing initial pair from silently
ending observation collection.

Refresh schedules can be inspected or configured through authenticated routes:
- `GET /v1/workspaces/{workspace_id}/solana/refresh`
- `POST /v1/workspaces/{workspace_id}/solana/refresh` with `mint`,
  `pool_address`, and `interval_seconds` (300–86400; default 300).


Timestamp semantics: the scanner's `observed_at_ms` is ETI's scan-time timestamp
for when it sampled the public provider response. The current DexScreener
adapter does not expose a reliable quote-observation timestamp, so this must
not be represented as an exchange trade timestamp. If missing or invalid, the
snapshot uses task completion time.


For model evaluation, `chronological_walk_forward_splits` now provides expanding
training windows and sequential validation windows. All pools belonging to a mint remain in
one group, labels crossing a fold boundary are purged, and validation begins
only after the configured embargo. The helper refuses insufficient data or
folds emptied by purging. The single chronological split helper remains
available for simpler diagnostics. Preprocessing, feature selection,
calibration and threshold selection must be fit within each training fold;
the helper deliberately does not fit any of those transformations.


## Dataset quality and training gate (implemented)

The authenticated endpoint
`GET /v1/workspaces/{workspace_id}/solana/dataset-report` returns
workspace-scoped collection diagnostics: total scan/audit snapshots, distinct
mints and mint/pool pairs, observation span, median sampled interval, long
sampling gaps, malformed payload count, per-feature coverage, and label counts
by horizon including distinct mints and available price returns.

The report deliberately keeps `training_gate.ready=false` and reports
`RESEARCH_DATA_COLLECTION`. No sample-size, coverage, regime-diversity,
execution-cost, calibration, or out-of-sample acceptance thresholds have yet
been independently approved. A high snapshot count alone is not a reason to
train or deploy a model. The endpoint is descriptive, owner/workspace scoped,
and does not expose another account's observations.

## Research dataset export and feature diagnostics (implemented)

The authenticated endpoint
`GET /v1/workspaces/{workspace_id}/solana/dataset-export` provides a
deterministically ordered, bounded JSON export of point-in-time snapshots and
labels whose baseline snapshots are included. It supports an optional mint
filter and a maximum of 5,000 snapshots per request. Each snapshot preserves
its task ID, capability, mint/pool identity, schema version, sampled timestamp,
normalized payload and SHA-256 digest. Labels retain their baseline and target
snapshot references and horizon, plus explicit
`baseline_snapshot_in_export` and `target_snapshot_in_export` flags. A bounded
export can include a baseline label while omitting its later target snapshot;
consumers must check the target flag before attempting to join the target row.
Only baseline snapshot fields may be used as prediction inputs.

The export includes non-mutating validation diagnostics for malformed payloads,
unsupported snapshot schema versions, missing evidence/features, invalid numeric
values, negative values, zero price or liquidity, invalid timestamps, and duplicate
entity/timestamp observations. It also checks each label's baseline and target
snapshot references for existence, mint/pool identity consistency, chronological
ordering, and minimum horizon elapsed; `label_reference_issue_count` and each
label's `reference_issues` expose failed checks. A target that exists in the same
workspace but falls outside a bounded export is still checked for integrity, while
`target_snapshot_in_export` separately indicates whether it is present in the
returned snapshot list. Diagnostics flag data for review; they do not coerce,
repair, impute, or discard source values. The export is account/workspace scoped and requires normal ETI
authentication. It is a research artifact, not a training approval or execution
authorization. When using labels, only baseline snapshot fields may be used as
model inputs; target snapshot fields are outcomes and must never leak into
features.

The research-only `prepare_solana_training_rows` helper builds a conservative
baseline-feature matrix only when both referenced snapshots are present in the
export, pass snapshot validation, match the label's mint/pool identity, and the
label has no reference-integrity issues and a finite sampled-return outcome.
Malformed labels, invalid baseline features, unsupported snapshots and targets
omitted by bounded exports are excluded with reason counts. The helper copies
features only from the baseline snapshot; target observations contribute only
the already-computed outcome. Its status is always `PREPARATION_ONLY` and
`training_authorized` is always false. It does not fit or approve a model, and
sampled returns are not executable returns or net P&L.


## Fixed benchmark diagnostics (implemented; no fitting)

`eti/solana_evaluation.py` provides a non-fitting comparator for chronological
walk-forward folds. It evaluates validation rows only and reports a no-trade
reference, an always-long sampled-return reference, and a fixed buy-pressure
rule (1-hour buys divided by buys plus sells, using a declared threshold).
The threshold is supplied before evaluation and is not selected or optimized
against validation outcomes. Malformed rows are excluded and counted.

Reports include trade counts, mean/median sampled return, win rate, compounded
sampled return and a simple sequential drawdown diagnostic. These are gross
sampled-mark comparisons only: fees, spread, impact, priority fees, latency,
failed exits and capital constraints are not modeled. Fold aggregates are
descriptive and are not an independent final holdout. The helper performs no
model fitting, threshold search, calibration, training approval or execution;
`training_authorized` remains false.


The benchmark helper expects each fold's `validation` list to contain prepared
rows, not raw persisted label records. Build the walk-forward folds from the
chronological label metadata, then join each validation label to its prepared
row by `label_id`; never move rows between folds or randomly repartition them.
Prepared rows retain `baseline_at` for deterministic chronological ordering.


The evaluation module now also exposes
`evaluate_prepared_solana_walk_forward`. It derives expanding, mint-grouped,
purged/embargoed folds from the exported label timestamps and joins prepared
rows by their stable label IDs. Only joined validation rows are passed to the
fixed benchmark; training rows are counted for diagnostics and never contribute
to benchmark metrics. Missing prepared validation rows are counted rather than
replaced or reassigned. The runner accepts only the fail-closed
`PREPARATION_ONLY` artifact and preserves `training_authorized=false`.


The walk-forward runner additionally reports benchmark results separately by
label horizon (using each prepared row's explicit horizon) so short- and
long-horizon outcomes are not hidden in a pooled aggregate. It reports
validation dependence diagnostics: distinct mints, repeated mint labels, and
overlapping baseline-to-target intervals within each mint. Repeated and
overlapping labels are dependent observations, not independent trades or
independent samples. These diagnostics do not de-duplicate or reweight the
underlying observations and do not establish statistical significance.


## Fail-closed execution-cost evidence diagnostics (implemented)

`eti/solana_costs.py` validates five required measured cost contributions:
spread, pool price impact, protocol fee, priority fee, and latency slippage.
Each component must be a non-negative finite value expressed as
`bps_of_entry_notional`, explicitly measured (not estimated), with a source
identifier and timezone-aware observation timestamp. Values are defined as
round-trip contributions relative to entry notional; missing components are
not silently treated as zero.

The diagnostic also requires sourced, timestamped evidence of an observed
successful exit. Missing evidence, a failed exit, malformed timestamps,
unsourced values, estimates, wrong units, negative values, and non-finite
values keep the cost evidence gate closed. When all fields pass, the module
may report a cost-adjusted sampled-mark scenario for research comparison.
It always leaves `net_return_pct=null` and
`realized_pnl_available=false`: sampled provider marks are not executable
fills, and a separately observed exit does not prove the baseline/target
marks were executable. This module does not simulate or execute transactions,
approve training, or authorize trading.


The prepared walk-forward evaluator can now accept a mapping of execution-cost
evidence keyed by stable exported label ID. It applies the evidence validator
only after a label has joined to a prepared **validation** row; training labels
are never cost-evaluated or included in the report. Missing evidence is
reported as incomplete rather than treated as zero cost. The report exposes
per-validation-label missing/invalid components and, where every required
component and successful-exit record is present, a mean/median
cost-adjusted sampled-mark scenario. It always keeps `net_return_pct=null`
and `realized_pnl_available=false`. This is diagnostic integration only;
the evidence is supplied to the evaluator and is not yet a durable evidence
ledger or independently verified provider integration.


### Cost-evidence identity binding

Walk-forward cost diagnostics now require each supplied evidence envelope to repeat the exact `label_id`, `mint`, and `pool_address` for the validation label being evaluated. Missing or mismatched envelope identity forces that row to remain incomplete; cost values are not used in the scenario aggregate. Prepared-row mint and pool identity must also match the exported label or evaluation fails closed. This prevents accidental cross-label or cross-pool evidence reuse inside the in-memory evaluator.

This is an identity-consistency check, not durable provenance verification. Evidence is still caller-supplied and is not yet authenticated against a provider, immutable ledger, transaction receipt, owner, or workspace. Durable owner/workspace-scoped evidence storage and source-integrity verification remain separate work.


### Durable execution-cost evidence ledger

The research store now has an append-only `solana_execution_cost_evidence` ledger. Each record is scoped to authenticated owner/workspace, references an existing label, and must match that label's mint and pool. The canonical JSON envelope is SHA-256 hashed. Re-submitting identical evidence is idempotent; changed evidence creates a new immutable record rather than overwriting prior provenance. Reads recompute the digest and expose `integrity_valid=false` if stored JSON no longer matches its recorded hash.

Authenticated API:
- `POST /v1/workspaces/{workspace_id}/solana/cost-evidence` with `label_id`, `mint`, `pool_address`, and an evidence envelope containing the same identity fields.
- `GET /v1/workspaces/{workspace_id}/solana/cost-evidence` with optional `label_id` and bounded `limit`.

The API derives owner identity from the verified bearer token. Workspace and label lookups are owner-scoped and foreign records are not disclosed. The ledger preserves caller-supplied evidence; it does not independently prove provider authenticity, on-chain execution, transaction finality, or the economic correctness of cost measurements. A valid hash proves stored-byte consistency, not source truth. The SQLite implementation remains single-instance oriented; production multi-replica writes require a PostgreSQL migration and operational controls.
### Ledger-backed walk-forward cost diagnostics

`eti/solana_cost_evaluation.py` bridges the append-only cost-evidence ledger to the prepared walk-forward evaluator. The authenticated endpoint

`POST /v1/workspaces/{workspace_id}/solana/walk-forward-evaluation`

builds the bounded dataset export for the authenticated owner/workspace, prepares baseline-only research rows, loads evidence only through the same owner/workspace-scoped store, and evaluates joined validation labels. Request parameters control fold count, minimum training mint groups, embargo duration, and the predeclared fixed buy-pressure threshold; they do not enable model fitting or training.

The service performs a batched, owner/workspace-scoped lookup for the requested label IDs rather than issuing one database query per label. For each label, it selects the newest ledger record ordered by `created_at DESC,id DESC`. If that newest record has an invalid stored-byte digest or malformed envelope, it is rejected and the service does not fall back to an older record. Missing records remain incomplete. Only prepared validation rows appear in cost diagnostics; evidence for training labels is not included in scenario calculations. The report exposes the selected ledger record ID, digest, integrity status, and counts for missing or invalid records.

The endpoint remains diagnostics-only: `training_authorized=false`, `net_return_pct=null`, and `realized_pnl_available=false`. The ledger digest establishes stored-byte consistency, not provider authenticity, transaction finality, measurement accuracy, or evidence freshness. No maximum evidence age has been invented; a freshness policy must be independently approved before stale-evidence rejection can be enabled. The endpoint does not sign, simulate, submit, or execute Solana transactions.
### Public RPC transaction receipt verification

`eti/solana_tx_verification.py` verifies a supplied transaction signature through the configured public Solana JSON-RPC endpoint. The authenticated endpoint

`POST /v1/workspaces/{workspace_id}/solana/transaction-verification`

accepts a base58 transaction signature, mint, expected pool address, and wallet address. It checks `getSignatureStatuses` with transaction-history search, requires `confirmationStatus=finalized` and no execution error, then fetches `getTransaction` at finalized commitment. The returned transaction must contain the requested signature, agree on slot with the status response, have no metadata execution error, reference the requested mint in token balances, reference both the expected pool and wallet in transaction account keys, and show a net decrease in the wallet's owner-attributed raw token balance for that mint. Unknown, pending, failed, unavailable, mismatched, mint-unrelated, or non-debiting transactions are never reported verified.

This is stronger receipt and wallet-balance evidence, but it is not complete swap decoding. Pool account-key presence alone does not prove a specific pool instruction caused the debit; transaction account-key references can occur for other reasons. The result does not establish fill price, complete transaction costs, or realized P&L. The configured RPC provider is not independently cross-validated. The endpoint returns a verification report; it does not mutate immutable evidence records or automatically promote user-submitted cost evidence to trusted evidence. A separate server-attested append-only record is still required before this verification can be consumed by cost evaluation.


### Immutable transaction receipt attestations

`POST /v1/workspaces/{workspace_id}/solana/transaction-attestations` accepts a stored research label ID plus signature, mint, expected pool, and wallet. The server checks the label's persisted mint/pool identity, queries the configured public Solana RPC itself, and persists only a positive `FINALIZED_MINT_BALANCE_DEBIT_OBSERVED` report. Negative or incomplete verification reports are returned without creating an attestation. `GET /v1/workspaces/{workspace_id}/solana/transaction-attestations` lists the owner's records, optionally filtered by label.

Attestations are stored separately from user-submitted execution-cost evidence in an append-only owner/workspace/label-scoped table. Canonical report bytes are SHA-256 hashed; identical reports are idempotent, changed reports append new records, and reads recalculate the digest. The report records `transaction_receipt_verified=true` while retaining `verified=false`. This is a server-side record of what the configured RPC returned, not independent proof of source truth, decoded swap causality, a fill, an exit, measured transaction costs, or realized P&L. It is not automatically injected into cost-evidence evaluation and cannot authorize model training or live execution.

### Receipt-attestation evaluation diagnostics

The authenticated walk-forward evaluation service selects the newest stored receipt attestation for each exported label and reports receipt status only for prepared validation rows. It checks the stored-byte digest and the record/report label, mint, pool, wallet, and signature identities. If the newest record is invalid, it is reported as invalid rather than silently falling back to an older record. Receipt diagnostics are emitted in a separate `transaction_receipt_attestations` section and never modify execution-cost rows, satisfy `exit_evidence`, or change net-return/realized-P&L fields. A positive receipt observation still does not prove a swap instruction, fill, or exit and does not change the fail-closed training gate.

### Transaction receipt verification semantics

The public-RPC verifier reports `FINALIZED_MINT_BALANCE_DEBIT_OBSERVED` and `transaction_receipt_verified=true` only for finalized successful transactions with the requested mint, expected pool and wallet referenced, and an owner-attributed net wallet token-balance decrease. The generic `verified` flag remains false: this evidence does not decode and prove the swap instruction, pool causality, fill price, or realized P&L. It must not be promoted to a verified exit or used to authorize training or execution.
