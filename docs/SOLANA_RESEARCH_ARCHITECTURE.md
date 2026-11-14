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
(default 15 minutes). Scheduling and enqueueing are transactional and repeated
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
persistence idempotent. Missing future observations do not produce a label.
Authenticated endpoints:
- `POST /v1/workspaces/{workspace_id}/solana/labels/generate`
- `GET /v1/workspaces/{workspace_id}/solana/labels`

Labels expose sampled price return, liquidity change and change in the
provider's rolling 24-hour volume field. That last metric is explicitly not
volume traded during the selected horizon. No future values are read from the
baseline feature record. Evaluation must split by time and group by mint/pool;
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
evaluation. It keeps each mint/pool group wholly within one fold, requires
training label outcomes to end before the validation boundary, applies an
explicit embargo (default 24 hours), and purges groups crossing either
boundary. It fails closed if there are too few distinct pools or if purging
leaves an empty fold. This is a validation primitive, not evidence that a model
is ready to train; minimum sample size, class balance, launch-regime coverage,
and independent test periods still need to be assessed before training.
