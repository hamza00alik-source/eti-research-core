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
registered. Follow-up enqueueing checks for an existing task tied to the source
scan to reduce replay duplicates; task completion and follow-up creation are
not yet one atomic transaction, so a process crash in that narrow interval
requires operational reconciliation.
