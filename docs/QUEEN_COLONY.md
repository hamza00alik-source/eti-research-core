# ETI Queen and Dynamic Colony

## Purpose

The Queen is a bounded supervisor for ETI's existing durable task queue. It
scales concurrent worker slots in response to queued work. It does not invent
new agent capabilities or grant new financial, model-promotion, credential,
or order-execution permissions.

## Architecture

- **Queen Orchestrator**: schedules already-authorized autonomy, paper, and
  live-execution tasks; observes the owner/workspace queue; reconciles worker
  concurrency against demand.
- **Worker slots**: concurrent instances of the existing `AgentWorker`.
  Each worker claims one durable task at a time using the existing SQLite
  lease and heartbeat mechanism.
- **Specialist capabilities**: task type selects a server-implemented handler
  from `AgentCapabilityRegistry` (market discovery, snapshots, coverage,
  derivatives context, strategy evaluation, or research model training).
- **Hive memory**: existing `MemoryStore`, task results, model registry, and
  audit events retain task history and model lineage.
- **Evolution**: existing strategy/model evaluation remains responsible for
  candidate comparison. Queen does not promote a candidate or mutate model
  quality gates.

## Elastic scaling policy

For queued task count `Q`, configured maximum `M`, and backlog-per-worker
`B`:

`desired_workers = 0` when `Q = 0`; otherwise
`min(M, ceil(Q / B))`.

Defaults: maximum 2 concurrent slots, 2 queued tasks per slot, 1-second poll,
120-second task lease. The maximum is hard-bounded to 8 slots. Retiring workers
finish their current leased task; they do not abandon a task or write a stale
result. Multiple worker slots coordinate through the durable lease mechanism.

## Start

Run this supervisor in the same deployment and database environment as ETI's
worker API. Supply the owner and workspace from a trusted operator context:

```bash
python -m eti.queen --owner-id OWNER_ID --workspace-id WORKSPACE_ID --max-workers 2
```

Or configure `ETI_QUEEN_OWNER_ID` and `ETI_QUEEN_WORKSPACE_ID`. Never accept
these identifiers from an unauthenticated public request. SQLite is suitable
for a single-instance deployment; migrate the store to PostgreSQL before
running multiple replicas against shared production workloads.

## Safety boundaries

- Queen scales compute concurrency only; it cannot raise the ETI research
  spend limits ($10/day and $100/month) or bypass approved research artifacts.
- No dynamic Python, shell, plugin, or model-generated capability execution.
- Workers only execute registered server-side capabilities.
- Existing owner-authorized task creation and live-execution supervisor gates
  remain in force.
- No broker credentials are passed to the Queen or general research workers.
- No model is promoted by worker count, task success, or Queen preference.
- Live execution remains separately permissioned; this supervisor does not
  turn research into live trading.
- Logs contain lifecycle metadata, not task payloads or exception messages.

## What "self-evolving" means in this release

The Queen can add and retire concurrent worker slots as workload changes.
Research/model/strategy evolution continues through ETI's separately tested
candidate-generation and evaluation paths. The Queen does **not** rewrite its
own source, create arbitrary new capabilities, train itself by changing its
own weights, or silently replace the production supervisor. Such changes
require versioned code, tests, review, and deployment.

## Operational limitations

This first implementation uses threads and the existing SQLite-backed store.
It is intended for one ETI application instance. Heavy model training can
consume CPU and memory, so the default colony is deliberately small. Increase
the cap only after measuring actual memory, CPU, provider rate limits, and
Kaggle/cloud spend. Horizontal multi-instance scaling requires a shared
PostgreSQL queue and distributed worker health/lease monitoring.
