# Snapshot-backed paper allocation

ETI exposes a worker capability named `portfolio.allocate_from_account`. It is
an owner-scoped adapter around the deterministic `portfolio.allocate`
planner. The worker injects owner and workspace identity from the claimed,
authenticated task record; task input cannot select another owner's portfolio.

Required task input:

- `portfolio_account_id`: an active paper portfolio account in the task's owner
  and workspace.
- `max_snapshot_age_seconds`: an explicit freshness limit from 60 to 86,400
  seconds. The adapter rejects missing, stale, future-dated or malformed
  snapshots.
- `candidates`: strategy research candidates with the evidence required by
  `plan_paper_allocations`.

Optional allocator limits are passed through. Equity and current exposure maps
are never accepted from task input: the adapter derives them from the latest
persisted snapshot. It also raises the cash-buffer floor to preserve the
snapshot's reported cash amount. If that floor exceeds the allocator's
supported maximum, it blocks the request.

The result records the exact account and snapshot IDs, timestamp, age, base
currency and `exchange_verified: false`. Snapshot source remains
`owner_attested`; this is a paper simulation input, not broker truth. Currency
conversion, broker reconciliation, covariance-based correlation, durable
proposal idempotency and paper-fill accounting are not implemented by this
adapter. Owner review remains required, and no order is submitted.

This capability does not grant live execution authority. It always returns
`paper_only: true`, `execution_enabled: false` and
`orders_submitted: false`.
