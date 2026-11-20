# Phase 3 — Immutable Research Data Evidence

## First implementation slice: local content-addressed OHLCV archive

Ledger-backed runs through `run_approved_multi_market_research` now archive the exact validated OHLCV values used by the evaluator before model fitting. The archive is stored as canonical JSON using UTC timestamps and hexadecimal float representations, with a SHA-256-derived object path:

```text
<archive-root>/<first-two-hash-characters>/<sha256>.json
```

The default root is `data/research_evidence`. Operators may set `ETI_RESEARCH_EVIDENCE_DIR` or pass `evidence_archive_dir` explicitly. The research-family evidence manifest stores a receipt containing the digest, relative path, byte size, row count and time bounds. `eti.research_evidence.verify_ohlcv_archive` verifies the path binding, content digest, schema and row count.

Writes use a temporary file plus an atomic no-overwrite hard link. If an object already exists, its bytes must match the requested content. A digest collision or conflicting existing object fails closed. Symbolic-link objects are rejected.

## Trust boundary

This is **content-addressed, tamper-evident local storage**, not a claim of cryptographic immutability or external truth:

- It preserves the exact validated OHLCV values ETI evaluated, not the provider's original HTTP response bytes.
- A matching digest proves that the archived bytes match the receipt; it does not prove that the provider supplied accurate, complete or licensed data.
- Local filesystem administrators can still replace or delete files. This is not WORM/object-lock storage and is not a substitute for independently replicated backups.
- The receipt does not authenticate a provider identity or a venue's data signature.
- The archive is local to the runner's filesystem. Multi-host workers require shared, access-controlled object storage and a retention/backup policy before deployment.
- Runs without a configured trial ledger do not create this research-evidence archive.

## Verification

The automated tests cover deterministic content addressing, changed-data separation, digest tampering, path traversal, non-finite values, naive timestamps and runner-to-ledger receipt integration.

## Live public market and paper-feed slice

See [LIVE_MARKET_PAPER_FEED.md](LIVE_MARKET_PAPER_FEED.md) for the read-only CCXT spot snapshot collector and the paper engine's fresh bid/ask pricing. This is REST polling, not a continuous WebSocket service; simulated fills still do not model order-book depth, queue position, partial fills or market impact. Live execution remains disabled.

## Remaining Phase 3 work

1. Add provider-specific source identity and raw-response capture where legally and technically available; independently reconcile samples against a second source.
2. Add replicated object storage with retention/object-lock controls, access audit and restore drills.
3. Replace assumed transaction-cost inputs with independently measured, venue/instrument/time-bounded fee schedules and observed spread/impact evidence. Preserve the distinction between measured fills and estimated future costs.
4. Add cost-evidence receipts and fail-closed eligibility rules so an assumed cost model cannot be labelled independently verified.
5. Add benchmark construction receipts and controls to detect experiments run outside ETI's trial ledger.

No change in this slice enables order execution, model promotion or capital allocation. Live order execution remains disabled, and ETI's economic result remains NO-GO until genuinely untouched out-of-sample evidence supports an edge.
