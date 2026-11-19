# ETI experiment and trial-family ledger

## Purpose

Multiple-testing-aware validation is only meaningful when ETI records the hypotheses actually attempted. The SQLite research ledger stores an append-only trial identity within a frozen comparison family and records a single terminal outcome for each started attempt. The approved research runner now registers and starts each nested candidate before fitting, then completes it with its candidate-specific outer-test return differential series.

## Family contract

The first trial registered in a family freezes:
- training-data SHA-256;
- evaluation-data SHA-256;
- cost-model SHA-256;
- benchmark identifier;
- an immutable source-evidence manifest and its SHA-256 digest (when created by the approved local runner).

The approved runner now fingerprints the exact validated OHLCV snapshot at the MarketDataHub boundary using a stable UTC-timestamp / IEEE-754 value serialization. It binds that snapshot digest, provider adapter identity, capture time, source provenance, data-quality report, validated cost-model object and its digest, plus the derived training/evaluation row hashes into one family manifest. The ledger stores the manifest and verifies its digest before returning diagnostics. Later trials must repeat the same manifest byte-for-byte.

The authenticated public registration API does not accept a source-evidence manifest; API-created families therefore retain an empty manifest unless registered through the local runner. This makes the evidence receipt useful for reproducibility without pretending that a caller-supplied API hash is independently verified. A content fingerprint proves which values were captured, not that the upstream provider was truthful, complete, licensed or manipulation-free. It also cannot prove that every experiment outside ETI was disclosed.

## Trial lifecycle

1. **registered** — the hypothesis/configuration is recorded but has not started. It is not counted as an attempted trial.
2. **running** — execution has started. It is counted in the family trial count, even if it later fails.
3. **completed** — the trial stores a timestamped per-observation net return differential versus the family's common benchmark. Completion is write-once.
4. **failed** — the started attempt is retained with a failure reason and remains counted in the family trial count.

A trial ID cannot be reused. A completed or failed attempt cannot be restarted or overwritten. The storage layer does not provide a delete/update method for trial identity.

## Return-series contract

Completion requires at least eight paired observations, identical lengths for timestamps and net differentials, timezone-aware ISO-8601 timestamps, strictly increasing unique observation times, and finite numeric differentials. The series is normalized to UTC and hashed with its family/trial identity and benchmark metadata.

The family matrix accessor includes only completed trials, recomputes each stored return-series digest before use, and refuses to construct a matrix unless every completed trial has exactly the same timestamp vector. A digest mismatch blocks the family assessment. It returns an observations-by-trials matrix suitable for PBO, White Reality Check and SPA. It also returns the attempted-trial count (running + completed + failed), while separately reporting registered-but-not-started trials.

## Statistical usage

- Pass the family snapshot to `assess_research_trial_family(family, selected_trial_id=...)` to run all four statistical diagnostics together.
- The integrated assessment applies the DSR estimator to the selected trial's net benchmark-relative differential (an information-ratio basis) and uses the full attempted-trial count; do not use only the completed shortlist. If the selected trial has zero volatility, DSR is reported as undefined while the other family diagnostics remain available.
- If a failed attempt has no return series, it is counted for multiple-testing penalty but is not fabricated into the return matrix.
- Use the family benchmark ID, evaluation-data hash and cost-model hash as mandatory inputs to downstream report provenance.
- CSCV/PBO requires an observation count divisible by the requested split count. If not, the integrated assessment marks PBO as not computed but still returns the block-bootstrap diagnostics.

## Limitations and next integration

Trial-family identifiers are scoped to an owner and workspace. Reusing a trial identifier within that scope is rejected, while independent workspaces may use the same local identifier.

The ledger is connected to authenticated, workspace-scoped API endpoints and the approved research runner. The runner records the two current nested candidates against a common no-trade benchmark. Ledger recording requires step_size == test_size, unique observations, and timestamps exactly one configured candle interval apart; this prevents row-based block bootstraps from treating gaps as adjacent observations. If registration succeeds but starting fails, the registered-only trial remains visible and is excluded from attempted counts; only successfully started trials are marked failed after a run error. The ledger verifies the persisted source-manifest digest and exact captured-value fingerprint for runner-created families, but cannot independently authenticate the upstream provider, independently measure venue costs, or prove the benchmark was correctly constructed. API clients may optionally submit a source-evidence manifest; it is caller-declared and digest-verified, not independently authenticated, and legacy/API-created families may still have an empty manifest. It also cannot prove that all experiments performed outside ETI were registered.

The aggregate training-row and evaluation-row hashes describe the union of rows used across expanding walk-forward folds. They are not globally disjoint: an earlier outer-test observation can legitimately enter the training set of a later fold. Each fold's own test window remains chronologically held out from that fold's training data. SQLite remains a single-instance store; multi-replica deployment requires a PostgreSQL migration.

The ledger does not approve models, create signals, allocate capital or place orders. Live order execution remains hard-disabled.
