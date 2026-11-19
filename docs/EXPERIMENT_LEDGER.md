# ETI experiment and trial-family ledger

## Purpose

Multiple-testing-aware validation is only meaningful when ETI records the hypotheses actually attempted. The SQLite research ledger stores an append-only trial identity within a frozen comparison family and records a single terminal outcome for each started attempt.

## Family contract

The first trial registered in a family freezes:
- training-data SHA-256;
- evaluation-data SHA-256;
- cost-model SHA-256;
- benchmark identifier.

Every later trial in that family must repeat the exact same values. A family cannot silently change its dataset, cost assumptions or benchmark midway through a comparison. Each trial also records its strategy name and configuration SHA-256.

## Trial lifecycle

1. **registered** — the hypothesis/configuration is recorded but has not started. It is not counted as an attempted trial.
2. **running** — execution has started. It is counted in the family trial count, even if it later fails.
3. **completed** — the trial stores a timestamped per-observation net return differential versus the family's common benchmark. Completion is write-once.
4. **failed** — the started attempt is retained with a failure reason and remains counted in the family trial count.

A trial ID cannot be reused. A completed or failed attempt cannot be restarted or overwritten. The storage layer does not provide a delete/update method for trial identity.

## Return-series contract

Completion requires at least eight paired observations, identical lengths for timestamps and net differentials, timezone-aware ISO-8601 timestamps, strictly increasing unique observation times, and finite numeric differentials. The series is normalized to UTC and hashed with its family/trial identity and benchmark metadata.

The family matrix accessor includes only completed trials and refuses to construct a matrix unless every completed trial has exactly the same timestamp vector. It returns an observations-by-trials matrix suitable for PBO, White Reality Check and SPA. It also returns the attempted-trial count (running + completed + failed), while separately reporting registered-but-not-started trials.

## Statistical usage

- Pass the family snapshot to `assess_research_trial_family(family, selected_trial_id=...)` to run all four statistical diagnostics together.
- The integrated assessment applies the DSR estimator to the selected trial's net benchmark-relative differential (an information-ratio basis) and uses the full attempted-trial count; do not use only the completed shortlist.
- If a failed attempt has no return series, it is counted for multiple-testing penalty but is not fabricated into the return matrix.
- Use the family benchmark ID, evaluation-data hash and cost-model hash as mandatory inputs to downstream report provenance.

## Limitations and next integration

Trial-family identifiers are scoped to an owner and workspace. Reusing a trial identifier within that scope is rejected, while independent workspaces may use the same local identifier.

This first ledger is a storage primitive. It is not yet connected to an authenticated API or the autonomous research runner. Callers must obtain owner_id from the authenticated principal and must register each attempted hypothesis before execution. The ledger validates declared hashes and alignment but cannot authenticate source data, independently verify costs, or prove the benchmark was correctly constructed. SQLite remains a single-instance store; multi-replica deployment requires a PostgreSQL migration.

The ledger does not approve models, create signals, allocate capital or place orders. Live order execution remains hard-disabled.
