# ETI Phase 2 — Research Validation and Trial Evidence

## Scope

Phase 2 builds an auditable research-evaluation path. It is not a profitability certification, production deployment approval, or permission to trade live.

The implementation is split across stacked pull requests:

- **#212 — statistical validation:** Deflated Sharpe Ratio, CSCV/PBO, White Reality Check and Hansen SPA-style block-bootstrap diagnostics.
- **#214 — experiment ledger:** append-only trial lifecycle, immutable family declarations, aligned return streams and attempt accounting.
- **#215 — authenticated API:** workspace-scoped registration, lifecycle, family retrieval and diagnostics endpoints.
- **#216 — runner integration:** registers candidates before fitting, persists candidate-specific outer-test return differentials, records failed attempts and enforces contiguous non-overlapping test windows.
- **#217 — evidence trust boundary:** distinguishes client attestations from runner receipts, returns evidence status in diagnostics and cross-checks runner receipt hashes against frozen family declarations.

All pull requests remain unmerged. Their dependency order must be preserved.

## Phase 2 exit gates

| Gate | Current implementation | Remaining proof / limitation |
|---|---|---|
| Multiple-testing-aware statistics | DSR, PBO/CSCV, White Reality Check and SPA-style bootstrap | DSR sampling assumptions and bootstrap block choice still require expert review; no statistical test proves future profitability. |
| Complete in-platform attempt accounting | Registered, running, completed and failed lifecycle; attempted count includes running/completed/failed | ETI cannot discover experiments omitted or performed outside its ledger. |
| Frozen family identity | Config, training/evaluation hashes, cost hash and benchmark ID are immutable | API declarations are not independent source authentication. |
| Return-series integrity | UTC timestamps, strict order, equal spacing, common aligned timestamps, SHA-256 evidence | A caller can still submit fabricated return values through the authenticated API; API evidence is explicitly unverified. |
| Runner integration | Candidate trials registered before fitting and candidate-specific OOS returns persisted | Runner-generated snapshot fingerprint is a content receipt, not raw-data archival or provider authentication. |
| Cost assumptions | Evidence-backed, instrument-specific, time-bounded cost model validation | Referenced source URLs/licences and venue costs are not independently fetched or measured by this ledger. |
| Promotion and execution safety | No automatic promotion; human approval remains separate | Research diagnostics do not grant promotion, allocation or execution rights. Live execution remains disabled. |
| Build and test | CI pipeline includes Python tests, JS syntax, Compose validation and production image build | A passing run for the current PR #217 head is still required before Phase 2 can be called CI-verified. |

## Evidence trust levels

- `absent`: no source manifest was supplied.
- `unverified_client_attestation`: an authenticated API caller supplied a manifest. Authentication identifies the account, not the truth of the evidence.
- `captured_value_fingerprint_only`: the local runner captured and fingerprinted validated OHLCV values and records provider metadata. The fingerprint binds values but does not prove provider authenticity, completeness, licensing or correctness.
- `unverified_manifest`: a manifest exists but does not match a recognized capture path.

The ledger verifies the manifest digest and, for runner-captured receipts, cross-binds the snapshot fingerprint to provenance, training/evaluation hashes to the family and the canonical cost-model digest to the frozen family cost hash. These checks detect internal inconsistency and persisted tampering; they do not authenticate the external world.

## Statistical interpretation rules

1. Use net-of-configured-cost return differentials against the same declared benchmark for every trial in a family.
2. Register every candidate before fitting; include failed started attempts in the multiple-testing count.
3. Keep outer-test windows contiguous and non-overlapping for the current row-based block-bootstrap implementation.
4. Preserve timezone-aware, strictly increasing, equally spaced observation timestamps.
5. Select bootstrap block length before inspecting outcomes and report sensitivity to plausible alternatives before relying on a result.
6. Treat DSR as dependent on its sampling assumptions. Serial dependence, non-stationarity, omitted trials and unverified costs can invalidate inference.
7. Do not treat a low p-value, high DSR, or positive backtest as proof of a deployable edge.

## Phase 2 completion status

**Implementation complete and CI-verified on the integrated PR #217 merge ref; independent review and ordered PR integration remain pending.**

Latest integrated validation:
- GitHub Actions run [44573262669](https://github.com/hamza00alik-source/eti-research-core/actions/runs/44573262669) succeeded on 2026-11-19.
- **604 passed, 1 skipped, 8 warnings**.
- Python compilation, JavaScript syntax checks, Docker Compose validation and production container image build all passed.

Remaining governance / acceptance gates:

- Independent review confirms statistical implementation, candidate selection, trial-family accounting and evidence trust boundaries.
- The stacked PRs are integrated in dependency order after review and explicit repository-owner authorization.
- Phase 3 planning addresses raw immutable data artifact retention, provider/source verification, independently measured execution costs, trial-family completeness controls and stronger benchmark construction.

The passing integrated CI establishes code/build validation, not independent source authenticity, cost accuracy, complete capture of off-platform experiments, or a profitable trading edge. ETI must continue to report economic performance as unproven until genuinely out-of-sample, independently costed evidence supports it.
