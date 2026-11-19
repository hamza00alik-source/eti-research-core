# Phase 1 correctness and governance controls

## Canonical identity

All newly trained models use `eti.model_identity.model_id_for_instrument`. The identity contains venue, asset class, market type, normalized symbol and timeframe. Live crypto-spot execution and live authorization use the same shared `live_spot_model_ids` resolver. It prefers the canonical ID and accepts the exact pre-canonical spot ID only as a migration fallback. No derivatives execution is enabled by this compatibility path.

## Promotion authority

Autonomous training registers immutable candidate versions only. A passed training-time evaluation or calibrated artifact does not authorize promotion. Human review remains required. Existing approvals are not automatically demoted.

## Causality tests

`tests/test_features_causality.py` verifies that future-candle perturbations do not alter earlier feature values and that feature values calculated on a historical prefix match those from the full history at the same timestamps.

`tests/test_backtest_timing.py` verifies that a signal formed on a bar is exposed only to the following bar's return and that changing future signals cannot rewrite earlier equity.

These tests are necessary safeguards, not a mathematical proof against every possible leakage path. Any new feature, label, resampling rule or strategy pipeline must add its own timestamp-aware tests. Higher-timeframe joins and event-time fundamentals require dedicated tests before those data are eligible for model training.

## Promotion evidence still required

The approval API now fails closed unless a candidate carries a non-empty experiment ID and four valid SHA-256 lineage digests: code, configuration, training dataset and cost model. The challenger evidence must repeat the same experiment ID and match all four candidate digests. Storage rejects malformed cost-model digests as well as malformed code/config/dataset digests. This prevents manually registered candidates with missing lineage from bypassing the training-path provenance contract; it does not authenticate the truth of the supplied hashes.

Before a candidate can be promoted, the reviewer must inspect locked chronological out-of-sample evidence and fold-level results; measured or independently sourced fees, spreads and slippage with assumptions clearly labeled; cost stress scenarios; drawdown, turnover, liquidity and concentration behavior; leakage, provenance and historical-availability checks; the number of hypotheses and parameter/model trials considered; and calibration and artifact-schema integrity.

## Temporal-integrity acceptance tests

`tests/test_phase1_temporal_integrity.py` checks completed-candle filtering at exact timeframe boundaries, sorting and duplicate removal before the completion filter, and deterministic replay of identical backtest inputs. These tests complement the feature prefix/perturbation checks and next-bar signal timing checks above.

No higher-timeframe join or event-time fundamentals feature is currently admitted to the training feature pipeline. Such inputs remain out of scope until their source-availability timestamp, publication/receipt time, as-of join behavior, and future-perturbation/prefix-invariance tests are implemented. This is an explicit admission boundary, not a claim that those future data paths have been validated.

The current phase establishes the candidate-only boundary and causality test foundations. It does not claim that ETI already computes PBO, Deflated Sharpe Ratio, White's Reality Check or Hansen SPA; those remain follow-on research-quality work.