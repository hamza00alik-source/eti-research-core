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

Before a candidate can be promoted, the reviewer must inspect locked chronological out-of-sample evidence and fold-level results; measured or independently sourced fees, spreads and slippage with assumptions clearly labeled; cost stress scenarios; drawdown, turnover, liquidity and concentration behavior; leakage, provenance and historical-availability checks; the number of hypotheses and parameter/model trials considered; and calibration and artifact-schema integrity.

The current phase establishes the candidate-only boundary and causality test foundations. It does not claim that ETI already computes PBO, Deflated Sharpe Ratio, White's Reality Check or Hansen SPA; those remain follow-on research-quality work.