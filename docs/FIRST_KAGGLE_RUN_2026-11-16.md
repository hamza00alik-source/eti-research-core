# First ETI Kaggle research run — 2026-11-16

## Provenance

- GitHub output-retrieval workflow run: `43861460394` (success).
- Retrieved GitHub Actions artifact: `14223945750` (`eti-kaggle-training-output`).
- Notebook source revision used by the run: `cfb894e8ace194cac73c0f413b11ac5f6befc7e7`.
- Public data provider: Coinbase Exchange.
- Requested symbol: `BTC/USDT`; actual venue symbol: `BTC/USD`; the mapping was explicitly recorded.
- Timeframe: 1 hour.
- Completed candles: 1,999 of 2,000 requested.
- Usable feature rows: 1,564.
- Nested outer-test folds: 5.
- Nested and deployable model-fit calls: 11 total, within the 32-call ceiling.
- Training completed: true; execution enabled: false.

## Exported candidate artifact

- Model ID: `eti-btc-1h`.
- Original base version in this run: `0.1.0`.
- Artifact size: 275,276 bytes.
- SHA-256: `22483eac09f7d3b33de6f049559837e313bae8f27354f0175736876e9bc6f27d`.
- The artifact bytes were independently hashed after retrieving the workflow ZIP; the digest matched the report.
- Separate chronological deployable-model calibration used 387 rows.
- Independent chronological probability validation used 393 rows and reported passed.
- Calibration improved held-out validation NLL from 1.41499 to 1.09342 and Brier score from 0.80715 to 0.66289.

The aggregate fold-local probability report assessed and confirmed four of five folds; one fold selected the majority-class comparator and had no temperature-scaled calibration. The export calibration and fold-local nested calibration are separate evidence streams and should not be conflated.

## Economic result

<badge>Research only — NO-GO</badge>

- Mean outer-test net return at base illustrative costs: **-1.3388%**.
- Positive-return folds: **0/5**.
- Folds beating no-trade: **0/5**.
- Folds beating buy-and-hold: **2/5**.
- Mean annualized Sharpe: **-9.1883**.
- Mean maximum drawdown: **-1.9349%**.
- Mean net return at 2x illustrative costs: **-2.8519%**.
- 2x-cost positive-return folds: **0/5**.

The simulation used illustrative assumptions of 6 bps fee and 2 bps slippage at 1x costs (12 bps and 4 bps at 2x). They are not verified venue fee-tier or realized-fill measurements.

## Disposition and restrictions

This is a successful training/export pipeline run but a failed economic candidate. It must remain a research-only `no_go`. Probability calibration does not offset negative trading economics. Do not promote, approve for live inference, allocate capital to, or trade this candidate.

The subsequent immutable-version and specialist-metadata changes were merged after this notebook revision; this artifact predates those changes. A fresh intentional Kaggle run is needed to produce an artifact carrying the new experiment-lineage metadata. Kaggle execution is manually confirmed and is not silently re-triggered by this documentation update.
