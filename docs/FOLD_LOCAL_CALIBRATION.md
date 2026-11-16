# Fold-local probability calibration

ETI nested multi-market research uses a separate chronological block for each stage:

1. **Training:** fit each candidate model once on the fold's training rows.
2. **Candidate selection:** compare candidate predictions on the first part of the validation window.
3. **Calibration:** after a purge gap equal to the label horizon, fit temperature scaling for the selected model on a later, disjoint calibration block.
4. **Outer test:** after another label-horizon purge, assess the frozen selected model on the untouched outer-test block.

The candidate-selection and calibration blocks never overlap. Outer-test labels are not passed to model fitting or temperature optimization. The outer test reports raw and temperature-scaled negative log-likelihood, multiclass Brier score, reliability bins / expected calibration error, and per-class counts. Calibration confirmation is withheld when outer-test class coverage is inadequate or calibrated NLL/Brier is worse than raw probabilities.

If the configured validation window cannot provide at least 30 calibration rows after the internal purge, ETI reports the fold as uncalibrated instead of reusing candidate-selection labels. A selected baseline model must also have at least five observations of each class in the calibration block before temperature fitting can pass.

## Fit budget

Each candidate is fitted once per outer fold. Temperature scaling optimizes one scalar on the calibration probabilities and does not refit the estimator. The research fit-call estimate is therefore `fold_count × candidate_count`; this is a compute proxy, not a Kaggle dollar-cost estimate.

## Interpretation and limitations

- A successful temperature fit is not proof of calibrated future probabilities. The independent outer-test confirmation is reported separately.
- Expected calibration error depends on binning and sample size; inspect reliability bins and class counts rather than treating ECE as a standalone pass/fail score.
- Overlapping outer test windows are statistically dependent.
- Temperature scaling preserves class ranking; it changes confidence, not the predicted class ordering.
- A small test block may not contain enough examples of every class to confirm calibration.
- The current approach calibrates the selected model within each fold. It does not yet establish a single globally calibrated model for deployment.
- These reports do not authorize model promotion, capital allocation, broker selection, or order execution. ETI remains research-only.

Run focused checks with `python -m pytest -q tests/test_calibrated_evaluation.py tests/test_nested_pipeline.py tests/test_model_calibration.py`.
