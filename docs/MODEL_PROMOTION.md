# Champion/challenger promotion gate

`eti.promotion.assess_challenger_promotion` evaluates a preregistered paired comparison over identical, non-overlapping outer test windows. It requires:
- SHA-256 identifiers for model, training data, locked evaluation data and cost model;
- distinct model hashes, but the same training-data, evaluation-data and cost-model hashes for champion and challenger;
- at least five non-overlapping paired folds and at least 1,000 test bars by default;
- cost-adjusted net return per fold, max drawdown and out-of-sample ECE;
- a declared comparison-family size, used for Bonferroni adjustment.

The statistical test is an exact one-sided sign test over non-tied fold-level net-return improvements; ties are excluded from the sign-test denominator. This deliberately avoids treating individual bars as independent samples. The challenger must also meet per-fold and mean drawdown/ECE ceilings and may not worsen mean drawdown or out-of-sample ECE beyond small configured tolerances versus the champion. A positive result only returns `eligible_for_human_review`; it never modifies an artifact, changes champion status, approves a model, or enables orders.

## Important limitations

- `evaluation_locked`, hashes and comparison-family size are declarations. The module cannot authenticate data access history, prove that the evaluation set was never inspected, or verify preregistration.
- Non-overlapping test intervals do not guarantee statistical independence across market regimes.
- Five folds is a minimum guardrail, not a universal sample-size guarantee. A production process should normally require substantially more independent time and regimes.
- The sign test ignores effect magnitude beyond the minimum mean improvement threshold and should be complemented by economic and stability review.
- ECE must be measured on a genuinely out-of-sample calibration/evaluation segment; the existing model calibration report's before/after ECE on a calibration window is not automatically a substitute.
- Repeatedly tuning after seeing the same holdout invalidates the evidence. Use a fresh locked holdout for a new comparison family.

This is a governance primitive only. No model is automatically promoted and no live execution permission is granted.


## Candidate registration and approval boundary

Autonomous training workers register every result as a `candidate`; they do not approve models, even when the training report marks its internal evaluation checks as passed. Calibration and artifact-schema verification are necessary artifact-integrity checks, not sufficient economic evidence for promotion.

A candidate can become approved only through a separately authorized review workflow that evaluates the locked champion/challenger evidence described above and records the human reviewer. A positive statistical comparison only establishes eligibility for human review; it does not itself approve the model. Existing approved models are not automatically demoted by this change.

The worker's output therefore reports `approval_required=true` for newly registered candidates and never assigns `eti.quality_gate` as an automatic approval authority.
