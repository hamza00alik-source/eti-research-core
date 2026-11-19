"""Multiple-testing-aware strategy research diagnostics.

These are research diagnostics, not profitability guarantees or live-trading
authorization. Inputs must be aligned, net-of-cost strategy returns relative to
a declared benchmark where the test requires benchmark differentials.
"""
from __future__ import annotations

from itertools import combinations
import math
from typing import Any

import numpy as np
from scipy.stats import kurtosis, norm, rankdata, skew

_EULER_MASCHERONI = 0.5772156649015329

def _return_vector(values: Any, name: str = "returns") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 3:
        raise ValueError(f"{name} must be a one-dimensional array with at least 3 observations")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite observations")
    return array

def _return_matrix(values: Any, *, min_strategies: int = 2, require_nonzero_volatility: bool = True) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 8 or array.shape[1] < min_strategies:
        raise ValueError("returns must be a finite observations-by-strategies matrix")
    if not np.isfinite(array).all():
        raise ValueError("returns matrix must contain only finite observations")
    if require_nonzero_volatility and np.any(np.std(array, axis=0, ddof=1) <= 0):
        raise ValueError("every strategy must have non-zero sample volatility")
    return array

def deflated_sharpe_ratio(returns: Any, *, trials: int, risk_free_per_period: float = 0.0) -> dict[str, Any]:
    """Estimate probability that Sharpe exceeds the multiple-trial benchmark.

    Returns are per-period arithmetic returns. The reported Sharpe is not
    annualized. trials must count every explored strategy/parameter/model.
    """
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer")
    if (isinstance(risk_free_per_period, bool) or not isinstance(risk_free_per_period, (int, float))
            or not math.isfinite(float(risk_free_per_period))):
        raise ValueError("risk_free_per_period must be finite")
    values = _return_vector(returns) - float(risk_free_per_period)
    observations = int(values.size)
    volatility = float(np.std(values, ddof=1))
    volatility_floor = np.finfo(float).eps * max(1.0, float(np.max(np.abs(values))))
    if volatility <= volatility_floor:
        raise ValueError("returns must have positive sample volatility")
    observed = float(np.mean(values) / volatility)
    sample_skew = float(skew(values, bias=False))
    sample_kurtosis = float(kurtosis(values, fisher=False, bias=False))
    variance_factor = 1.0 - sample_skew * observed + ((sample_kurtosis - 1.0) / 4.0) * observed * observed
    if not math.isfinite(variance_factor) or variance_factor <= 0:
        raise ValueError("Sharpe sampling variance is not positive; sample is unsuitable for DSR")
    sharpe_variance = variance_factor / (observations - 1)
    sharpe_standard_error = math.sqrt(sharpe_variance)
    if trials == 1:
        expected_maximum = 0.0
    else:
        upper = norm.ppf(1.0 - 1.0 / trials)
        lower = norm.ppf(1.0 - 1.0 / (trials * math.e))
        expected_maximum = sharpe_standard_error * ((1.0 - _EULER_MASCHERONI) * upper + _EULER_MASCHERONI * lower)
    z_score = (observed - expected_maximum) / sharpe_standard_error
    probability = float(norm.cdf(z_score))
    return {
        "method": "deflated_sharpe_ratio", "observations": observations, "trial_count": trials,
        "period_sharpe": observed, "expected_max_sharpe_under_trials": float(expected_maximum),
        "sharpe_standard_error": sharpe_standard_error, "sample_skewness": sample_skew,
        "sample_kurtosis_pearson": sample_kurtosis, "dsr_probability": probability,
        "dsr_pass_95": probability >= 0.95, "annualized": False,
        "warning": "Credibility depends on complete trial count and return series. Serial dependence, nonstationarity and unreported searches can invalidate this estimate.",
    }

def _score_columns(values: np.ndarray, metric: str) -> np.ndarray:
    means = np.mean(values, axis=0)
    if metric == "mean":
        return means
    if metric == "sharpe":
        deviations = np.std(values, axis=0, ddof=1)
        zero_volatility = deviations <= np.finfo(float).eps
        if np.any(zero_volatility & (np.abs(means) > 1e-15)):
            raise ValueError("zero-volatility strategies must have zero mean for Sharpe ranking")
        return np.divide(means, deviations, out=np.zeros_like(means), where=~zero_volatility)
    raise ValueError("metric must be mean or sharpe")

def probability_of_backtest_overfitting(values: Any, *, n_splits: int = 8, metric: str = "sharpe", max_combinations: int = 10000) -> dict[str, Any]:
    """Estimate PBO with combinatorially symmetric cross-validation (CSCV)."""
    if metric not in {"mean", "sharpe"}:
        raise ValueError("metric must be mean or sharpe")
    returns = _return_matrix(values, require_nonzero_volatility=False)
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 4 or n_splits % 2:
        raise ValueError("n_splits must be an even integer >= 4")
    if returns.shape[0] % n_splits:
        raise ValueError("observation count must be divisible by n_splits for equal CSCV blocks")
    if isinstance(max_combinations, bool) or not isinstance(max_combinations, int) or max_combinations < 1:
        raise ValueError("max_combinations must be a positive integer")
    half = n_splits // 2
    split_count = math.comb(n_splits, half)
    if split_count > max_combinations:
        raise ValueError("CSCV split count exceeds max_combinations; reduce n_splits or raise the explicit cap")
    blocks = np.split(returns, n_splits, axis=0)
    logits, selected_indices, relative_ranks = [], [], []
    for in_blocks in combinations(range(n_splits), half):
        in_set = set(in_blocks)
        out_blocks = [index for index in range(n_splits) if index not in in_set]
        in_sample = np.concatenate([blocks[index] for index in in_blocks], axis=0)
        out_sample = np.concatenate([blocks[index] for index in out_blocks], axis=0)
        selected = int(np.argmax(_score_columns(in_sample, metric)))
        out_scores = _score_columns(out_sample, metric)
        rank = float(rankdata(out_scores, method="average")[selected])
        omega = min(max(rank / (returns.shape[1] + 1.0), 1e-12), 1.0 - 1e-12)
        logits.append(float(math.log(omega / (1.0 - omega))))
        selected_indices.append(selected)
        relative_ranks.append(omega)
    pbo = float(np.mean(np.asarray(logits) <= 0.0))
    return {
        "method": "combinatorially_symmetric_cross_validation", "metric": metric,
        "observations": int(returns.shape[0]), "strategy_count": int(returns.shape[1]),
        "n_splits": n_splits, "combination_count": split_count, "pbo": pbo,
        "median_logit_oos_rank": float(np.median(logits)), "mean_logit_oos_rank": float(np.mean(logits)),
        "selected_strategy_counts": {str(i): int(selected_indices.count(i)) for i in sorted(set(selected_indices))},
        "logit_oos_ranks": logits,
        "warning": "CSCV requires aligned strategy returns and contiguous evaluation blocks. PBO is a selection-overfit diagnostic, not a forecast of future failure.",
    }

def _circular_block_indices(observations: int, *, block_length: int, rng: np.random.Generator) -> np.ndarray:
    chunks, collected = [], 0
    while collected < observations:
        start = int(rng.integers(0, observations))
        chunks.append((start + np.arange(block_length)) % observations)
        collected += block_length
    return np.concatenate(chunks)[:observations]

def _bootstrap_means(centered: np.ndarray, *, repetitions: int, block_length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty((repetitions, centered.shape[1]), dtype=float)
    for index in range(repetitions):
        sample = _circular_block_indices(centered.shape[0], block_length=block_length, rng=rng)
        output[index] = centered[sample].mean(axis=0)
    return output

def _validate_bootstrap(repetitions: int, block_length: int, observations: int) -> None:
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 200:
        raise ValueError("repetitions must be an integer >= 200")
    if isinstance(block_length, bool) or not isinstance(block_length, int) or not 1 <= block_length <= observations:
        raise ValueError("block_length must be between 1 and the observation count")

def white_reality_check(performance_differentials: Any, *, repetitions: int = 2000, block_length: int = 5, seed: int = 42) -> dict[str, Any]:
    """White Reality Check for best mean benchmark-relative return differential."""
    values = _return_matrix(performance_differentials, require_nonzero_volatility=False)
    _validate_bootstrap(repetitions, block_length, values.shape[0])
    means = values.mean(axis=0)
    observed = float(max(0.0, math.sqrt(values.shape[0]) * float(np.max(means))))
    bootstrap = _bootstrap_means(values - means, repetitions=repetitions, block_length=block_length, seed=seed)
    statistics = np.maximum(0.0, math.sqrt(values.shape[0]) * np.max(bootstrap, axis=1))
    p_value = (int(np.sum(statistics >= observed)) + 1) / (repetitions + 1)
    return {
        "method": "white_reality_check_circular_block_bootstrap", "observations": int(values.shape[0]),
        "strategy_count": int(values.shape[1]), "observed_max_mean_differential": float(np.max(means)),
        "test_statistic": observed, "p_value": float(p_value), "bootstrap_repetitions": repetitions,
        "block_length": block_length, "seed": seed, "reject_null_at_5pct": bool(p_value < 0.05),
        "warning": "Family-wise test of maximum mean differential, not a Sharpe test. Use aligned net-of-cost returns and a common benchmark; choose block length before inspecting results.",
    }

def hansen_spa_test(performance_differentials: Any, *, repetitions: int = 2000, block_length: int = 5, seed: int = 42) -> dict[str, Any]:
    """Hansen SPA-style studentized max test with consistent null recentering."""
    values = _return_matrix(performance_differentials, require_nonzero_volatility=False)
    _validate_bootstrap(repetitions, block_length, values.shape[0])
    n = values.shape[0]
    means = values.mean(axis=0)
    deviations = values.std(axis=0, ddof=1)
    if np.any((deviations == 0.0) & (np.abs(means) > 1e-15)):
        raise ValueError("zero-volatility strategy differentials must have zero mean")
    safe_deviations = np.where(deviations > 0.0, deviations, 1.0)
    observed_t = math.sqrt(n) * means / safe_deviations
    observed = float(max(0.0, float(np.max(observed_t))))
    threshold = -math.sqrt(2.0 * math.log(max(math.log(n), 1.0000001)))
    recentering = np.where(observed_t < threshold, means, 0.0)
    # Under the null, center viable candidates at zero while retaining the
    # observed negative mean only for clearly inferior alternatives.
    null_centered = values - means + recentering
    bootstrap_means = _bootstrap_means(null_centered, repetitions=repetitions, block_length=block_length, seed=seed)
    bootstrap_t = math.sqrt(n) * bootstrap_means / safe_deviations
    statistics = np.maximum(0.0, np.max(bootstrap_t, axis=1))
    p_value = (int(np.sum(statistics >= observed)) + 1) / (repetitions + 1)
    return {
        "method": "hansen_spa_circular_block_bootstrap", "observations": int(n),
        "strategy_count": int(values.shape[1]), "observed_max_studentized_statistic": observed,
        "p_value": float(p_value), "bootstrap_repetitions": repetitions, "block_length": block_length,
        "seed": seed, "consistent_recentring_threshold": float(threshold),
        "recentered_strategy_count": int(np.sum(recentering != 0.0)),
        "reject_null_at_5pct": bool(p_value < 0.05),
        "warning": "SPA validity depends on a common benchmark, aligned net returns, defensible block length and complete strategy family. Bootstrap inference is approximate.",
    }

def assess_research_trial_family(
    family: dict[str, Any],
    *,
    selected_trial_id: str,
    bootstrap_repetitions: int = 2000,
    block_length: int = 5,
    seed: int = 42,
    pbo_splits: int = 8,
) -> dict[str, Any]:
    """Run DSR/PBO/White-RC/SPA from one immutable ledger family snapshot.

    DSR uses every started attempt (including failed attempts) as the trial
    count. PBO and benchmark-relative bootstrap tests use only completed,
    timestamp-aligned return columns supplied by the ledger.
    """
    if not isinstance(family, dict):
        raise ValueError("family must be a research trial family object")
    family_id = family.get("family_id")
    benchmark_id = family.get("benchmark_id")
    if not isinstance(family_id, str) or not family_id.strip():
        raise ValueError("family_id is required")
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise ValueError("benchmark_id is required")
    trial_ids = family.get("trial_ids")
    strategy_names = family.get("strategy_names")
    if (not isinstance(trial_ids, list) or len(trial_ids) < 2
            or not isinstance(strategy_names, list) or len(strategy_names) != len(trial_ids)):
        raise ValueError("family must contain at least two completed trial IDs and matching strategy names")
    if len(set(trial_ids)) != len(trial_ids) or any(not isinstance(item, str) or not item for item in trial_ids):
        raise ValueError("completed trial IDs must be unique non-empty strings")
    attempted = family.get("attempted_trial_count")
    if isinstance(attempted, bool) or not isinstance(attempted, int) or attempted < len(trial_ids):
        raise ValueError("attempted_trial_count cannot be less than completed trial count")
    if selected_trial_id not in trial_ids:
        raise ValueError("selected_trial_id must identify a completed trial in this family")
    observation_times = family.get("observation_times")
    if not isinstance(observation_times, list) or len(observation_times) < 8:
        raise ValueError("family must include the aligned observation timestamp vector")
    matrix = _return_matrix(family.get("net_return_differentials"), require_nonzero_volatility=False)
    if matrix.shape != (len(observation_times), len(trial_ids)):
        raise ValueError("family return matrix dimensions do not match timestamps and completed trial IDs")
    selected_index = trial_ids.index(selected_trial_id)
    selected_returns = matrix[:, selected_index]
    if np.std(selected_returns, ddof=1) <= np.finfo(float).eps:
        dsr = {
            "method": "deflated_sharpe_ratio",
            "status": "undefined_zero_volatility",
            "observations": int(len(selected_returns)),
            "trial_count": attempted,
            "period_sharpe": None,
            "dsr_probability": None,
            "dsr_pass_95": False,
            "warning": "Selected trial has zero volatility; DSR is undefined. Family PBO and benchmark-relative bootstrap tests are still reported.",
        }
    else:
        dsr = deflated_sharpe_ratio(selected_returns, trials=attempted)
    pbo = probability_of_backtest_overfitting(matrix, n_splits=pbo_splits, metric="sharpe")
    white = white_reality_check(
        matrix, repetitions=bootstrap_repetitions, block_length=block_length, seed=seed
    )
    spa = hansen_spa_test(
        matrix, repetitions=bootstrap_repetitions, block_length=block_length, seed=seed
    )
    return {
        "kind": "research_trial_family_statistical_assessment",
        "family_id": family_id,
        "benchmark_id": benchmark_id,
        "training_data_sha256": family.get("training_data_sha256"),
        "evaluation_data_sha256": family.get("evaluation_data_sha256"),
        "cost_model_sha256": family.get("cost_model_sha256"),
        "selected_trial_id": selected_trial_id,
        "attempted_trial_count": attempted,
        "completed_trial_count": len(trial_ids),
        "failed_trial_count": family.get("failed_trial_count", 0),
        "selected_trial_deflated_information_ratio": dsr,
        "selected_trial_return_basis": "net strategy return differential versus the family benchmark",
        "family_pbo": pbo,
        "white_reality_check": white,
        "hansen_spa": spa,
        "warning": (
            "This report consumes declared ledger inputs. It does not authenticate source data, "
            "benchmark construction, cost evidence or completeness of registered trials; statistical "
            "tests do not establish future profitability or authorize live trading."
        ),
    }
