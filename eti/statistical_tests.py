"""Multiple-testing-aware statistical diagnostics for ETI strategy research."""
from __future__ import annotations
from itertools import combinations
import math
from typing import Any
import numpy as np
from scipy.stats import kurtosis, norm, rankdata, skew


def _return_matrix(returns: Any, *, min_trials: int = 1) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2:
        raise ValueError("returns must be a 2D array shaped (trials, observations)")
    if values.shape[0] < min_trials or values.shape[1] < 3:
        raise ValueError("returns require enough trials and at least three observations")
    if not np.isfinite(values).all():
        raise ValueError("returns must be finite and aligned; do not silently drop observations")
    return values


def _sample_sharpe(values: np.ndarray, annualization: float = 1.0) -> np.ndarray:
    if not math.isfinite(annualization) or annualization <= 0:
        raise ValueError("annualization must be finite and positive")
    std = np.std(values, axis=1, ddof=1)
    means = np.mean(values, axis=1)
    return np.divide(means, std, out=np.zeros_like(means), where=std > 0) * math.sqrt(annualization)


def deflated_sharpe_ratio(returns: Any, *, benchmark_sharpe: float = 0.0,
                          annualization: float = 1.0) -> dict[str, Any]:
    """PSR estimate against an expected-maximum-Sharpe hurdle.

    One row per attempted strategy/configuration. Kurtosis uses Pearson convention.
    """
    matrix = _return_matrix(returns)
    if isinstance(benchmark_sharpe, bool) or not math.isfinite(float(benchmark_sharpe)):
        raise ValueError("benchmark_sharpe must be finite")
    if not math.isfinite(annualization) or annualization <= 0:
        raise ValueError("annualization must be finite and positive")
    n_trials, n_obs = matrix.shape
    sharpes = _sample_sharpe(matrix, annualization)
    best_index = int(np.argmax(sharpes))
    selected = matrix[best_index]
    sr_raw = float(np.mean(selected) / np.std(selected, ddof=1)) if np.std(selected, ddof=1) > 0 else 0.0
    trial_dispersion = float(np.std(sharpes / math.sqrt(annualization), ddof=1)) if n_trials > 1 else 0.0
    if n_trials > 1 and trial_dispersion > 0:
        gamma = 0.5772156649015329
        expected_max = trial_dispersion * (
            (1.0 - gamma) * norm.ppf(1.0 - 1.0 / n_trials)
            + gamma * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        )
    else:
        expected_max = 0.0
    hurdle = max(float(benchmark_sharpe), float(expected_max))
    sample_std = float(np.std(selected, ddof=1))
    if sample_std == 0:
        return {"selected_trial_index": best_index, "trial_count": n_trials,
                "observations": n_obs, "observed_sharpe": 0.0,
                "expected_max_sharpe_hurdle": hurdle * math.sqrt(annualization),
                "probability_sharpe_is_greater": 0.0, "status": "zero_variance_returns",
                "warning": "Degenerate constant-return series; not inferential evidence."}
    sample_skew = float(skew(selected, bias=False))
    sample_kurtosis = float(kurtosis(selected, fisher=False, bias=False)) if n_obs > 3 else 3.0
    variance_term = 1.0 - sample_skew * sr_raw + ((sample_kurtosis - 1.0) / 4.0) * sr_raw**2
    if not math.isfinite(variance_term) or variance_term <= 0:
        raise ValueError("Sharpe sampling variance is non-positive")
    z_score = (sr_raw - hurdle) * math.sqrt(n_obs - 1) / math.sqrt(variance_term)
    return {"selected_trial_index": best_index, "trial_count": n_trials,
            "observations": n_obs, "observed_sharpe": sr_raw * math.sqrt(annualization),
            "expected_max_sharpe_hurdle": hurdle * math.sqrt(annualization),
            "probability_sharpe_is_greater": float(norm.cdf(z_score)),
            "z_score": float(z_score), "sample_skewness": sample_skew,
            "sample_pearson_kurtosis": sample_kurtosis, "status": "ok",
            "warning": "Depends on complete trial disclosure and return sampling assumptions; does not prove future profitability."}


def probability_of_backtest_overfitting(returns: Any, *, n_splits: int = 8) -> dict[str, Any]:
    """CSCV PBO: best in-sample mean return, ranked on complementary blocks."""
    matrix = _return_matrix(returns, min_trials=2)
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 4 or n_splits % 2:
        raise ValueError("n_splits must be an even integer >= 4")
    if matrix.shape[1] < n_splits * 2:
        raise ValueError("observations must provide at least two per CSCV block")
    blocks = np.array_split(np.arange(matrix.shape[1]), n_splits)
    if min(map(len, blocks)) < 2:
        raise ValueError("each CSCV block must contain at least two observations")
    ranks_out, logits = [], []
    for chosen in combinations(range(n_splits), n_splits // 2):
        ins = set(chosen)
        outs = [i for i in range(n_splits) if i not in ins]
        is_idx = np.concatenate([blocks[i] for i in sorted(ins)])
        oos_idx = np.concatenate([blocks[i] for i in outs])
        selected = int(np.argmax(np.mean(matrix[:, is_idx], axis=1)))
        oos_scores = np.mean(matrix[:, oos_idx], axis=1)
        percentile = float(rankdata(oos_scores, method="average")[selected] / (matrix.shape[0] + 1.0))
        percentile = min(max(percentile, 1e-12), 1.0 - 1e-12)
        ranks_out.append(percentile)
        logits.append(math.log(percentile / (1.0 - percentile)))
    return {"pbo": float(np.mean(np.asarray(ranks_out) <= 0.5)),
            "median_oos_rank_percentile": float(np.median(ranks_out)),
            "median_logit_rank": float(np.median(logits)), "split_count": len(logits),
            "trial_count": int(matrix.shape[0]), "observations": int(matrix.shape[1]),
            "method": "combinatorially_symmetric_cross_validation",
            "warning": "CSCV is a selection-overfit diagnostic, not chronological deployment validation."}


def _circular_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(math.ceil(n / block)))
    return np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]


def _bootstrap_max(matrix: np.ndarray, *, n_bootstrap: int, block_length: int,
                   seed: int, spa: bool) -> dict[str, Any]:
    n_trials, n_obs = matrix.shape
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 999:
        raise ValueError("n_bootstrap must be an integer >= 999")
    if isinstance(block_length, bool) or not isinstance(block_length, int) or not 1 <= block_length <= n_obs:
        raise ValueError("block_length must be between 1 and observations")
    means = np.mean(matrix, axis=1)
    stds = np.std(matrix, axis=1, ddof=1)
    denom = np.where(stds > 0, stds, np.inf)
    observed_t = math.sqrt(n_obs) * means / denom
    observed = float(np.max(observed_t))
    if spa:
        threshold = -math.sqrt(2.0 * math.log(max(math.log(n_obs), 1.0000001)))
        recenter = np.where(observed_t < threshold, means, 0.0)
    else:
        recenter = means.copy()
    centered = matrix - recenter[:, None]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = _circular_indices(n_obs, block_length, rng)
        sample = centered[:, idx]
        sample_std = np.std(sample, axis=1, ddof=1)
        sample_denom = np.where(sample_std > 0, sample_std, np.inf)
        stats[b] = float(np.max(math.sqrt(n_obs) * np.mean(sample, axis=1) / sample_denom))
    p = float((1 + np.count_nonzero(stats >= observed)) / (n_bootstrap + 1))
    return {"statistic": observed, "p_value": p, "bootstrap_samples": n_bootstrap,
            "block_length": block_length, "seed": seed, "trial_count": n_trials,
            "observations": n_obs, "recentered_means": recenter.tolist(), "status": "ok",
            "warning": "Block-bootstrap inference depends on block length, stationarity and complete trial-family disclosure."}


def white_reality_check(differential_returns: Any, *, n_bootstrap: int = 1999,
                        block_length: int = 5, seed: int = 7) -> dict[str, Any]:
    """White Reality Check on strategy-minus-benchmark return differentials."""
    result = _bootstrap_max(_return_matrix(differential_returns), n_bootstrap=n_bootstrap,
                            block_length=block_length, seed=seed, spa=False)
    result.update(method="white_reality_check",
                  null="maximum expected differential return is not positive")
    return result


def hansen_spa_test(differential_returns: Any, *, n_bootstrap: int = 1999,
                    block_length: int = 5, seed: int = 7) -> dict[str, Any]:
    """Hansen-style SPA max-studentized test with consistent null recentering."""
    result = _bootstrap_max(_return_matrix(differential_returns), n_bootstrap=n_bootstrap,
                            block_length=block_length, seed=seed, spa=True)
    result.update(method="hansen_spa",
                  null="no strategy has positive expected differential return")
    return result
