import numpy as np
import pytest

from eti.statistical_validation import (
    assess_research_trial_family,
    deflated_sharpe_ratio,
    hansen_spa_test,
    probability_of_backtest_overfitting,
    white_reality_check,
)


def test_deflated_sharpe_penalizes_a_larger_trial_family():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.01, 800)
    one = deflated_sharpe_ratio(returns, trials=1)
    many = deflated_sharpe_ratio(returns, trials=100)
    assert one["method"] == "deflated_sharpe_ratio"
    assert one["annualized"] is False
    assert many["expected_max_sharpe_under_trials"] > one["expected_max_sharpe_under_trials"]
    assert many["dsr_probability"] < one["dsr_probability"]


def test_deflated_sharpe_rejects_invalid_trial_counts_and_returns():
    with pytest.raises(ValueError, match="trials"):
        deflated_sharpe_ratio(np.arange(10.0), trials=0)
    with pytest.raises(ValueError, match="finite"):
        deflated_sharpe_ratio([0.1, np.nan, 0.2], trials=1)
    with pytest.raises(ValueError, match="volatility"):
        deflated_sharpe_ratio([0.01] * 20, trials=5)


def test_cscv_pbo_is_deterministic_and_reports_all_combinations():
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0002, 0.01, size=(400, 6))
    first = probability_of_backtest_overfitting(returns, n_splits=8)
    second = probability_of_backtest_overfitting(returns, n_splits=8)
    assert first["method"] == "combinatorially_symmetric_cross_validation"
    assert first["combination_count"] == 70
    assert first["pbo"] == second["pbo"]
    assert first["logit_oos_ranks"] == second["logit_oos_ranks"]
    assert 0.0 <= first["pbo"] <= 1.0


def test_cscv_flags_block_specific_winners_that_reverse_out_of_sample():
    # Every strategy wins only in its own block and loses in every other block.
    # The in-sample winner is therefore systematically poor on the complementary blocks.
    returns = np.full((800, 8), -0.001)
    for block in range(8):
        start, end = block * 100, (block + 1) * 100
        returns[start:end, block] = 0.02
    report = probability_of_backtest_overfitting(returns, n_splits=8, metric="mean")
    assert report["combination_count"] == 70
    assert report["pbo"] == 1.0


def test_cscv_rejects_unequal_blocks_odd_splits_and_excessive_combinations():
    returns = np.random.default_rng(12).normal(size=(402, 3))
    with pytest.raises(ValueError, match="divisible"):
        probability_of_backtest_overfitting(returns, n_splits=8)
    with pytest.raises(ValueError, match="even"):
        probability_of_backtest_overfitting(returns[:400], n_splits=5)
    with pytest.raises(ValueError, match="exceeds"):
        probability_of_backtest_overfitting(returns[:400], n_splits=8, max_combinations=10)


def test_cscv_rejects_non_finite_and_invalid_constant_strategy_columns():
    returns = np.random.default_rng(13).normal(size=(400, 3))
    returns[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        probability_of_backtest_overfitting(returns)
    returns = np.random.default_rng(13).normal(size=(400, 3))
    returns[:, 0] = 0.0  # Constant cash/no-trade strategy has a zero Sharpe by convention.
    report = probability_of_backtest_overfitting(returns)
    assert 0.0 <= report["pbo"] <= 1.0
    returns[:, 1] = 0.01  # Constant positive return with zero volatility is invalid for Sharpe ranking.
    with pytest.raises(ValueError, match="zero-volatility"):
        probability_of_backtest_overfitting(returns)


def test_white_reality_check_detects_a_large_net_benchmark_relative_edge():
    rng = np.random.default_rng(21)
    returns = rng.normal(0.0, 0.01, size=(500, 4))
    returns[:, 0] += 0.004
    returns[:, 3] = 0.0  # A cash/no-trade comparator has a valid zero-volatility differential.
    report = white_reality_check(returns, repetitions=500, block_length=4, seed=9)
    assert report["method"] == "white_reality_check_circular_block_bootstrap"
    assert report["p_value"] < 0.05
    assert report["reject_null_at_5pct"] is True


def test_hansen_spa_detects_a_large_net_benchmark_relative_edge():
    rng = np.random.default_rng(22)
    returns = rng.normal(0.0, 0.01, size=(500, 4))
    returns[:, 2] += 0.004
    returns[:, 3] = 0.0  # The cash/no-trade differential is exactly zero.
    report = hansen_spa_test(returns, repetitions=500, block_length=4, seed=10)
    assert report["method"] == "hansen_spa_circular_block_bootstrap"
    assert report["p_value"] < 0.05
    assert report["reject_null_at_5pct"] is True


def test_bootstrap_tests_are_reproducible_and_validate_parameters():
    returns = np.random.default_rng(23).normal(0.0, 0.01, size=(100, 3))
    first = white_reality_check(returns, repetitions=250, block_length=3, seed=77)
    second = white_reality_check(returns, repetitions=250, block_length=3, seed=77)
    assert first["p_value"] == second["p_value"]
    assert first["test_statistic"] == second["test_statistic"]
    with pytest.raises(ValueError, match="repetitions"):
        hansen_spa_test(returns, repetitions=10)
    with pytest.raises(ValueError, match="block_length"):
        white_reality_check(returns, block_length=101)

def test_trial_family_assessment_uses_full_attempted_count_and_all_four_tests():
    rng = np.random.default_rng(31)
    matrix = rng.normal(0.0001, 0.01, size=(400, 3))
    matrix[:, 1] += 0.001
    family = {
        "family_id": "family-x",
        "benchmark_id": "cash-usd",
        "training_data_sha256": "a" * 64,
        "evaluation_data_sha256": "b" * 64,
        "cost_model_sha256": "c" * 64,
        "attempted_trial_count": 7,
        "completed_trial_count": 3,
        "failed_trial_count": 4,
        "trial_ids": ["trial-a", "trial-b", "trial-c"],
        "strategy_names": ["trend", "momentum", "breakout"],
        "observation_times": [f"t-{i}" for i in range(400)],
        "net_return_differentials": matrix.tolist(),
    }
    report = assess_research_trial_family(
        family, selected_trial_id="trial-b",
        bootstrap_repetitions=250, block_length=4, seed=17,
    )
    assert report["kind"] == "research_trial_family_statistical_assessment"
    assert report["attempted_trial_count"] == 7
    assert report["selected_trial_deflated_information_ratio"]["trial_count"] == 7
    assert "differential" in report["selected_trial_return_basis"]
    assert report["family_pbo"]["combination_count"] == 70
    assert report["white_reality_check"]["bootstrap_repetitions"] == 250
    assert report["hansen_spa"]["bootstrap_repetitions"] == 250


def test_trial_family_assessment_rejects_missing_selected_trial_and_underreported_count():
    matrix = np.random.default_rng(32).normal(0.0, 0.01, size=(400, 2))
    family = {
        "family_id": "family-x", "benchmark_id": "cash",
        "attempted_trial_count": 2, "trial_ids": ["a", "b"],
        "strategy_names": ["one", "two"], "observation_times": [f"t-{i}" for i in range(400)],
        "net_return_differentials": matrix.tolist(),
    }
    with pytest.raises(ValueError, match="selected_trial_id"):
        assess_research_trial_family(family, selected_trial_id="missing", bootstrap_repetitions=200)
    family["attempted_trial_count"] = 1
    with pytest.raises(ValueError, match="cannot be less"):
        assess_research_trial_family(family, selected_trial_id="a", bootstrap_repetitions=200)

def test_trial_family_assessment_reports_undefined_dsr_but_keeps_other_tests_for_no_trade_family():
    family = {
        "family_id": "cash-only",
        "benchmark_id": "cash",
        "attempted_trial_count": 3,
        "trial_ids": ["cash-a", "cash-b"],
        "strategy_names": ["no-trade-a", "no-trade-b"],
        "observation_times": [f"t-{i}" for i in range(400)],
        "net_return_differentials": np.zeros((400, 2)).tolist(),
    }
    report = assess_research_trial_family(
        family, selected_trial_id="cash-a",
        bootstrap_repetitions=200, block_length=4, seed=4,
    )
    assert report["selected_trial_deflated_information_ratio"]["status"] == "undefined_zero_volatility"
    assert report["selected_trial_deflated_information_ratio"]["dsr_probability"] is None
    assert report["family_pbo"]["pbo"] == 0.5
    assert "median-rank ties count 0.5" in report["family_pbo"]["tie_handling"]
    assert report["white_reality_check"]["p_value"] == 1.0
    assert report["hansen_spa"]["p_value"] == 1.0


def test_trial_family_keeps_bootstrap_diagnostics_when_cscv_blocks_do_not_divide_evenly():
    rng = np.random.default_rng(2026)
    matrix = rng.normal(0.0001, 0.01, size=(10, 3))
    family = {
        "family_id": "uneven-cscv-family",
        "benchmark_id": "no_trade",
        "attempted_trial_count": 3,
        "trial_ids": ["trial-a", "trial-b", "trial-c"],
        "strategy_names": ["candidate-a", "candidate-b", "candidate-c"],
        "observation_times": [f"t-{i}" for i in range(10)],
        "net_return_differentials": matrix.tolist(),
    }
    report = assess_research_trial_family(
        family, selected_trial_id="trial-a",
        bootstrap_repetitions=200, block_length=2, pbo_splits=8, seed=2026,
    )
    assert report["family_pbo"]["status"] == "not_computed"
    assert "divisible" in report["family_pbo"]["reason"]
    assert report["white_reality_check"]["bootstrap_repetitions"] == 200
    assert report["hansen_spa"]["bootstrap_repetitions"] == 200
