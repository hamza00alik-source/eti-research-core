import numpy as np
import pytest

from eti.statistical_tests import (
    deflated_sharpe_ratio, hansen_spa_test,
    probability_of_backtest_overfitting, white_reality_check,
)


def test_dsr_penalizes_larger_trial_family():
    rng = np.random.default_rng(10)
    values = rng.normal(0.0002, 0.01, size=(40, 500))
    small, large = deflated_sharpe_ratio(values[:2]), deflated_sharpe_ratio(values)
    assert large["trial_count"] == 40
    assert large["expected_max_sharpe_hurdle"] >= small["expected_max_sharpe_hurdle"]
    assert 0 <= large["probability_sharpe_is_greater"] <= 1


def test_dsr_zero_variance_and_invalid_inputs():
    assert deflated_sharpe_ratio(np.ones((1, 20)))["status"] == "zero_variance_returns"
    with pytest.raises(ValueError, match="finite"):
        deflated_sharpe_ratio([[0.1, np.nan, 0.2]])
    with pytest.raises(ValueError, match="2D"):
        deflated_sharpe_ratio([0.1, 0.2, 0.3])


def test_pbo_all_cscv_partitions_and_bounds():
    rng = np.random.default_rng(2)
    result = probability_of_backtest_overfitting(rng.normal(size=(8, 160)), n_splits=8)
    assert result["split_count"] == 70
    assert 0 <= result["pbo"] <= 1


def test_pbo_rejects_odd_split_count():
    with pytest.raises(ValueError, match="even"):
        probability_of_backtest_overfitting(np.ones((3, 80)), n_splits=5)


@pytest.mark.parametrize("test", [white_reality_check, hansen_spa_test])
def test_bootstrap_deterministic_and_bounded(test):
    rng = np.random.default_rng(3)
    values = rng.normal(0.0, 0.01, size=(5, 120))
    a = test(values, n_bootstrap=999, block_length=4, seed=12)
    b = test(values, n_bootstrap=999, block_length=4, seed=12)
    assert a["p_value"] == b["p_value"]
    assert 0 < a["p_value"] <= 1


def test_white_reality_check_detects_strong_positive_differential():
    rng = np.random.default_rng(11)
    result = white_reality_check(rng.normal(0.003, 0.001, size=(3, 300)),
                                 n_bootstrap=999, block_length=3)
    assert result["p_value"] < 0.05


def test_bootstrap_requires_minimum_resamples():
    with pytest.raises(ValueError, match="999"):
        hansen_spa_test(np.ones((2, 20)), n_bootstrap=100)
