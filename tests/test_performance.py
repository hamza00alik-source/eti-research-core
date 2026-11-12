import json

import pandas as pd
import pytest

from eti.performance import evaluate_predictions


def _series(values):
    return pd.Series(values, dtype=float)


def test_performance_report_scores_predictions_and_cost_aware_baselines():
    close = _series([100, 102, 104, 103, 101, 99, 100, 103])
    labels = pd.Series([2, 2, 2, 1, 0, 0, 2, 2])
    predictions = labels.copy()
    report = evaluate_predictions(close, labels, predictions, fee_bps=6, slippage_bps=2)

    assert report["classification"]["accuracy"] == 1.0
    assert report["classification"]["balanced_accuracy"] == 1.0
    assert report["classification"]["macro_f1"] == 1.0
    assert report["classification"]["confusion_matrix"] == [
        [2, 0, 0], [0, 1, 0], [0, 0, 5]
    ]
    assert report["trading"]["model"]["bars"] == len(close)
    assert report["trading"]["model"]["trade_count"] >= 1
    assert report["trading"]["model"]["short_exposure"] == 0.0
    assert report["trading"]["model_long_short"]["bars"] == len(close)
    assert report["trading"]["model"]["trading_cost_total"] > 0
    assert report["trading"]["no_trade"]["total_return"] == 0.0
    assert report["trading"]["no_trade"]["max_drawdown"] == 0.0
    assert report["trading"]["buy_and_hold"]["total_return"] != 0.0
    assert report["trading"]["cost_assumptions_bps"]["fee_per_unit_turnover"] == 6
    assert report["trading"]["label_oracle_upper_bound"]["bars"] == len(close)
    json.dumps(report, allow_nan=False)


def test_higher_costs_do_not_improve_identical_active_strategy_return():
    close = _series([100, 101, 103, 102, 104, 103, 105])
    labels = pd.Series([2, 2, 2, 0, 0, 2, 2])
    low = evaluate_predictions(close, labels, labels, fee_bps=0, slippage_bps=0)
    high = evaluate_predictions(close, labels, labels, fee_bps=20, slippage_bps=10)
    assert high["trading"]["model"]["total_return"] < low["trading"]["model"]["total_return"]


def test_long_only_mode_has_no_short_exposure():
    close = _series([100, 99, 98, 100, 102, 101])
    labels = pd.Series([0, 0, 0, 2, 2, 2])
    report = evaluate_predictions(close, labels, labels, allow_short=False)
    assert report["trading"]["model"]["short_exposure"] == 0.0


def test_performance_rejects_misaligned_and_invalid_labels():
    close = pd.Series([100, 101, 102], index=[0, 1, 2])
    labels = pd.Series([2, 1, 0], index=[1, 2, 3])
    with pytest.raises(ValueError, match="indexes must match"):
        evaluate_predictions(close, labels, labels)
    aligned = pd.Series([2, 1, 9], index=[0, 1, 2])
    with pytest.raises(ValueError, match="class labels"):
        evaluate_predictions(close, aligned, aligned)


def test_performance_rejects_invalid_costs_and_time_scaling():
    close = _series([100, 101, 102])
    labels = pd.Series([2, 2, 2])
    with pytest.raises(ValueError, match="fee_bps"):
        evaluate_predictions(close, labels, labels, fee_bps=-1)
    with pytest.raises(ValueError, match="bars_per_year"):
        evaluate_predictions(close, labels, labels, bars_per_year=0)
