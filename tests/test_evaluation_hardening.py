import numpy as np
import pandas as pd
import pytest

from eti.config import RiskConfig
from eti.portfolio import PortfolioLimits, assess_portfolio
from eti.evaluation import run_nested_walk_forward, run_multi_market_walk_forward, bootstrap_fold_confidence_interval


def test_risk_config_rejects_invalid_caps_and_live_execution():
    with pytest.raises(ValueError):
        RiskConfig(max_gross_exposure=1.2)
    with pytest.raises(ValueError):
        RiskConfig(max_leverage=0.5)
    with pytest.raises(ValueError):
        RiskConfig(allow_live_orders=True)


def test_portfolio_limits_reject_invalid_count_and_fraction():
    with pytest.raises(ValueError):
        PortfolioLimits(max_positions=2.5)
    with pytest.raises(ValueError):
        PortfolioLimits(max_single_asset_fraction=1.5)


def test_portfolio_rejects_malformed_position_mapping():
    result = assess_portfolio(1000, None, "BTC/USDT", 10)
    assert not result.approved
    assert "invalid_existing_positions" in result.reasons


def test_nested_walk_forward_selects_on_validation_and_reports_outer_test():
    frame = pd.DataFrame({"x": np.arange(320, dtype=float),
                          "label": np.tile([0, 1], 160)})
    def majority(train_x, train_y, eval_x):
        return np.repeat(int(train_y.mode().iloc[0]), len(eval_x))
    def alternate(train_x, train_y, eval_x):
        return np.zeros(len(eval_x), dtype=int)
    result = run_nested_walk_forward(
        frame, ["x"], "label", {"majority": majority, "zero": alternate},
        min_train=100, validation_size=30, test_size=30, step_size=30, purge_bars=5)
    assert result["fold_count"] > 0
    assert all("selected_candidate" in fold for fold in result["folds"])
    assert "test" not in result["folds"][0]["validation_scores"]
    assert set(result["folds"][0]["test_candidate_metrics"]) == {"majority", "zero"}
    assert all(
        "balanced_accuracy" in metrics and "accuracy" in metrics
        and "per_class" in metrics and "confusion_matrix" in metrics
        for metrics in result["folds"][0]["test_candidate_metrics"].values()
    )
    assert set(result["folds"][0]["test_candidate_metrics"]["majority"]["per_class"]) == {"0", "1"}
    assert len(result["folds"][0]["test_candidate_metrics"]["majority"]["confusion_matrix"]) == 2
    assert set(result["candidate_summary"]) == {"majority", "zero"}
    assert sum(
        item["selected_on_validation_count"] for item in result["candidate_summary"].values()
    ) == result["fold_count"]
    assert all(
        "worst_outer_test_balanced_accuracy" in item
        and "std_outer_test_balanced_accuracy" in item
        for item in result["candidate_summary"].values()
    )
    assert result["folds"][0]["test_balanced_accuracy"] == result["folds"][0]["test_candidate_metrics"][
        result["folds"][0]["selected_candidate"]
    ]["balanced_accuracy"]


def test_multi_market_walk_forward_and_bootstrap_summary():
    frame = pd.DataFrame({"x": np.arange(300, dtype=float),
                          "label": np.tile([0, 1], 150)})
    def predictor(train_x, train_y, eval_x):
        return np.zeros(len(eval_x), dtype=int)
    result = run_multi_market_walk_forward(
        {"BTC": frame, "EURUSD": frame}, ["x"], "label", predictor,
        min_train=100, validation_size=30, test_size=30, step_size=30, purge_bars=5)
    assert result["market_count"] == 2
    ci = bootstrap_fold_confidence_interval([0.4, 0.5, 0.6], repetitions=200)
    assert ci["lower"] <= ci["mean"] <= ci["upper"]


def test_budget_screen_estimates_model_fit_calls_before_training():
    from eti.pipeline import estimate_nested_model_fits_from_rows

    # 1,964 usable rows produce seven folds at these screening settings.
    assert estimate_nested_model_fits_from_rows(
        1964, horizon=6, min_train=400,
        validation_size=100, test_size=100, step_size=200,
    ) == 28
    assert estimate_nested_model_fits_from_rows(
        1964, horizon=6, min_train=400,
        validation_size=100, test_size=100, step_size=200,
        candidate_count=1,
    ) == 14
