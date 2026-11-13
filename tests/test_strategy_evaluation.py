import numpy as np
import pandas as pd
import pytest

from eti.strategy_evaluation import run_strategy_walk_forward


def sample_close(rows=600):
    rng = np.random.default_rng(17)
    returns = rng.normal(0.0002, 0.01, rows)
    return pd.Series(100 * np.exp(np.cumsum(returns)),
                     index=pd.date_range("2024-01-01", periods=rows, freq="h"))


def test_strategy_walk_forward_selects_on_validation_and_reports_test_only():
    report = run_strategy_walk_forward(
        sample_close(), min_history=100, validation_size=100,
        test_size=80, step_size=80, purge_bars=2,
    )
    assert report["method"] == "validation_selected_rule_based_walk_forward_v1"
    assert report["candidate_count"] == 6
    assert report["fold_count"] >= 2
    assert report["safety"]["is_trained_model"] is False
    assert report["safety"]["places_orders"] is False
    for fold in report["folds"]:
        assert fold["boundaries"]["validation_end"] <= fold["boundaries"]["test_start"] - 2
        assert fold["test"]["bars"] == 80
        assert len(fold["validation_candidates"]) == 6
        assert fold["selected_candidate"]["name"] in {"ma", "momentum"}


def test_strategy_walk_forward_rejects_bad_prices():
    close = sample_close()
    close.iloc[10] = np.nan
    with pytest.raises(ValueError, match="finite and positive"):
        run_strategy_walk_forward(close)


def test_strategy_walk_forward_rejects_invalid_costs():
    with pytest.raises(ValueError, match="fee_bps"):
        run_strategy_walk_forward(sample_close(), fee_bps=-1)
