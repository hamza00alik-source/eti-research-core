import numpy as np
import pandas as pd
import pytest

from eti.market_intelligence_training import train_market_intelligence_specialists


def make_frame(n=1100, seed=23):
    rng = np.random.default_rng(seed)
    # Alternating volatility blocks provide more than one future regime.
    scales = np.where((np.arange(n) // 140) % 2 == 0, 0.002, 0.012)
    returns = rng.normal(0.0001, scales)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.001, 0.0004, n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC"),
        "open": open_,
        "high": np.maximum(open_, close) * (1 + spread),
        "low": np.minimum(open_, close) * (1 - spread),
        "close": close,
        "volume": rng.uniform(10, 1000, n),
    })


def test_market_intelligence_pipeline_reports_two_separate_research_roles():
    result = train_market_intelligence_specialists(
        make_frame(), horizon=4, min_train=300, validation_size=60,
        test_size=60, step_size=100, max_model_fits=32,
    )
    assert result["department"] == "market_intelligence"
    assert set(result["roles"]) == {"regime_classifier", "volatility_estimator"}
    assert result["roles"]["regime_classifier"]["model_role"] == "regime_classifier"
    assert result["roles"]["volatility_estimator"]["model_role"] == "volatility_estimator"
    assert result["compute_budget"]["estimated_model_fit_calls"] == result["fold_count"] * 4
    assert result["promotion"]["eligible"] is False
    assert result["execution_enabled"] is False
    assert result["lineage"]["experiment_id"].startswith("eti-exp-")
    assert len(result["lineage"]["dataset_sha256"]) == 64
    assert all("test_start" in fold for fold in result["roles"]["regime_classifier"]["outer_test_folds"])


def test_fit_budget_is_checked_before_any_model_fit(monkeypatch):
    import eti.market_intelligence_training as training

    def fail_if_called(*args, **kwargs):
        raise AssertionError("model fit must not run when budget is exceeded")

    monkeypatch.setattr(training.HistGradientBoostingClassifier, "fit", fail_if_called)
    monkeypatch.setattr(training.HistGradientBoostingRegressor, "fit", fail_if_called)
    with pytest.raises(ValueError, match="stopped before fitting"):
        training.train_market_intelligence_specialists(
            make_frame(), horizon=4, min_train=300, validation_size=60,
            test_size=60, step_size=100, max_model_fits=1,
        )


def test_future_observations_do_not_change_features_or_labels_before_cutoff():
    import eti.market_intelligence_training as training

    original, _ = training._prepare_dataset(make_frame(), 4)
    changed_frame = make_frame()
    changed_frame.loc[900:, ["open", "high", "low", "close", "volume"]] *= 1.5
    changed, _ = training._prepare_dataset(changed_frame, 4)
    original_prefix = original[original["timestamp"] < pd.Timestamp("2022-02-05", tz="UTC")]
    changed_prefix = changed[changed["timestamp"] < pd.Timestamp("2022-02-05", tz="UTC")]
    pd.testing.assert_frame_equal(original_prefix.reset_index(drop=True), changed_prefix.reset_index(drop=True))


def test_invalid_time_order_and_invalid_budget_fail_closed():
    frame = make_frame()
    frame.loc[10, "timestamp"] = frame.loc[9, "timestamp"]
    with pytest.raises(ValueError, match="unique and chronological"):
        train_market_intelligence_specialists(frame, min_train=300, validation_size=60, test_size=60)
    with pytest.raises(ValueError, match="max_model_fits"):
        train_market_intelligence_specialists(make_frame(), max_model_fits=0)


def test_single_class_outer_test_reports_undefined_balanced_accuracy_without_warning():
    from eti.market_intelligence_training import _classification_metrics

    metrics = _classification_metrics(pd.Series(["range"] * 8), np.array(["range"] * 7 + ["uptrend"]))
    assert metrics["accuracy"] == pytest.approx(7 / 8)
    assert metrics["balanced_accuracy"] is None
    assert metrics["balanced_accuracy_status"] == "insufficient_test_class_coverage"
    assert metrics["observed_test_classes"] == ["range"]
    assert metrics["test_class_count"] == 1
    assert metrics["per_class"]["range"]["support"] == 8
    assert metrics["per_class"]["uptrend"]["support"] == 0
    assert metrics["per_class"]["uptrend"]["recall"] is None


def test_multiclass_outer_test_balanced_accuracy_uses_observed_classes_only():
    from eti.market_intelligence_training import _classification_metrics

    truth = pd.Series(["range", "range", "uptrend", "uptrend", "downtrend"])
    predicted = np.array(["range", "uptrend", "uptrend", "uptrend", "range"])
    metrics = _classification_metrics(truth, predicted)
    assert metrics["balanced_accuracy"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert metrics["balanced_accuracy_status"] == "ok"
    assert metrics["observed_test_classes"] == ["downtrend", "range", "uptrend"]
    assert metrics["test_class_count"] == 3
    assert metrics["class_coverage"] == pytest.approx(0.75)
