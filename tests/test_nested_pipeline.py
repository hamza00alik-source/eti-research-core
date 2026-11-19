import numpy as np
import pandas as pd
import pytest

import eti.pipeline as pipeline


def test_nested_training_pipeline_orchestrates_market_folds(monkeypatch):
    rng = np.random.default_rng(123)
    n = 260
    returns = rng.normal(0, 0.015, n)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "open": open_,
        "high": np.maximum(open_, close) * 1.002,
        "low": np.minimum(open_, close) * 0.998,
        "close": close,
        "volume": rng.uniform(1, 100, n),
    })

    class TinyModel:
        def fit(self, x, y):
            self.label = int(y.mode().iloc[0])
            return self
        def predict(self, x):
            return np.full(len(x), self.label, dtype=int)

    monkeypatch.setattr(pipeline, "BaselineModel", TinyModel)
    result = pipeline.run_nested_experiment(
        frame, horizon=3, min_train=80, validation_size=25,
        test_size=25, step_size=25)
    assert result["training_performed"] is True
    assert result["execution_enabled"] is False
    assert result["fold_count"] > 0
    assert result["annualization"]["method"] == "observed_usable_bars_per_calendar_year"
    assert result["annualization"]["bars_per_year"] == pytest.approx(365.25 * 24, rel=0.02)
    assert "hist_gradient_boosting" in result["folds"][0]["validation_scores"]
    assert "candidate_summary" in result
    assert result["compute_budget"]["estimated_model_fit_calls"] == result["fold_count"] * 2
    assert result["probability_calibration"]["method"] == "fold_local_temperature_scaling"
    assert result["folds"][0]["probability_calibration"]["calibration_fit"]["status"] == "insufficient_calibration_window"
    assert "hist_gradient_boosting" in result["candidate_summary"]
    # The exact frozen outer-test predictions must yield ledger-ready,
    # candidate-specific net return streams (not only selected-model metrics).
    assert set(result["candidate_oos_returns"]) == set(pipeline.NESTED_CANDIDATE_NAMES)
    for candidate_name, stream in result["candidate_oos_returns"].items():
        assert stream["benchmark_id"] == "no_trade"
        assert len(stream["timestamps"]) == len(stream["net_return_differentials"])
        assert len(stream["timestamps"]) >= 8
        assert len(stream["timestamps"]) == len(set(stream["timestamps"]))
        assert stream["ledger_compatible"] is True
    lineage = result["trial_lineage"]
    assert len(lineage["training_data_sha256"]) == 64
    assert len(lineage["evaluation_data_sha256"]) == 64
    assert lineage["training_row_count"] > 0
    assert lineage["evaluation_row_count"] == sum(
        len(fold["timestamps"]) for fold in result["trading_performance"]["folds"]
    )
    # Independently reproduce the runner's preflight lineage calculation.
    # These hashes must match the pipeline report exactly before a trial is
    # registered, otherwise the ledger could bind returns to different rows.
    from eti.features import build_features, feature_columns_for_frame, make_labels
    from eti.evaluation import walk_forward_folds
    usable = build_features(frame).join(
        make_labels(frame, horizon=3).rename("label")
    ).dropna(subset=feature_columns_for_frame(frame) + ["label"])
    usable["close"] = frame.loc[usable.index, "close"].astype(float)
    usable["timestamp"] = frame.loc[usable.index, "timestamp"].to_numpy()
    folds = walk_forward_folds(
        len(usable), min_train=80, validation_size=25,
        test_size=25, step_size=25, purge_bars=3,
    )
    train_rows = sorted({i for fold in folds for i in range(fold.train_end)})
    test_rows = sorted({
        i for fold in folds for i in range(fold.test_start, fold.test_end)
    })
    import hashlib
    def digest(indices):
        payload = pd.util.hash_pandas_object(usable.iloc[indices], index=True).to_numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()
    assert lineage["training_data_sha256"] == digest(train_rows)
    assert lineage["evaluation_data_sha256"] == digest(test_rows)
    scenarios = result["cost_sensitivity"]["scenarios"]
    assert [item["cost_multiplier"] for item in scenarios] == [1.0, 1.5, 2.0]
    assert all(
        "mean_net_return" in item
        and "positive_return_folds" in item
        and "mean_max_drawdown" in item
        for item in scenarios
    )
    outer_fold = result["trading_performance"]["folds"][0]
    model_stream = outer_fold["trading"]["model"]
    assert len(outer_fold["timestamps"]) == model_stream["bars"]
    assert len(model_stream["bar_returns"]) == model_stream["bars"]
    assert len(model_stream["positions"]) == model_stream["bars"]



def test_observed_annualization_accounts_for_weekends_and_sessions():
    timestamps = []
    for day in pd.bdate_range("2024-01-01", periods=30):
        for hour in range(10, 16):
            timestamps.append(day + pd.Timedelta(hours=hour))
    frame = pd.DataFrame({"timestamp": pd.to_datetime(timestamps, utc=True)})
    dataset = pd.DataFrame(index=range(len(frame)))

    bars_per_year, details = pipeline._estimate_observed_bars_per_year(frame, dataset)

    continuous_hourly = 365.25 * 24
    assert 1000 < bars_per_year < 2200
    assert bars_per_year < continuous_hourly * 0.25
    assert details["usable_rows"] == len(dataset)
    assert details["warning"].startswith("Sample-rate estimate")


def test_observed_annualization_rejects_invalid_timestamp_span():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"])
    })
    dataset = pd.DataFrame(index=[0, 1])
    with pytest.raises(ValueError, match="positive duration"):
        pipeline._estimate_observed_bars_per_year(frame, dataset)
