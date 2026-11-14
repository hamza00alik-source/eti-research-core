import numpy as np
import pandas as pd

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
    assert result["fold_count"] > 0
    assert "hist_gradient_boosting" in result["folds"][0]["validation_scores"]
    assert "candidate_summary" in result
    assert "hist_gradient_boosting" in result["candidate_summary"]
    scenarios = result["cost_sensitivity"]["scenarios"]
    assert [item["cost_multiplier"] for item in scenarios] == [1.0, 1.5, 2.0]
    assert all(
        "mean_net_return" in item
        and "positive_return_folds" in item
        and "mean_max_drawdown" in item
        for item in scenarios
    )
