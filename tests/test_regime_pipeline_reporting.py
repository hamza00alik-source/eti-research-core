import numpy as np
import pandas as pd

import eti.pipeline as pipeline


def test_nested_report_includes_causal_regime_counts_for_each_test_fold(monkeypatch):
    rng = np.random.default_rng(321)
    n = 300
    returns = rng.normal(0.0002, 0.012, n)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
        "open": open_,
        "high": np.maximum(open_, close) * 1.003,
        "low": np.minimum(open_, close) * 0.997,
        "close": close,
        "volume": rng.uniform(10, 1000, n),
    })

    class TinyModel:
        def fit(self, x, y):
            self.label = int(y.mode().iloc[0])
            return self

        def predict(self, x):
            return np.full(len(x), self.label, dtype=int)

    monkeypatch.setattr(pipeline, "BaselineModel", TinyModel)
    report = pipeline.run_nested_experiment(
        frame, horizon=3, min_train=80, validation_size=25,
        test_size=25, step_size=25,
    )

    folds = report["trading_performance"]["folds"]
    assert folds
    for fold in folds:
        regimes = fold["market_regimes"]
        assert regimes["bar_count"] == 25
        assert sum(regimes["counts"].values()) == 25
        assert set(regimes["counts"]).issubset(
            {"unknown", "high_volatility", "uptrend", "downtrend", "range"}
        )
        assert abs(sum(regimes["fractions"].values()) - 1.0) < 1e-9
