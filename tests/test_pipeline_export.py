import numpy as np
import pandas as pd

from eti.model import BaselineModel
from eti.features import FEATURE_COLUMNS
from eti.pipeline import fit_deployable_model


def test_deployable_fit_uses_exact_feature_schema_and_only_completed_rows(monkeypatch):
    index = pd.date_range("2025-01-01", periods=260, freq="h", tz="UTC")
    frame = pd.DataFrame({"close": np.linspace(100, 130, len(index))}, index=index)
    feature_frame = pd.DataFrame(
        {name: np.arange(len(index), dtype=float) / (position + 1)
         for position, name in enumerate(FEATURE_COLUMNS)},
        index=index,
    )
    labels = pd.Series(np.tile([0, 1], len(index) // 2), index=index)
    labels.iloc[-6:] = np.nan
    seen = {}

    monkeypatch.setattr("eti.pipeline.validate_ohlcv", lambda data: data)
    monkeypatch.setattr("eti.pipeline.build_features", lambda data: feature_frame)
    monkeypatch.setattr("eti.pipeline.make_labels", lambda data, horizon: labels)

    def fake_fit(self, x, y):
        seen["rows"] = len(x)
        seen["columns"] = list(x.columns)
        seen["classes"] = sorted(set(y.tolist()))
        self.classes_ = np.array(seen["classes"])
        return self

    monkeypatch.setattr(BaselineModel, "fit", fake_fit)
    model, schema = fit_deployable_model(frame, horizon=6)
    assert model.classes_.tolist() == [0, 1]
    assert seen["rows"] == 254
    assert seen["columns"] == FEATURE_COLUMNS
    assert schema["features"] == FEATURE_COLUMNS
    assert schema["training_rows"] == 254
    assert schema["label_horizon_bars"] == 6
    assert schema["training_rows"] == 254
