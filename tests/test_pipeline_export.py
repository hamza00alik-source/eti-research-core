import numpy as np
import pandas as pd

from eti.model import BaselineModel
from eti.features import FEATURE_COLUMNS
from eti.pipeline import export_deployable_model, fit_deployable_model


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

    def fake_calibration(self, x, y):
        seen["calibration_rows"] = len(x)
        seen["calibration_columns"] = list(x.columns)
        report = {"passed": True, "method": "temperature_scaling",
                  "sample_count": len(y), "temperature": 1.2}
        self.calibration_status = "calibrated"
        self.calibration_report = report
        return report

    monkeypatch.setattr(BaselineModel, "fit", fake_fit)
    monkeypatch.setattr(BaselineModel, "fit_temperature_calibration", fake_calibration)
    model, schema = fit_deployable_model(frame, horizon=6)
    assert model.classes_.tolist() == [0, 1]
    assert seen["rows"] == 203
    assert seen["columns"] == FEATURE_COLUMNS
    assert seen["calibration_rows"] == 51
    assert seen["calibration_columns"] == FEATURE_COLUMNS
    assert schema["features"] == FEATURE_COLUMNS
    assert schema["training_rows"] == 203
    assert schema["calibration_rows"] == 51
    assert schema["calibration_status"] == "calibrated"
    assert schema["label_horizon_bars"] == 6


def test_exported_model_is_explicitly_marked_uncalibrated(monkeypatch):
    schema = {"features": list(FEATURE_COLUMNS)}
    class UncalibratedStub:
        calibration_status = "uncalibrated"
        calibration_report = None

    monkeypatch.setattr(
        "eti.pipeline.fit_deployable_model",
        lambda frame, horizon: (UncalibratedStub(), schema),
    )
    monkeypatch.setattr(
        "eti.pipeline.ModelArtifactStore.save_model",
        lambda self, model: {
            "artifact_uri": "artifact://sha256/" + "c" * 64,
            "artifact_sha256": "c" * 64,
        },
    )
    exported = export_deployable_model(
        pd.DataFrame(), "eti-test", "0.1.0", "dataset:test"
    )
    assert exported["calibration_status"] == "uncalibrated"
    assert exported["supported_classes"] == ["crypto_spot"]
