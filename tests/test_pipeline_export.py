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

    monkeypatch.setattr(BaselineModel, "fit", fake_fit)
    model, schema = fit_deployable_model(frame, horizon=6)
    assert model.classes_.tolist() == [0, 1]
    assert seen["rows"] == 254
    assert seen["columns"] == FEATURE_COLUMNS
    assert schema["features"] == FEATURE_COLUMNS
    assert schema["training_rows"] == 254
    assert schema["label_horizon_bars"] == 6


def test_exported_model_is_explicitly_marked_uncalibrated(monkeypatch):
    schema = {"features": list(FEATURE_COLUMNS)}
    monkeypatch.setattr(
        "eti.pipeline.fit_deployable_model",
        lambda frame, horizon: (object(), schema),
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


def test_training_cli_does_not_export_models_by_default(tmp_path, monkeypatch):
    import json
    import sys
    import eti.pipeline as pipeline

    output = tmp_path / "metrics.json"
    monkeypatch.setattr(sys, "argv", ["eti.pipeline", "--output", str(output)])
    monkeypatch.setattr(
        pipeline, "fetch_ohlcv",
        lambda *args: pd.DataFrame({"close": [100.0, 101.0]}),
    )
    monkeypatch.setattr(
        pipeline, "run_experiment",
        lambda *args, **kwargs: {"test": {"balanced_accuracy": 0.4}},
    )
    monkeypatch.setattr(
        pipeline, "export_deployable_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("export must be opt-in")),
    )

    pipeline.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert "exported_model" not in report


def test_training_cli_exports_only_when_explicit_flag_is_set(tmp_path, monkeypatch):
    import json
    import sys
    import eti.pipeline as pipeline

    output = tmp_path / "metrics-export.json"
    calls = {}
    monkeypatch.setattr(
        sys, "argv",
        ["eti.pipeline", "--output", str(output), "--save-model",
         "--model-id", "eti-test", "--model-version", "0.2.0"],
    )
    monkeypatch.setattr(
        pipeline, "fetch_ohlcv",
        lambda *args: pd.DataFrame({"close": [100.0, 101.0]}),
    )
    monkeypatch.setattr(
        pipeline, "run_experiment",
        lambda *args, **kwargs: {"test": {"balanced_accuracy": 0.4}},
    )

    def fake_export(frame, model_id, version, training_data_ref):
        calls.update(model_id=model_id, version=version, training_data_ref=training_data_ref)
        return {"artifact_uri": "artifact://sha256/" + "d" * 64,
                "calibration_status": "uncalibrated"}

    monkeypatch.setattr(pipeline, "export_deployable_model", fake_export)
    pipeline.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert calls == {
        "model_id": "eti-test",
        "version": "0.2.0",
        "training_data_ref": "ccxt:binance:BTC/USDT:1h:3000",
    }
    assert report["exported_model"]["calibration_status"] == "uncalibrated"
    assert "candidate artifact only" in report["warning"]
