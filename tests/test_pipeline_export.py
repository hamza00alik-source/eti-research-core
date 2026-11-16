import numpy as np
import pandas as pd

from eti.model import BaselineModel
from eti.features import FEATURE_COLUMNS
from eti.pipeline import export_deployable_model, fit_deployable_model, _reserve_final_export_holdout


def test_deployable_fit_uses_exact_feature_schema_and_only_completed_rows(monkeypatch):
    index = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    frame = pd.DataFrame({"close": np.linspace(100, 130, len(index))}, index=index)
    feature_frame = pd.DataFrame(
        {name: np.arange(len(index), dtype=float) / (position + 1)
         for position, name in enumerate(FEATURE_COLUMNS)},
        index=index,
    )
    labels = pd.Series(np.tile([0, 1, 2], (len(index) + 2) // 3)[:len(index)], index=index)
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
    def fake_raw_probabilities(self, x):
        return np.tile([0.5, 0.5, 0.0], (len(x), 1))

    monkeypatch.setattr(BaselineModel, "predict_proba_raw", fake_raw_probabilities)
    monkeypatch.setattr(BaselineModel, "predict_proba", fake_raw_probabilities)
    monkeypatch.setattr(BaselineModel, "fit_temperature_calibration", fake_calibration)
    model, schema = fit_deployable_model(frame, horizon=6)
    assert model.classes_.tolist() == [0, 1, 2]
    assert seen["rows"] == 230
    assert seen["columns"] == FEATURE_COLUMNS
    assert seen["calibration_rows"] == 73
    assert seen["calibration_columns"] == FEATURE_COLUMNS
    assert schema["features"] == FEATURE_COLUMNS
    assert len(schema["dataset_sha256"]) == 64
    assert schema["training_rows"] == 230
    assert schema["calibration_rows"] == 73
    assert schema["independent_validation_rows"] == 79
    assert schema["purged_rows"] == 12
    assert schema["calibration_status"] == "calibrated"
    assert schema["training_end"] < schema["calibration_start"]
    assert schema["calibration_end"] < schema["independent_validation_start"]
    assert schema["independent_probability_validation"]["passed"] is True
    assert schema["independent_probability_validation"]["reserved_from_prior_research_evaluation"] is False
    assert schema["calibration_report"]["independent_validation"]["calibration_applied_to_export"] is True
    assert schema["label_horizon_bars"] == 6


def test_exported_model_is_explicitly_marked_uncalibrated(monkeypatch):
    schema = {"features": list(FEATURE_COLUMNS), "dataset_sha256": "a" * 64,
              "feature_version": "eti.features.v1",
              "independent_validation_end": "2025-01-10T00:00:00+00:00"}
    class UncalibratedStub:
        calibration_status = "uncalibrated"
        calibration_report = None

    monkeypatch.setattr(
        "eti.pipeline.fit_deployable_model",
        lambda frame, horizon, **kwargs: (UncalibratedStub(), schema),
    )
    monkeypatch.setattr(
        "eti.pipeline.ModelArtifactStore.save_model",
        lambda self, model: {
            "artifact_uri": "artifact://sha256/" + "c" * 64,
            "artifact_sha256": "c" * 64,
        },
    )
    exported = export_deployable_model(
        pd.DataFrame(), "eti-test", "0.1.0", "dataset:test", timeframe="1h"
    )
    assert exported["calibration_status"] == "uncalibrated"
    assert exported["supported_classes"] == ["crypto_spot"]
    assert exported["base_version"] == "0.1.0"
    assert exported["version"] == "0.1.0+" + "c" * 12
    assert exported["department"] == "prediction"
    assert exported["model_role"] == "direction_classifier"
    assert exported["input_schema"]["department"] == "prediction"
    assert exported["input_schema"]["market_types"] == ["spot"]
    assert exported["input_schema"]["timeframes"] == ["1h"]
    assert exported["input_schema"]["experiment_id"].startswith("eti-exp-")
    assert len(exported["input_schema"]["lineage"]["code_sha256"]) == 64
    assert len(exported["input_schema"]["lineage"]["config_sha256"]) == 64

def test_training_cli_does_not_export_models_by_default(tmp_path, monkeypatch):
    import json
    import sys
    import eti.pipeline as pipeline

    output = tmp_path / "metrics.json"
    monkeypatch.setattr(sys, "argv", ["eti.pipeline", "--output", str(output)])
    sample = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"),
        "open": np.linspace(100, 101, 100), "high": np.linspace(101, 102, 100),
        "low": np.linspace(99, 100, 100), "close": np.linspace(100, 101, 100),
        "volume": np.full(100, 10.0),
    })
    monkeypatch.setattr(
        pipeline, "fetch_ohlcv_auto",
        lambda *args, **kwargs: (sample.copy(), {
            "provider": "coinbaseexchange", "venue_symbol": "BTC/USD",
            "requested_symbol": "BTC/USDT", "mode": "automatic_fallback",
        }),
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
    sample = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"),
        "open": np.linspace(100, 101, 100), "high": np.linspace(101, 102, 100),
        "low": np.linspace(99, 100, 100), "close": np.linspace(100, 101, 100),
        "volume": np.full(100, 10.0),
    })
    monkeypatch.setattr(
        pipeline, "fetch_ohlcv_auto",
        lambda *args, **kwargs: (sample.copy(), {
            "provider": "coinbaseexchange", "venue_symbol": "BTC/USD",
            "requested_symbol": "BTC/USDT", "mode": "automatic_fallback",
        }),
    )
    monkeypatch.setattr(
        pipeline, "run_experiment",
        lambda *args, **kwargs: {"test": {"balanced_accuracy": 0.4}},
    )

    def fake_export(frame, model_id, version, training_data_ref, horizon=6, *,
                    holdout_reserved_from_prior_evaluation=False):
        calls.update(
            model_id=model_id, version=version, training_data_ref=training_data_ref,
            horizon=horizon,
            holdout_reserved_from_prior_evaluation=holdout_reserved_from_prior_evaluation,
        )
        return {"artifact_uri": "artifact://sha256/" + "d" * 64,
                "calibration_status": "uncalibrated"}

    monkeypatch.setattr(pipeline, "export_deployable_model", fake_export)
    pipeline.main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert calls == {
        "model_id": "eti-test",
        "version": "0.2.0",
        "training_data_ref": "ccxt:coinbaseexchange:BTC/USD:1h:100",
        "horizon": 6,
        "holdout_reserved_from_prior_evaluation": True,
    }
    assert report["exported_model"]["calibration_status"] == "uncalibrated"
    assert "candidate artifact only" in report["warning"]



def test_calibration_is_rejected_when_independent_holdout_nll_worsens(monkeypatch):
    index = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    frame = pd.DataFrame({"close": np.linspace(100, 130, len(index))}, index=index)
    feature_frame = pd.DataFrame(
        {name: np.arange(len(index), dtype=float) / (position + 1)
         for position, name in enumerate(FEATURE_COLUMNS)},
        index=index,
    )
    labels = pd.Series(np.tile([0, 1, 2], (len(index) + 2) // 3)[:len(index)], index=index)
    labels.iloc[-6:] = np.nan
    monkeypatch.setattr("eti.pipeline.validate_ohlcv", lambda data: data)
    monkeypatch.setattr("eti.pipeline.build_features", lambda data: feature_frame)
    monkeypatch.setattr("eti.pipeline.make_labels", lambda data, horizon: labels)

    def fake_fit(self, x, y):
        self.classes_ = np.array([0, 1, 2])
        return self

    def fake_calibration(self, x, y):
        self.calibration_status = "calibrated"
        self.calibration_report = {
            "passed": True, "method": "temperature_scaling",
            "sample_count": len(y), "temperature": 1.2,
        }
        return self.calibration_report

    monkeypatch.setattr(BaselineModel, "fit", fake_fit)
    monkeypatch.setattr(BaselineModel, "fit_temperature_calibration", fake_calibration)
    monkeypatch.setattr(
        BaselineModel, "predict_proba_raw",
        lambda self, x: np.tile([0.5, 0.5, 0.0], (len(x), 1)),
    )
    monkeypatch.setattr(
        BaselineModel, "predict_proba",
        lambda self, x: np.tile([0.2, 0.8, 0.0], (len(x), 1)),
    )
    model, schema = fit_deployable_model(frame, horizon=6)
    assert model.calibration_status == "uncalibrated"
    assert schema["calibration_status"] == "uncalibrated"
    assert schema["calibration_report"]["passed"] is False
    assert schema["calibration_report"]["reason"] == "temperature_scaling_failed_independent_holdout_nll"
    assert schema["independent_probability_validation"]["passed"] is False
    assert schema["independent_probability_validation"]["calibration_applied_to_export"] is False



def test_final_export_holdout_is_excluded_from_prior_research_evaluation():
    frame = pd.DataFrame({"close": np.arange(1000, dtype=float)})
    evaluation, protocol = _reserve_final_export_holdout(frame, save_model=True)
    assert len(evaluation) == 800
    assert len(frame) - len(evaluation) == 200
    assert evaluation.index.max() < frame.index.max()
    assert protocol["enabled"] is True
    assert protocol["reserved_rows"] == 200
    assert protocol["reserved_fraction"] == 0.2


def test_research_evaluation_uses_full_data_when_not_exporting_model():
    frame = pd.DataFrame({"close": np.arange(1000, dtype=float)})
    evaluation, protocol = _reserve_final_export_holdout(frame, save_model=False)
    assert len(evaluation) == len(frame)
    assert protocol["enabled"] is False
    assert protocol["reserved_rows"] == 0
