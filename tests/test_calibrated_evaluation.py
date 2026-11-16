import numpy as np
import pandas as pd

from eti.calibrated_evaluation import run_nested_calibrated_walk_forward


class RecordingCalibratedModel:
    instances = []

    def __init__(self):
        self.fit_rows = None
        self.calibration_rows = None
        self.calibration_called = False
        self.instances.append(self)

    def fit(self, x, y):
        self.fit_rows = list(x.index)
        self.label = int(y.mode().iloc[0])
        return self

    def predict(self, x):
        return np.zeros(len(x), dtype=int)

    def predict_proba_raw(self, x):
        return np.tile(np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=float), (len(x), 1))

    def predict_proba(self, x):
        return self.predict_proba_raw(x)

    def fit_temperature_calibration(self, x, y, min_samples=30):
        self.calibration_called = True
        self.calibration_rows = list(x.index)
        return {
            "passed": True,
            "method": "temperature_scaling",
            "sample_count": len(y),
            "temperature": 1.0,
        }


def sample_dataset(rows=500):
    labels = np.arange(rows) % 3
    return pd.DataFrame({
        "feature": np.linspace(-1.0, 1.0, rows),
        "label": labels,
    })


def test_fold_calibration_uses_disjoint_chronological_blocks_and_one_fit():
    RecordingCalibratedModel.instances = []
    result = run_nested_calibrated_walk_forward(
        sample_dataset(), ["feature"], "label",
        {"recording_model": RecordingCalibratedModel},
        min_train=100, validation_size=100, test_size=50,
        step_size=50, purge_bars=5,
    )

    assert result["fold_count"] > 0
    first = result["folds"][0]
    model = RecordingCalibratedModel.instances[0]
    boundaries = first["boundaries"]

    assert len(model.fit_rows) == boundaries["train_end"]
    assert max(model.fit_rows) < boundaries["selection_start"]
    assert model.calibration_called is True
    assert min(model.calibration_rows) == boundaries["calibration_start"]
    assert max(model.calibration_rows) < boundaries["calibration_end"]
    assert boundaries["selection_end"] + boundaries["purge_bars"] == boundaries["calibration_start"]
    assert boundaries["calibration_end"] + boundaries["purge_bars"] == boundaries["test_start"]
    assert first["probability_calibration"]["status"] == "calibrated_and_outer_test_confirmed"
    assert first["probability_calibration"]["raw"]["sample_count"] == 50
    assert first["probability_calibration"]["temperature_scaled"]["sample_count"] == 50
    assert result["probability_calibration"]["outer_test_confirmed_folds"] == result["fold_count"]
    assert len(RecordingCalibratedModel.instances) == result["fold_count"]


def test_small_validation_window_is_reported_uncalibrated_not_reused():
    RecordingCalibratedModel.instances = []
    result = run_nested_calibrated_walk_forward(
        sample_dataset(260), ["feature"], "label",
        {"recording_model": RecordingCalibratedModel},
        min_train=80, validation_size=25, test_size=25,
        step_size=25, purge_bars=3,
    )

    first = result["folds"][0]
    model = RecordingCalibratedModel.instances[0]
    assert model.calibration_called is False
    assert first["probability_calibration"]["status"] == "uncalibrated"
    assert first["probability_calibration"]["calibration_fit"]["status"] == "insufficient_calibration_window"
    assert first["boundaries"]["calibration_start"] is None
