import numpy as np
import pandas as pd
import pytest

from eti.model import BaselineModel


class FixedEstimator:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, frame):
        return np.tile(np.array([[0.1, 0.8, 0.1]]), (len(frame), 1))


def test_temperature_calibration_is_applied_to_model_probabilities():
    model = BaselineModel()
    model.estimator = FixedEstimator()
    model.classes_ = np.array([0, 1, 2])
    model.calibration_status = "calibrated"
    model.calibration_report = {
        "passed": True,
        "method": "temperature_scaling",
        "sample_count": 50,
        "class_counts": {"0": 16, "1": 17, "2": 17},
        "temperature": 2.0,
        "nll_before": 1.2,
        "nll_after": 1.0,
    }
    probabilities = model.predict_proba(pd.DataFrame({"x": [1.0]}))[0]
    expected = np.power(np.array([0.1, 0.8, 0.1]), 0.5)
    expected /= expected.sum()
    np.testing.assert_allclose(probabilities, expected)
    assert model.has_valid_calibration()


def test_calibrated_status_without_valid_report_fails_closed():
    model = BaselineModel()
    model.estimator = FixedEstimator()
    model.classes_ = np.array([0, 1, 2])
    model.calibration_status = "calibrated"
    model.calibration_report = {"passed": True, "method": "temperature_scaling",
                                "sample_count": 2, "temperature": 1.0}
    assert not model.has_valid_calibration()
    with pytest.raises(RuntimeError, match="calibration report is invalid"):
        model.predict_proba(pd.DataFrame({"x": [1.0]}))


def test_uncalibrated_baseline_keeps_raw_probabilities():
    model = BaselineModel()
    model.estimator = FixedEstimator()
    model.classes_ = np.array([0, 1, 2])
    probabilities = model.predict_proba(pd.DataFrame({"x": [1.0]}))[0]
    np.testing.assert_allclose(probabilities, [0.1, 0.8, 0.1])


def test_refitting_model_resets_previous_calibration_state():
    class RefitEstimator:
        def fit(self, features, labels):
            self.classes_ = np.array([0, 1, 2])
            return self

    model = BaselineModel()
    model.estimator = RefitEstimator()
    model.calibration_status = "calibrated"
    model.calibration_report = {
        "passed": True, "method": "temperature_scaling",
        "sample_count": 50, "class_counts": {"0": 16, "1": 17, "2": 17},
        "temperature": 1.5,
        "nll_before": 1.2, "nll_after": 1.0,
    }

    model.fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), np.array([0, 1, 2]))

    assert model.calibration_status == "uncalibrated"
    assert model.calibration_report is None
    assert not model.has_valid_calibration()


def test_temperature_scaling_fits_only_on_separate_calibration_labels():
    model = BaselineModel()
    model.estimator = FixedEstimator()
    model.classes_ = np.array([0, 1, 2])
    labels = np.tile(np.array([0, 1, 2]), 20)

    report = model.fit_temperature_calibration(
        pd.DataFrame({"x": np.arange(len(labels), dtype=float)}), labels
    )

    assert report["passed"] is True
    assert report["method"] == "temperature_scaling"
    assert report["sample_count"] == 60
    assert report["temperature"] > 0
    assert report["nll_after"] < report["nll_before"]
    assert model.has_valid_calibration()
    probabilities = model.predict_proba(pd.DataFrame({"x": [1.0]}))[0]
    assert np.isclose(probabilities.sum(), 1.0)
    assert not np.allclose(probabilities, [0.1, 0.8, 0.1])


def test_temperature_calibration_fails_closed_for_small_or_missing_class_samples():
    model = BaselineModel()
    model.estimator = FixedEstimator()
    model.classes_ = np.array([0, 1, 2])

    too_small = model.fit_temperature_calibration(
        pd.DataFrame({"x": np.arange(18, dtype=float)}),
        np.tile(np.array([0, 1, 2]), 6),
    )
    assert too_small["passed"] is False
    assert model.calibration_status == "uncalibrated"

    missing_class = model.fit_temperature_calibration(
        pd.DataFrame({"x": np.arange(60, dtype=float)}),
        np.tile(np.array([0, 1]), 30),
    )
    assert missing_class["passed"] is False
    assert missing_class["reason"] == "each_class_requires_at_least_five_calibration_samples"
    assert not model.has_valid_calibration()


def test_model_missing_training_class_cannot_be_calibrated():
    class TwoClassEstimator:
        classes_ = np.array([0, 1])

        def predict_proba(self, frame):
            return np.tile(np.array([[0.3, 0.7]]), (len(frame), 1))

    model = BaselineModel()
    model.estimator = TwoClassEstimator()
    model.classes_ = np.array([0, 1])
    labels = np.tile(np.array([0, 1, 2]), 20)
    report = model.fit_temperature_calibration(
        pd.DataFrame({"x": np.arange(len(labels), dtype=float)}), labels
    )
    assert report["passed"] is False
    assert report["reason"] == "training_model_missing_one_or_more_classes"
    assert model.calibration_status == "uncalibrated"


@pytest.mark.parametrize("labels", [
    np.array([0, 1, 3]),
    np.array([0.0, 1.5, 2.0]),
    np.array([0.0, 1.0, np.nan]),
])
def test_training_rejects_invalid_class_ids(labels):
    model = BaselineModel()
    with pytest.raises(ValueError):
        model.fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), labels)
