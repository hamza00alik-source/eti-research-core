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
        "temperature": 2.0,
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
