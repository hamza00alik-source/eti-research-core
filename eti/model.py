"""Small baseline classifier with stable class-probability alignment."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

CLASSES = np.array([0, 1, 2])


class BaselineModel:
    def __init__(self, random_state: int = 42):
        self.estimator = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=random_state,
        )
        self.classes_: np.ndarray | None = None

    def fit(self, x, y):
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("Training data contains fewer than two label classes")
        self.estimator.fit(x, y)
        self.classes_ = self.estimator.classes_
        return self

    def calibration_temperature(self) -> float | None:
        if self.calibration_status != "calibrated":
            return None
        report = self.calibration_report
        if not isinstance(report, dict):
            raise RuntimeError("Calibrated model is missing its calibration report")
        temperature = report.get("temperature")
        sample_count = report.get("sample_count")
        if (
            report.get("passed") is not True
            or report.get("method") != "temperature_scaling"
            or isinstance(sample_count, bool) or not isinstance(sample_count, int)
            or sample_count < 30
            or isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not np.isfinite(temperature) or temperature <= 0
        ):
            raise RuntimeError("Model calibration report is invalid")
        return float(temperature)

    def has_valid_calibration(self) -> bool:
        try:
            return self.calibration_temperature() is not None
        except RuntimeError:
            return False

    def predict_proba(self, x) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Model has not been fitted")
        raw = self.estimator.predict_proba(x)
        aligned = np.zeros((len(raw), len(CLASSES)), dtype=float)
        for i, cls in enumerate(self.classes_):
            aligned[:, int(cls)] = raw[:, i]
        temperature = self.calibration_temperature()
        if temperature is not None:
            logits = np.log(np.clip(aligned, 1e-12, 1.0)) / temperature
            logits -= logits.max(axis=1, keepdims=True)
            scaled = np.exp(logits)
            aligned = scaled / scaled.sum(axis=1, keepdims=True)
        return aligned

    def predict(self, x) -> np.ndarray:
        return CLASSES[self.predict_proba(x).argmax(axis=1)]
