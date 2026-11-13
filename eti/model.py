"""Small baseline classifier with stable class-probability alignment."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier

CLASSES = np.array([0, 1, 2])


class BaselineModel:
    def __init__(self, random_state: int = 42):
        self.estimator = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=random_state,
        )
        self.classes_: np.ndarray | None = None
        self.calibration_status = "uncalibrated"
        self.calibration_report: dict | None = None

    def fit(self, x, y):
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("Training data contains fewer than two label classes")
        self.estimator.fit(x, y)
        self.classes_ = self.estimator.classes_
        self.calibration_status = "uncalibrated"
        self.calibration_report = None
        return self

    def fit_temperature_calibration(self, x, y, min_samples: int = 30) -> dict:
        """Fit one scalar temperature on a separate chronological calibration set."""
        if self.classes_ is None:
            raise RuntimeError("Model must be fitted before calibration")
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 30:
            raise ValueError("min_samples must be an integer of at least 30")
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(x):
            raise ValueError("Calibration features and labels must have matching row counts")
        if not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all():
            raise ValueError("Calibration labels must be finite numeric class IDs")
        if not np.equal(labels, np.floor(labels)).all():
            raise ValueError("Calibration labels must be integer class IDs")
        labels = labels.astype(int)
        if not np.isin(labels, CLASSES).all():
            raise ValueError("Calibration labels must be in the short/flat/long class set")

        sample_count = int(len(labels))
        counts = {str(int(cls)): int(np.sum(labels == cls)) for cls in CLASSES}
        report = {
            "passed": False,
            "method": "temperature_scaling",
            "sample_count": sample_count,
            "class_counts": counts,
            "temperature": 1.0,
        }
        self.calibration_status = "uncalibrated"
        self.calibration_report = report

        if sample_count < min_samples:
            report["reason"] = "insufficient_calibration_samples"
            return report
        if any(counts[str(int(cls))] < 5 for cls in CLASSES):
            report["reason"] = "each_class_requires_at_least_five_calibration_samples"
            return report

        raw = self.estimator.predict_proba(x)
        aligned = np.zeros((len(raw), len(CLASSES)), dtype=float)
        for index, cls in enumerate(self.classes_):
            aligned[:, int(cls)] = raw[:, index]
        if (
            aligned.shape != (sample_count, len(CLASSES))
            or not np.isfinite(aligned).all()
            or (aligned < 0).any()
            or (aligned > 1).any()
            or (aligned.sum(axis=1) <= 0).any()
        ):
            report["reason"] = "invalid_raw_probabilities"
            return report
        aligned = aligned / aligned.sum(axis=1, keepdims=True)
        one_hot = np.eye(len(CLASSES), dtype=float)[labels]

        def probabilities(log_temperature: float) -> np.ndarray:
            logits = np.log(np.clip(aligned, 1e-12, 1.0)) / np.exp(log_temperature)
            logits -= logits.max(axis=1, keepdims=True)
            scaled = np.exp(logits)
            return scaled / scaled.sum(axis=1, keepdims=True)

        def nll(log_temperature: float) -> float:
            calibrated = probabilities(log_temperature)
            return float(-np.log(np.clip(
                calibrated[np.arange(sample_count), labels], 1e-12, 1.0
            )).mean())

        before = nll(0.0)
        result = minimize_scalar(
            nll, bounds=(float(np.log(0.05)), float(np.log(20.0))),
            method="bounded",
        )
        temperature = float(np.exp(result.x))
        after = nll(float(result.x))
        brier_before = float(np.mean(np.sum((aligned - one_hot) ** 2, axis=1)))
        calibrated = probabilities(float(result.x))
        brier_after = float(np.mean(np.sum((calibrated - one_hot) ** 2, axis=1)))
        passed = bool(
            result.success and np.isfinite(temperature) and temperature > 0
            and np.isfinite(after) and after <= before + 1e-9
        )
        report.update({
            "passed": passed,
            "temperature": temperature if passed else 1.0,
            "nll_before": before,
            "nll_after": after,
            "brier_before": brier_before,
            "brier_after": brier_after,
        })
        if passed:
            self.calibration_status = "calibrated"
        else:
            report["reason"] = "temperature_scaling_did_not_improve_validation_nll"
        return report

    def calibration_temperature(self) -> float | None:
        if self.calibration_status != "calibrated":
            return None
        report = self.calibration_report
        if not isinstance(report, dict):
            raise RuntimeError("Calibrated model is missing its calibration report")
        temperature = report.get("temperature")
        sample_count = report.get("sample_count")
        class_counts = report.get("class_counts")
        nll_before = report.get("nll_before")
        nll_after = report.get("nll_after")
        if (
            report.get("passed") is not True
            or report.get("method") != "temperature_scaling"
            or isinstance(sample_count, bool) or not isinstance(sample_count, int)
            or sample_count < 30
            or not isinstance(class_counts, dict)
            or any(
                isinstance(class_counts.get(str(int(cls))), bool)
                or not isinstance(class_counts.get(str(int(cls))), int)
                or class_counts.get(str(int(cls))) < 5
                for cls in CLASSES
            )
            or isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not np.isfinite(temperature) or temperature <= 0
            or isinstance(nll_before, bool) or not isinstance(nll_before, (int, float))
            or isinstance(nll_after, bool) or not isinstance(nll_after, (int, float))
            or not np.isfinite(nll_before) or not np.isfinite(nll_after)
            or nll_after > nll_before + 1e-9
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
