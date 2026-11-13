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

    def predict_proba(self, x) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Model has not been fitted")
        raw = self.estimator.predict_proba(x)
        aligned = np.zeros((len(raw), len(CLASSES)), dtype=float)
        for i, cls in enumerate(self.classes_):
            aligned[:, int(cls)] = raw[:, i]
        return aligned

    def predict(self, x) -> np.ndarray:
        return CLASSES[self.predict_proba(x).argmax(axis=1)]
