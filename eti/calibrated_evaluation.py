"""Nested walk-forward evaluation with a purged, fold-local probability-calibration block.

Candidate selection, temperature fitting and outer-test assessment use disjoint
chronological windows. The outer test is never used to fit or change a model.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    precision_recall_fscore_support,
)

from .evaluation import walk_forward_folds
from .model import CLASSES, _reliability_diagnostics


def _probability_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if (probabilities.shape != (len(labels), len(CLASSES)) or not len(labels)
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or (probabilities.sum(axis=1) <= 0).any()
            or not np.isin(labels, CLASSES).all()):
        raise ValueError("Probability evaluation inputs are invalid")
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    chosen = probabilities[np.arange(len(labels)), labels]
    one_hot = np.eye(len(CLASSES), dtype=float)[labels]
    counts = {str(int(cls)): int(np.sum(labels == cls)) for cls in CLASSES}
    return {
        "sample_count": int(len(labels)),
        "class_counts": counts,
        "negative_log_likelihood": float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean()),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "reliability": _reliability_diagnostics(probabilities, labels),
    }


def _aligned_probabilities(model, features: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba_raw"):
        return np.asarray(model.predict_proba_raw(features), dtype=float)
    estimator = getattr(model, "estimator", model)
    if not hasattr(estimator, "predict_proba") or not hasattr(estimator, "classes_"):
        return None
    raw = np.asarray(estimator.predict_proba(features), dtype=float)
    aligned = np.zeros((len(raw), len(CLASSES)), dtype=float)
    for index, label in enumerate(estimator.classes_):
        if int(label) in CLASSES:
            aligned[:, int(label)] = raw[:, index]
    if not np.isfinite(aligned).all() or (aligned.sum(axis=1) <= 0).any():
        return None
    return aligned / aligned.sum(axis=1, keepdims=True)


def run_nested_calibrated_walk_forward(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    model_factories: dict[str, Callable[[], object]],
    *,
    min_train: int = 500,
    validation_size: int = 100,
    test_size: int = 100,
    step_size: int = 100,
    purge_bars: int = 6,
    minimum_calibration_samples: int = 30,
    minimum_samples_per_class: int = 5,
) -> dict:
    """Fit each candidate once per fold; reserve disjoint validation and calibration blocks.

    The chronological layout is:
      train -> purge -> candidate-selection validation -> purge ->
      temperature calibration -> purge -> untouched outer test.

    If the configured validation window is too small to contain a calibration
    block, the fold remains evaluable but is explicitly reported uncalibrated.
    """
    if not model_factories or any(not isinstance(name, str) or not name.strip()
                                  for name in model_factories):
        raise ValueError("At least one named model factory is required")
    if not feature_columns or label_column in feature_columns:
        raise ValueError("Feature columns must be non-empty and distinct from the label")
    required = feature_columns + [label_column]
    if set(required) - set(dataset.columns):
        raise ValueError("Dataset is missing required feature or label columns")
    if (isinstance(minimum_calibration_samples, bool)
            or not isinstance(minimum_calibration_samples, int)
            or minimum_calibration_samples < 30):
        raise ValueError("minimum_calibration_samples must be an integer >= 30")
    if (isinstance(minimum_samples_per_class, bool)
            or not isinstance(minimum_samples_per_class, int)
            or minimum_samples_per_class < 1):
        raise ValueError("minimum_samples_per_class must be a positive integer")

    clean = dataset[required].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    folds = walk_forward_folds(
        len(clean), min_train=min_train, validation_size=validation_size,
        test_size=test_size, step_size=step_size, purge_bars=purge_bars,
    )
    x, y = clean[feature_columns], clean[label_column].astype(int)
    if not y.isin(CLASSES).all():
        raise ValueError("Labels must use the short/flat/long class IDs 0, 1 and 2")
    class_labels = sorted(int(item) for item in pd.unique(y))
    results = []

    for number, fold in enumerate(folds):
        train_x, train_y = x.iloc[:fold.train_end], y.iloc[:fold.train_end]
        validation_start, validation_end = fold.validation_start, fold.validation_end
        test_x = x.iloc[fold.test_start:fold.test_end]
        test_y = y.iloc[fold.test_start:fold.test_end]
        if train_y.nunique() < 2:
            raise ValueError(f"Fold {number} training data contains fewer than two classes")

        # Leave a full purge gap between candidate selection and calibration.
        selection_size = max(1, (validation_size - purge_bars) // 2)
        calibration_start = validation_start + selection_size + purge_bars
        calibration_x = x.iloc[calibration_start:validation_end]
        calibration_y = y.iloc[calibration_start:validation_end]
        has_calibration_window = len(calibration_y) >= minimum_calibration_samples
        if has_calibration_window:
            selection_x = x.iloc[validation_start:validation_start + selection_size]
            selection_y = y.iloc[validation_start:validation_start + selection_size]
        else:
            # Do not pretend a tiny remainder is an independent calibration set.
            selection_x = x.iloc[validation_start:validation_end]
            selection_y = y.iloc[validation_start:validation_end]

        fitted_models = {}
        validation_scores = {}
        for name, factory in model_factories.items():
            model = factory()
            model.fit(train_x, train_y)
            prediction = np.asarray(model.predict(selection_x))
            if len(prediction) != len(selection_y):
                raise ValueError(f"Candidate {name} returned an invalid selection prediction count")
            fitted_models[name] = model
            validation_scores[name] = float(balanced_accuracy_score(selection_y, prediction))

        selected = max(model_factories, key=lambda name: (validation_scores[name], name))
        selected_model = fitted_models[selected]
        calibration_fit = {
            "status": "not_attempted",
            "method": "temperature_scaling",
            "calibration_samples": int(len(calibration_y)) if has_calibration_window else 0,
            "purge_bars": int(purge_bars),
        }
        if has_calibration_window and hasattr(selected_model, "fit_temperature_calibration"):
            calibration_fit = selected_model.fit_temperature_calibration(
                calibration_x, calibration_y.to_numpy(),
                min_samples=minimum_calibration_samples,
            )
            calibration_fit["status"] = (
                "fit" if calibration_fit.get("passed") is True else "not_calibrated"
            )
            calibration_fit["purge_bars"] = int(purge_bars)
        elif not has_calibration_window:
            calibration_fit["status"] = "insufficient_calibration_window"
        else:
            calibration_fit["status"] = "selected_model_does_not_support_temperature_scaling"

        test_candidate_metrics = {}
        predictions_by_candidate = {}
        for name, model in fitted_models.items():
            pred = np.asarray(model.predict(test_x)).astype(int)
            if len(pred) != len(test_y):
                raise ValueError(f"Candidate {name} returned an invalid outer-test prediction count")
            predictions_by_candidate[name] = pred
            precision, recall, f1, support = precision_recall_fscore_support(
                test_y, pred, labels=class_labels, zero_division=0
            )
            test_candidate_metrics[name] = {
                "balanced_accuracy": float(balanced_accuracy_score(test_y, pred)),
                "accuracy": float(accuracy_score(test_y, pred)),
                "per_class": {
                    str(label): {
                        "precision": float(precision[j]), "recall": float(recall[j]),
                        "f1": float(f1[j]), "support": int(support[j]),
                    } for j, label in enumerate(class_labels)
                },
                "confusion_matrix_labels": [str(label) for label in class_labels],
                "confusion_matrix": confusion_matrix(
                    test_y, pred, labels=class_labels
                ).astype(int).tolist(),
            }

        raw_probabilities = _aligned_probabilities(selected_model, test_x)
        calibrated_probabilities = None
        if raw_probabilities is not None and calibration_fit.get("passed") is True:
            if hasattr(selected_model, "predict_proba"):
                calibrated_probabilities = np.asarray(selected_model.predict_proba(test_x), dtype=float)
        selected_test_pred = predictions_by_candidate[selected]
        probability_report = {
            "status": "unavailable",
            "calibration_fit": calibration_fit,
            "raw": None,
            "temperature_scaled": None,
            "outer_test_confirmation": "not_assessed",
        }
        if raw_probabilities is not None:
            raw_metrics = _probability_metrics(raw_probabilities, test_y.to_numpy())
            probability_report["raw"] = raw_metrics
            probability_report["status"] = "uncalibrated"
            if calibrated_probabilities is not None:
                calibrated_metrics = _probability_metrics(
                    calibrated_probabilities, test_y.to_numpy()
                )
                probability_report["temperature_scaled"] = calibrated_metrics
                class_coverage = all(
                    calibrated_metrics["class_counts"][str(int(cls))] >= minimum_samples_per_class
                    for cls in CLASSES
                )
                nll_not_worse = (
                    calibrated_metrics["negative_log_likelihood"]
                    <= raw_metrics["negative_log_likelihood"] + 1e-9
                )
                brier_not_worse = (
                    calibrated_metrics["brier_score"] <= raw_metrics["brier_score"] + 1e-9
                )
                if not class_coverage:
                    confirmation = "insufficient_outer_test_class_coverage"
                elif not nll_not_worse:
                    confirmation = "outer_test_nll_worse_than_raw"
                elif not brier_not_worse:
                    confirmation = "outer_test_brier_worse_than_raw"
                else:
                    confirmation = "calibration_confirmed_on_outer_test"
                probability_report["outer_test_confirmation"] = confirmation
                probability_report["outer_test_class_coverage_passed"] = bool(class_coverage)
                probability_report["outer_test_nll_not_worse"] = bool(nll_not_worse)
                probability_report["outer_test_brier_not_worse"] = bool(brier_not_worse)
                probability_report["status"] = (
                    "calibrated_and_outer_test_confirmed"
                    if confirmation == "calibration_confirmed_on_outer_test"
                    else "calibration_not_confirmed_on_outer_test"
                )
                selected_test_pred = np.asarray(CLASSES)[calibrated_probabilities.argmax(axis=1)]

        test_candidate_metrics[selected]["balanced_accuracy"] = float(
            balanced_accuracy_score(test_y, selected_test_pred)
        )
        test_candidate_metrics[selected]["accuracy"] = float(
            accuracy_score(test_y, selected_test_pred)
        )
        results.append({
            "fold": number,
            "selected_candidate": selected,
            "validation_scores": validation_scores,
            "test_candidate_metrics": test_candidate_metrics,
            "test_balanced_accuracy": test_candidate_metrics[selected]["balanced_accuracy"],
            "test_accuracy": test_candidate_metrics[selected]["accuracy"],
            "test_predictions": selected_test_pred.astype(int).tolist(),
            "test_actual": test_y.astype(int).tolist(),
            "test_class_counts": {
                str(int(label)): int((test_y == label).sum()) for label in CLASSES
            },
            "probability_calibration": probability_report,
            "boundaries": {
                "train_start": int(fold.train_start), "train_end": int(fold.train_end),
                "selection_start": int(validation_start),
                "selection_end": int(validation_start + selection_size) if has_calibration_window else int(validation_end),
                "calibration_start": int(calibration_start) if has_calibration_window else None,
                "calibration_end": int(validation_end) if has_calibration_window else None,
                "test_start": int(fold.test_start), "test_end": int(fold.test_end),
                "purge_bars": int(purge_bars),
            },
        })

    candidate_summary = {}
    for name in model_factories:
        scores = [row["test_candidate_metrics"][name]["balanced_accuracy"] for row in results]
        accuracies = [row["test_candidate_metrics"][name]["accuracy"] for row in results]
        candidate_summary[name] = {
            "mean_outer_test_balanced_accuracy": float(np.mean(scores)),
            "std_outer_test_balanced_accuracy": float(np.std(scores)),
            "worst_outer_test_balanced_accuracy": float(np.min(scores)),
            "mean_outer_test_accuracy": float(np.mean(accuracies)),
            "selected_on_validation_count": int(sum(row["selected_candidate"] == name for row in results)),
            "fold_count": len(results),
        }
    calibration_rows = [row["probability_calibration"] for row in results]
    confirmed = [row for row in calibration_rows
                 if row["status"] == "calibrated_and_outer_test_confirmed"]
    assessed = [row for row in calibration_rows if row["temperature_scaled"] is not None]
    return {
        "fold_count": len(results),
        "folds": results,
        "mean_test_balanced_accuracy": float(np.mean([row["test_balanced_accuracy"] for row in results])),
        "candidate_summary": candidate_summary,
        "probability_calibration": {
            "method": "fold_local_temperature_scaling",
            "fold_count": len(results),
            "calibration_attempted_folds": int(sum(
                row["calibration_fit"].get("status") in ("fit", "not_calibrated")
                for row in calibration_rows
            )),
            "outer_test_assessed_folds": len(assessed),
            "outer_test_confirmed_folds": len(confirmed),
            "mean_raw_outer_test_nll": (
                float(np.mean([row["raw"]["negative_log_likelihood"] for row in calibration_rows
                               if row["raw"] is not None])) if any(row["raw"] is not None for row in calibration_rows) else None
            ),
            "mean_calibrated_outer_test_nll": (
                float(np.mean([row["temperature_scaled"]["negative_log_likelihood"] for row in assessed]))
                if assessed else None
            ),
            "mean_raw_outer_test_brier": (
                float(np.mean([row["raw"]["brier_score"] for row in calibration_rows
                               if row["raw"] is not None])) if any(row["raw"] is not None for row in calibration_rows) else None
            ),
            "mean_calibrated_outer_test_brier": (
                float(np.mean([row["temperature_scaled"]["brier_score"] for row in assessed]))
                if assessed else None
            ),
            "warning": "Calibration is fitted only on each fold's purged calibration block. Outer-test metrics are independent diagnostics and never change the fitted temperature or candidate choice. Small test windows may not support per-class confirmation.",
        },
        "warning": "Nested research evaluation only. Candidate selection uses a separate validation block; temperature scaling uses a later purged calibration block; outer test is untouched until final assessment. Fold metrics are not proof of future profitability.",
    }
