"""Executable purged walk-forward evaluation; selection never consumes test labels."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

@dataclass(frozen=True)
class Fold:
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int

def walk_forward_folds(n_rows: int, min_train: int = 500, validation_size: int = 100,
                       test_size: int = 100, step_size: int = 100,
                       purge_bars: int = 6) -> list[Fold]:
    """Expanding-window folds with purge gaps before validation and test windows."""
    sizes = (n_rows, min_train, validation_size, test_size, step_size, purge_bars)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in sizes):
        raise ValueError("Row counts and window sizes must be integers")
    if min(n_rows, min_train, validation_size, test_size, step_size) <= 0 or purge_bars < 0:
        raise ValueError("Row counts and window sizes must be positive; purge cannot be negative")
    folds = []
    test_start = min_train + purge_bars + validation_size + purge_bars
    while test_start + test_size <= n_rows:
        validation_end = test_start - purge_bars
        validation_start = validation_end - validation_size
        train_end = validation_start - purge_bars
        if train_end >= min_train:
            folds.append(Fold(0, train_end, validation_start, validation_end,
                              test_start, test_start + test_size))
        test_start += step_size
    if not folds:
        raise ValueError("Not enough rows to construct one purged walk-forward fold")
    return folds

def run_walk_forward(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    fit_predict: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray],
    min_train: int = 500,
    validation_size: int = 100,
    test_size: int = 100,
    step_size: int = 100,
    purge_bars: int = 6,
) -> dict:
    """Run a caller-supplied deterministic model pipeline on chronological folds.

    fit_predict must fit only on the supplied train data and return predictions
    for the supplied evaluation features. This function never fits a model itself.
    """
    if not feature_columns or label_column in feature_columns:
        raise ValueError("Provide feature columns distinct from the label")
    required = feature_columns + [label_column]
    missing = set(required) - set(dataset.columns)
    if missing:
        raise ValueError(f"Missing evaluation columns: {sorted(missing)}")
    clean = dataset[required].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(clean) < min_train + validation_size + test_size + 2 * purge_bars:
        raise ValueError("Insufficient clean rows for walk-forward evaluation")
    folds = walk_forward_folds(len(clean), min_train, validation_size, test_size,
                               step_size, purge_bars)
    x, y = clean[feature_columns], clean[label_column]
    results = []
    for number, fold in enumerate(folds):
        train_x, train_y = x.iloc[fold.train_start:fold.train_end], y.iloc[fold.train_start:fold.train_end]
        valid_x, valid_y = x.iloc[fold.validation_start:fold.validation_end], y.iloc[fold.validation_start:fold.validation_end]
        test_x, test_y = x.iloc[fold.test_start:fold.test_end], y.iloc[fold.test_start:fold.test_end]
        if train_y.nunique() < 2:
            raise ValueError(f"Fold {number} training data contains fewer than two classes")
        valid_pred = np.asarray(fit_predict(train_x, train_y, valid_x))
        # Refit on pre-test history only; validation labels may now join training.
        pretest_end = fold.test_start - purge_bars
        refit_x, refit_y = x.iloc[:pretest_end], y.iloc[:pretest_end]
        if refit_y.nunique() < 2:
            raise ValueError(f"Fold {number} refit data contains fewer than two classes")
        test_pred = np.asarray(fit_predict(refit_x, refit_y, test_x))
        if len(valid_pred) != len(valid_y) or len(test_pred) != len(test_y):
            raise ValueError("fit_predict returned a prediction count that does not match evaluation rows")
        results.append({
            "fold": number, "boundaries": asdict(fold),
            "validation": {"accuracy": float(accuracy_score(valid_y, valid_pred)),
                           "balanced_accuracy": float(balanced_accuracy_score(valid_y, valid_pred))},
            "test": {"accuracy": float(accuracy_score(test_y, test_pred)),
                     "balanced_accuracy": float(balanced_accuracy_score(test_y, test_pred))},
        })
    return {"fold_count": len(results), "folds": results,
            "mean_test_accuracy": float(np.mean([r["test"]["accuracy"] for r in results])),
            "mean_test_balanced_accuracy": float(np.mean([r["test"]["balanced_accuracy"] for r in results])),
            "warning": "Research evaluation only. Test folds are reported, never used for candidate selection; results do not establish future profitability."}


def run_nested_walk_forward(dataset: pd.DataFrame, feature_columns: list[str],
                            label_column: str,
                            candidates: dict[str, Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray]],
                            min_train: int = 500, validation_size: int = 100,
                            test_size: int = 100, step_size: int = 100,
                            purge_bars: int = 6) -> dict:
    """Select a candidate on each outer validation window, then score untouched test.

    Each candidate callback must create and fit its own estimator per invocation.
    Outer test labels are never used for candidate choice.
    """
    if not candidates:
        raise ValueError("At least one named candidate callback is required")
    required = feature_columns + [label_column]
    if not feature_columns or label_column in feature_columns or set(required) - set(dataset.columns):
        raise ValueError("Invalid feature/label columns")
    clean = dataset[required].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    folds = walk_forward_folds(len(clean), min_train, validation_size, test_size,
                               step_size, purge_bars)
    x, y = clean[feature_columns], clean[label_column]
    from sklearn.metrics import balanced_accuracy_score
    results = []
    for i, fold in enumerate(folds):
        train_x, train_y = x.iloc[:fold.train_end], y.iloc[:fold.train_end]
        valid_x, valid_y = x.iloc[fold.validation_start:fold.validation_end], y.iloc[fold.validation_start:fold.validation_end]
        if train_y.nunique() < 2:
            raise ValueError(f"Fold {i} training data contains fewer than two classes")
        candidate_scores = {}
        for name, callback in candidates.items():
            if not name:
                raise ValueError("Candidate names cannot be empty")
            pred = np.asarray(callback(train_x, train_y, valid_x))
            if len(pred) != len(valid_y):
                raise ValueError(f"Candidate {name} returned an invalid validation prediction count")
            candidate_scores[name] = float(balanced_accuracy_score(valid_y, pred))
        selected = max(candidates, key=lambda name: (candidate_scores[name], name))
        test_start = fold.test_start
        refit_end = test_start - purge_bars
        refit_x, refit_y = x.iloc[:refit_end], y.iloc[:refit_end]
        test_x, test_y = x.iloc[test_start:fold.test_end], y.iloc[test_start:fold.test_end]
        test_pred = np.asarray(candidates[selected](refit_x, refit_y, test_x))
        if len(test_pred) != len(test_y):
            raise ValueError(f"Selected candidate {selected} returned an invalid test prediction count")
        results.append({"fold": i, "selected_candidate": selected,
                        "validation_scores": candidate_scores,
                        "test_balanced_accuracy": float(balanced_accuracy_score(test_y, test_pred)),
                        "test_accuracy": float(accuracy_score(test_y, test_pred)),
                        "boundaries": asdict(fold)})
    scores = [r["test_balanced_accuracy"] for r in results]
    return {"fold_count": len(results), "folds": results,
            "mean_test_balanced_accuracy": float(np.mean(scores)),
            "warning": "Nested research evaluation; each outer test fold is excluded from candidate selection. Not evidence of future profitability."}


def run_multi_market_walk_forward(datasets: dict[str, pd.DataFrame],
                                  feature_columns: list[str], label_column: str,
                                  fit_predict: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray],
                                  **fold_options) -> dict:
    """Evaluate labelled instruments independently; never pool their price scales."""
    if not datasets:
        raise ValueError("At least one named market dataset is required")
    markets = {}
    for symbol, frame in sorted(datasets.items()):
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Market identifiers must be non-empty strings")
        markets[symbol] = run_walk_forward(frame, feature_columns, label_column,
                                            fit_predict, **fold_options)
    return {"market_count": len(markets), "markets": markets,
            "mean_market_test_balanced_accuracy": float(np.mean(
                [item["mean_test_balanced_accuracy"] for item in markets.values()])),
            "warning": "Markets are evaluated independently; this report is not a pooled portfolio backtest."}


def bootstrap_fold_confidence_interval(values: list[float], confidence: float = 0.95,
                                       repetitions: int = 2000, seed: int = 42) -> dict:
    """Percentile bootstrap over fold-level scores; interpret cautiously for few folds."""
    if not values or any(not np.isfinite(v) for v in values):
        raise ValueError("Provide a non-empty list of finite fold metrics")
    if not 0 < confidence < 1 or repetitions < 100:
        raise ValueError("confidence must be in (0,1) and repetitions >= 100")
    rng = np.random.default_rng(seed)
    samples = rng.choice(np.asarray(values, dtype=float),
                         size=(repetitions, len(values)), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return {"mean": float(np.mean(values)),
            "lower": float(np.quantile(samples, alpha)),
            "upper": float(np.quantile(samples, 1 - alpha)),
            "confidence": confidence, "repetitions": repetitions,
            "warning": "Fold bootstrap is descriptive; overlapping chronological folds are dependent."}
