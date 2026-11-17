"""Research-only training for independent market-intelligence specialists.

The regime target is the causal regime observed horizon bars ahead. The
volatility target is realized log-return volatility over the next horizon bars.
Every outer test is chronological and separated from training by a purge gap.
No artifact is promoted or made available to inference by this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error

from .evaluation import walk_forward_folds
from .features import build_features, feature_columns_for_frame
from .regimes import classify_market_regime

REGIME_CLASSES = ("downtrend", "high_volatility", "range", "uptrend")


def _positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _prepare_dataset(frame: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    data = frame.copy().reset_index(drop=True)
    timestamps = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("timestamps must be valid, unique and chronological")
    numeric = data[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("OHLCV values must be finite")
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any() or (numeric["volume"] < 0).any():
        raise ValueError("prices must be positive and volume cannot be negative")
    data[["open", "high", "low", "close", "volume"]] = numeric
    features = build_features(data)
    columns = feature_columns_for_frame(data)
    regimes = classify_market_regime(data[["close"]])["regime"]
    future_regime = regimes.shift(-horizon).rename("future_regime")
    log_returns = np.log(data["close"]).diff()
    # At row t, this window contains returns t+1 through t+horizon.
    future_volatility = log_returns.rolling(horizon, min_periods=horizon).std().shift(-horizon)
    dataset = features.copy()
    dataset["future_regime"] = future_regime
    dataset["future_volatility"] = future_volatility
    dataset["timestamp"] = timestamps
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna(subset=columns + ["future_regime", "future_volatility"]).reset_index(drop=True)
    dataset = dataset[dataset["future_regime"].isin(REGIME_CLASSES)].reset_index(drop=True)
    if len(dataset) == 0:
        raise ValueError("No usable rows remain after causal feature and future-target construction")
    return dataset, columns


def _fingerprint(dataset: pd.DataFrame, columns: list[str], horizon: int) -> dict[str, str]:
    data_digest = hashlib.sha256(
        pd.util.hash_pandas_object(dataset, index=True).values.tobytes()
    ).hexdigest()
    config = {"horizon": horizon, "features": columns, "roles": ["regime_classifier", "volatility_estimator"],
              "estimator_family": "HistGradientBoosting", "split": "expanding_purged_chronological"}
    config_digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    code_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    experiment = hashlib.sha256(f"{code_digest}:{config_digest}:{data_digest}".encode()).hexdigest()
    return {"dataset_sha256": data_digest, "config_sha256": config_digest,
            "code_sha256": code_digest, "experiment_id": "eti-exp-" + experiment[:16]}


def train_market_intelligence_specialists(
    frame: pd.DataFrame,
    *,
    horizon: int = 6,
    min_train: int = 500,
    validation_size: int = 100,
    test_size: int = 100,
    step_size: int = 100,
    max_model_fits: int = 32,
) -> dict:
    """Evaluate regime and forward-volatility specialists on untouched outer folds.

    The two model fits per role/fold include one naive comparator. The fit cap is
    checked before the first estimator is fitted. Returned models are deliberately
    not serialized, registered, promoted, or connected to inference.
    """
    horizon = _positive_int(horizon, "horizon")
    min_train = _positive_int(min_train, "min_train", 20)
    validation_size = _positive_int(validation_size, "validation_size", 10)
    test_size = _positive_int(test_size, "test_size", 10)
    step_size = _positive_int(step_size, "step_size")
    max_model_fits = _positive_int(max_model_fits, "max_model_fits")
    dataset, features = _prepare_dataset(frame, horizon)
    folds = walk_forward_folds(
        len(dataset), min_train=min_train, validation_size=validation_size,
        test_size=test_size, step_size=step_size, purge_bars=horizon,
    )
    roles = ("regime_classifier", "volatility_estimator")
    estimated_fits = len(folds) * len(roles) * 2
    if estimated_fits > max_model_fits:
        raise ValueError(
            f"Training stopped before fitting: estimated {estimated_fits} model fits "
            f"exceed cap {max_model_fits}"
        )

    regime_results, volatility_results = [], []
    for fold_number, fold in enumerate(folds):
        train = dataset.iloc[:fold.train_end]
        test = dataset.iloc[fold.test_start:fold.test_end]
        x_train, x_test = train[features], test[features]

        y_regime_train = train["future_regime"].astype(str)
        y_regime_test = test["future_regime"].astype(str)
        if y_regime_train.nunique() < 2:
            raise ValueError(f"Fold {fold_number} regime training has fewer than two classes")
        regime_model = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42,
        ).fit(x_train, y_regime_train)
        regime_baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_regime_train)
        pred = regime_model.predict(x_test)
        baseline_pred = regime_baseline.predict(x_test)
        regime_results.append({
            "fold": fold_number,
            "train_rows": len(train), "test_rows": len(test),
            "test_start": test["timestamp"].iloc[0].isoformat(),
            "test_end": test["timestamp"].iloc[-1].isoformat(),
            "class_counts": {name: int((y_regime_test == name).sum()) for name in REGIME_CLASSES},
            "model": {
                "accuracy": float(accuracy_score(y_regime_test, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_regime_test, pred)),
            },
            "majority_baseline": {
                "accuracy": float(accuracy_score(y_regime_test, baseline_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_regime_test, baseline_pred)),
            },
        })

        y_vol_train = train["future_volatility"].astype(float)
        y_vol_test = test["future_volatility"].astype(float)
        volatility_model = HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, loss="squared_error", random_state=42,
        ).fit(x_train, y_vol_train)
        volatility_baseline = DummyRegressor(strategy="median").fit(x_train, y_vol_train)
        vol_pred = np.maximum(volatility_model.predict(x_test), 0.0)
        vol_baseline_pred = np.maximum(volatility_baseline.predict(x_test), 0.0)
        volatility_results.append({
            "fold": fold_number,
            "train_rows": len(train), "test_rows": len(test),
            "test_start": test["timestamp"].iloc[0].isoformat(),
            "test_end": test["timestamp"].iloc[-1].isoformat(),
            "model": {
                "mae": float(mean_absolute_error(y_vol_test, vol_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_vol_test, vol_pred))),
                "mean_prediction": float(np.mean(vol_pred)),
                "mean_actual": float(y_vol_test.mean()),
            },
            "median_baseline": {
                "mae": float(mean_absolute_error(y_vol_test, vol_baseline_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_vol_test, vol_baseline_pred))),
            },
        })

    lineage = _fingerprint(dataset, features, horizon)
    return {
        "schema": "eti.market_intelligence_training.v1",
        "department": "market_intelligence",
        "horizon_bars": horizon,
        "rows": len(dataset),
        "feature_columns": features,
        "fold_count": len(folds),
        "roles": {
            "regime_classifier": {
                "model_role": "regime_classifier",
                "outer_test_folds": regime_results,
                "mean_outer_test_balanced_accuracy": float(np.mean([x["model"]["balanced_accuracy"] for x in regime_results])),
                "mean_baseline_balanced_accuracy": float(np.mean([x["majority_baseline"]["balanced_accuracy"] for x in regime_results])),
                "status": "research_candidate_only",
            },
            "volatility_estimator": {
                "model_role": "volatility_estimator",
                "target": "standard_deviation_of_next_horizon_log_returns",
                "outer_test_folds": volatility_results,
                "mean_outer_test_mae": float(np.mean([x["model"]["mae"] for x in volatility_results])),
                "mean_baseline_mae": float(np.mean([x["median_baseline"]["mae"] for x in volatility_results])),
                "status": "research_candidate_only",
            },
        },
        "compute_budget": {
            "estimated_model_fit_calls": estimated_fits,
            "actual_model_fit_calls": estimated_fits,
            "max_model_fit_calls": max_model_fits,
            "fit_calls_per_fold": 4,
            "note": "Fit-call count is a compute proxy, not a dollar-cost estimate.",
        },
        "lineage": lineage,
        "promotion": {
            "eligible": False,
            "reason": "Research-only pipeline; multi-market replication, calibration/uncertainty review and independent approval are still required.",
        },
        "execution_enabled": False,
        "warning": "Outer folds are chronological and purge horizon-overlapping labels. Fold metrics are descriptive; adjacent expanding folds may overlap and are not independent evidence.",
    }
