"""End-to-end baseline experiment. Run: python -m eti.pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from .data import drop_incomplete_candles, fetch_ohlcv, fetch_ohlcv_auto, validate_ohlcv
from .features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS, build_features, feature_columns_for_frame, make_labels
from .model import BaselineModel, CLASSES, _reliability_diagnostics
from .artifacts import ModelArtifactStore
from .simulator import simulate_long_flat
from .performance import evaluate_predictions
from .regimes import classify_market_regime
from .universe import scan_market_universe
from .calibrated_evaluation import run_nested_calibrated_walk_forward


NESTED_CANDIDATE_NAMES = ("hist_gradient_boosting", "majority_class")


def _estimate_observed_bars_per_year(
    frame: pd.DataFrame, dataset: pd.DataFrame
) -> tuple[float, dict]:
    """Estimate annual bar frequency from actual usable timestamps.

    This accounts for overnight/weekend/session gaps in exchange-traded markets.
    It is an observed sample-rate estimate, not a verified exchange calendar.
    """
    if len(dataset) < 2:
        raise ValueError("At least two usable observations are required for annualization")
    if "timestamp" not in frame.columns:
        raise ValueError("OHLCV timestamps are required for annualization")
    timestamps = pd.to_datetime(
        frame.loc[dataset.index, "timestamp"], utc=True, errors="coerce"
    )
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Usable market timestamps must be valid and chronological")
    elapsed_seconds = float((timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds())
    if not np.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("Usable market timestamps must span a positive duration")
    bars_per_year = float(
        len(dataset) * (365.25 * 24 * 60 * 60) / elapsed_seconds
    )
    if not np.isfinite(bars_per_year) or bars_per_year <= 0:
        raise ValueError("Unable to estimate a valid annual bar frequency")
    return bars_per_year, {
        "method": "observed_usable_bars_per_calendar_year",
        "usable_rows": int(len(dataset)),
        "elapsed_calendar_seconds": elapsed_seconds,
        "first_timestamp": timestamps.iloc[0].isoformat(),
        "last_timestamp": timestamps.iloc[-1].isoformat(),
        "bars_per_year": bars_per_year,
        "warning": (
            "Sample-rate estimate; not a substitute for a verified exchange "
            "calendar or holiday/session model."
        ),
    }


def run_experiment(frame: pd.DataFrame, horizon: int = 6, timeframe: str = "1h") -> dict:
    frame = validate_ohlcv(frame)
    features = build_features(frame)
    feature_columns = feature_columns_for_frame(frame)
    labels = make_labels(frame, horizon=horizon)
    dataset = features.join(labels.rename("label")).dropna(subset=feature_columns + ["label"])
    if len(dataset) < 200:
        raise ValueError(f"Need at least 200 usable rows; got {len(dataset)}")
    x, y = dataset[feature_columns], dataset["label"].astype(int)
    n = len(dataset)
    train_end, valid_end = int(n * 0.70), int(n * 0.85)
    train_stop = max(0, train_end - horizon)
    valid_start = min(n, train_end + horizon)
    valid_stop = max(valid_start, valid_end - horizon)
    test_start = min(n, valid_end + horizon)
    if train_stop < 100 or valid_stop - valid_start < 20 or n - test_start < 20:
        raise ValueError("Not enough rows after purging label-overlap at split boundaries")
    if len(np.unique(y.iloc[:train_stop])) < 2:
        raise ValueError("Training period has fewer than two classes; collect more data or adjust threshold")
    model = BaselineModel().fit(x.iloc[:train_stop], y.iloc[:train_stop])
    y_valid = y.iloc[valid_start:valid_stop]
    valid_pred = model.predict(x.iloc[valid_start:valid_stop])
    valid_metrics = {
        "accuracy": float(accuracy_score(y_valid, valid_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_valid, valid_pred)),
    }
    final_model = BaselineModel().fit(x.iloc[:valid_end-horizon], y.iloc[:valid_end-horizon])
    y_test = y.iloc[test_start:]
    test_pred = final_model.predict(x.iloc[test_start:])
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
    }
    bars_per_year, annualization = _estimate_observed_bars_per_year(frame, dataset)
    test_index = dataset.index[test_start:]
    sim = simulate_long_flat(
        frame.loc[test_index, "close"].reset_index(drop=True),
        pd.Series(test_pred),
        bars_per_year=bars_per_year,
    )
    return {
        "timeframe": timeframe,
        "rows": n,
        "split": {"train": train_stop, "validation": valid_stop-valid_start, "test": n-test_start, "purge_bars_each_boundary": horizon},
        "validation": valid_metrics,
        "test": test_metrics,
        "paper_simulation": {k: v for k, v in sim.items() if k != "equity_curve"},
        "annualization": annualization,
        "warning": "Single chronological holdout baseline with horizon purging. Not evidence of profitability or future performance.",
    }



def _probability_validation(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """Evaluate probabilities on a chronological sample not used to fit temperature."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if (
        probabilities.shape != (len(labels), len(CLASSES))
        or len(labels) == 0 or not np.isfinite(probabilities).all()
        or (probabilities < 0).any() or (probabilities > 1).any()
        or (probabilities.sum(axis=1) <= 0).any()
        or not np.isin(labels, CLASSES).all()
    ):
        raise ValueError("Independent probability-validation inputs are invalid")
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    one_hot = np.eye(len(CLASSES), dtype=float)[labels]
    chosen = probabilities[np.arange(len(labels)), labels]
    return {
        "sample_count": int(len(labels)),
        "class_counts": {str(int(cls)): int(np.sum(labels == cls)) for cls in CLASSES},
        "negative_log_likelihood": float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean()),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "reliability": _reliability_diagnostics(probabilities, labels),
    }


def fit_deployable_model(
    frame: pd.DataFrame, horizon: int = 6, *,
    holdout_reserved_from_prior_evaluation: bool = False,
) -> tuple[BaselineModel, dict]:
    """Fit, calibrate and independently validate probabilities on three chronological blocks.

    The classifier is fitted on the first 60%, temperature is fitted on the next
    20%, and the final 20% is reserved for independent probability validation.
    Label-horizon purge gaps separate all three blocks. The final block never
    influences model fitting or temperature selection.
    """
    frame = validate_ohlcv(frame)
    features = build_features(frame)
    feature_columns = feature_columns_for_frame(frame)
    labels = make_labels(frame, horizon=horizon)
    dataset = features.join(labels.rename("label")).dropna(subset=feature_columns + ["label"])
    if len(dataset) < 350:
        raise ValueError(
            f"Need at least 350 usable rows for a purged 60/20/20 fit, calibration "
            f"and independent-validation split; got {len(dataset)}"
        )
    calibration_start = int(len(dataset) * 0.60)
    validation_start = int(len(dataset) * 0.80)
    training_end = calibration_start - int(horizon)
    calibration_end = validation_start - int(horizon)
    training = dataset.iloc[:training_end]
    calibration = dataset.iloc[calibration_start:calibration_end]
    validation = dataset.iloc[validation_start:]
    purged_rows = (calibration_start - training_end) + (validation_start - calibration_end)
    if len(training) < 200 or len(calibration) < 30 or len(validation) < 30:
        raise ValueError(
            "Export split requires at least 200 training, 30 calibration and 30 "
            "independent-validation rows after label-horizon purging"
        )

    x_train = training[feature_columns]
    y_train = training["label"].astype(int)
    x_calibration = calibration[feature_columns]
    y_calibration = calibration["label"].astype(int)
    x_validation = validation[feature_columns]
    y_validation = validation["label"].astype(int)
    model = BaselineModel().fit(x_train, y_train)
    calibration_report = model.fit_temperature_calibration(x_calibration, y_calibration)

    raw_validation = model.predict_proba_raw(x_validation)
    calibrated_validation = model.predict_proba(x_validation)
    raw_metrics = _probability_validation(raw_validation, y_validation.to_numpy())
    calibrated_metrics = _probability_validation(calibrated_validation, y_validation.to_numpy())
    independent_class_coverage = all(
        count >= 5 for count in calibrated_metrics["class_counts"].values()
    ) and all(
        calibrated_metrics["class_counts"][str(int(cls))] >= 5 for cls in CLASSES
    )
    independent_nll_not_worse = (
        calibrated_metrics["negative_log_likelihood"]
        <= raw_metrics["negative_log_likelihood"] + 1e-9
    )
    independent_passed = bool(
        model.calibration_status == "calibrated"
        and independent_class_coverage
        and independent_nll_not_worse
    )
    independent_report = {
        "passed": independent_passed,
        "method": "untouched_chronological_holdout",
        "minimum_samples_per_class": 5,
        "class_coverage_passed": independent_class_coverage,
        "nll_not_worse_than_raw": independent_nll_not_worse,
        "raw": raw_metrics,
        "temperature_scaled": calibrated_metrics,
        "calibration_applied_to_export": independent_passed,
        "reserved_from_prior_research_evaluation": bool(holdout_reserved_from_prior_evaluation),
    }
    if model.calibration_status == "calibrated" and not independent_passed:
        model.calibration_status = "uncalibrated"
        calibration_report["passed"] = False
        calibration_report["temperature"] = 1.0
        calibration_report["reason"] = (
            "independent_holdout_missing_class_coverage"
            if not independent_class_coverage
            else "temperature_scaling_failed_independent_holdout_nll"
        )
    calibration_report["independent_validation"] = independent_report
    model.calibration_report = calibration_report

    dataset_sha256 = hashlib.sha256(
        pd.util.hash_pandas_object(dataset, index=True).values.tobytes()
    ).hexdigest()
    schema = {
        "features": list(feature_columns),
        "dataset_sha256": dataset_sha256,
        "feature_version": "eti.features.v2" if feature_columns != BASE_FEATURE_COLUMNS else "eti.features.v1",
        "training_rows": int(len(training)),
        "calibration_rows": int(len(calibration)),
        "independent_validation_rows": int(len(validation)),
        "purged_rows": int(purged_rows),
        "training_end": training.index[-1].isoformat() if hasattr(training.index[-1], "isoformat") else str(training.index[-1]),
        "calibration_start": calibration.index[0].isoformat() if hasattr(calibration.index[0], "isoformat") else str(calibration.index[0]),
        "calibration_end": calibration.index[-1].isoformat() if hasattr(calibration.index[-1], "isoformat") else str(calibration.index[-1]),
        "independent_validation_start": validation.index[0].isoformat() if hasattr(validation.index[0], "isoformat") else str(validation.index[0]),
        "independent_validation_end": validation.index[-1].isoformat() if hasattr(validation.index[-1], "isoformat") else str(validation.index[-1]),
        "label_horizon_bars": int(horizon),
        "calibration_status": model.calibration_status,
        "calibration_report": calibration_report,
        "independent_probability_validation": independent_report,
    }
    return model, schema

def export_deployable_model(
    frame: pd.DataFrame, model_id: str, version: str, training_data_ref: str,
    horizon: int = 6, *,
    timeframe: str | None = None,
    holdout_reserved_from_prior_evaluation: bool = False,
) -> dict:
    """Explicitly fit and store an immutable artifact; never called at import/startup."""
    model, input_schema = fit_deployable_model(
        frame, horizon=horizon,
        holdout_reserved_from_prior_evaluation=holdout_reserved_from_prior_evaluation,
    )
    if timeframe is not None:
        if not isinstance(timeframe, str) or timeframe not in {"1h", "4h", "1d"}:
            raise ValueError("export timeframe must be one of 1h, 4h, or 1d")
        input_schema["timeframes"] = [timeframe]
    input_schema["market_types"] = ["spot"]
    artifact = ModelArtifactStore().save_model(model)
    # This export entry point is the spot baseline; derivatives require their
    # own contract-aware export and cost-evidence workflow.
    # Every distinct artifact receives an immutable content-addressed version.
    # Repeating an identical deterministic experiment may reproduce the same
    # digest; it is not mislabeled as a different trained model.
    artifact_digest = artifact["artifact_sha256"]
    if len(version) > 65:
        raise ValueError("base model version must be at most 65 characters")
    base_version = version
    immutable_version = f"{base_version}+{artifact_digest[:12]}"
    config = {
        "estimator": "HistGradientBoostingClassifier",
        "max_iter": 120,
        "learning_rate": 0.06,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
        "random_state": 42,
        "horizon": int(horizon),
        "market_types": input_schema.get("market_types"),
        "timeframes": input_schema.get("timeframes"),
        "feature_version": input_schema.get("feature_version"),
        "features": input_schema.get("features"),
    }
    config_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    code_digest = hashlib.sha256()
    for source_path in (
        Path(__file__),
        Path(__file__).with_name("model.py"),
        Path(__file__).with_name("features.py"),
        Path(__file__).with_name("data.py"),
    ):
        code_digest.update(source_path.name.encode("utf-8"))
        code_digest.update(source_path.read_bytes())
    code_sha256 = code_digest.hexdigest()
    dataset_sha256 = input_schema["dataset_sha256"]
    experiment_digest = hashlib.sha256(
        f"{code_sha256}:{config_sha256}:{dataset_sha256}".encode("utf-8")
    ).hexdigest()
    experiment_id = "eti-exp-" + experiment_digest[:16]
    input_schema.update({
        "department": "prediction",
        "model_role": "direction_classifier",
        "experiment_id": experiment_id,
        "lineage": {
            "code_sha256": code_sha256,
            "config_sha256": config_sha256,
            "dataset_sha256": dataset_sha256,
            "data_cutoff": input_schema.get("independent_validation_end"),
        },
    })
    return {
        "model_id": model_id,
        "version": immutable_version,
        "base_version": base_version,
        "department": "prediction",
        "model_role": "direction_classifier",
        "experiment_id": experiment_id,
        **artifact,
        "input_schema": input_schema,
        "supported_classes": ["crypto_spot"],
        "training_data_ref": training_data_ref,
        "calibration_status": model.calibration_status,
        "calibration_report": model.calibration_report,
        "note": "Candidate artifact only. Review evaluation and calibration evidence, then obtain independent approval before inference.",
    }


def estimate_nested_model_fits_from_rows(n_rows: int, horizon: int = 6,
                                          min_train: int = 400,
                                          validation_size: int = 100,
                                          test_size: int = 100,
                                          step_size: int = 200,
                                          candidate_count: int = 2) -> int:
    """Estimate model fit calls before training; each candidate is fitted once per fold."""
    from .evaluation import walk_forward_folds
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count < 1:
        raise ValueError("candidate_count must be a positive integer")
    folds = walk_forward_folds(
        n_rows, min_train=min_train, validation_size=validation_size,
        test_size=test_size, step_size=step_size, purge_bars=horizon,
    )
    return len(folds) * candidate_count


def run_nested_experiment(frame: pd.DataFrame, horizon: int = 6,
                          min_train: int = 500, validation_size: int = 100,
                          test_size: int = 100, step_size: int = 100,
                          timeframe: str = "1h", allow_short: bool = False,
                          funding_bps_per_bar: float = 0.0,
                          fee_bps: float = 6.0, slippage_bps: float = 2.0,
                          max_model_fits: int | None = None,
                          funding_events: list[dict] | None = None) -> dict:
    """Train candidates inside purged folds; configure long/short and explicit derivative cost assumptions."""
    from sklearn.dummy import DummyClassifier
    from .evaluation import walk_forward_folds
    frame = validate_ohlcv(frame)
    if funding_events and funding_bps_per_bar != 0:
        raise ValueError("Use either event-timed funding or legacy per-bar funding, not both")
    features = build_features(frame)
    feature_columns = feature_columns_for_frame(frame)
    labels = make_labels(frame, horizon=horizon)
    dataset = features.join(labels.rename("label")).dropna(subset=feature_columns + ["label"])
    dataset["close"] = frame.loc[dataset.index, "close"].astype(float)
    dataset["timestamp"] = frame.loc[dataset.index, "timestamp"].to_numpy()
    if len(dataset) < min_train + validation_size + test_size + 2 * horizon:
        raise ValueError("Insufficient completed data for nested walk-forward training")
    planned_folds = walk_forward_folds(
        len(dataset), min_train=min_train, validation_size=validation_size,
        test_size=test_size, step_size=step_size, purge_bars=horizon,
    )
    estimated_model_fits = len(planned_folds) * 2
    if max_model_fits is not None:
        if isinstance(max_model_fits, bool) or not isinstance(max_model_fits, int) or max_model_fits < 1:
            raise ValueError("max_model_fits must be a positive integer")
        if estimated_model_fits > max_model_fits:
            raise ValueError(
                f"Training stopped before fitting: estimated {estimated_model_fits} model fits "
                f"exceed the configured cap of {max_model_fits}. Increase step_size or reduce data/folds."
            )
    report = run_nested_calibrated_walk_forward(
        dataset, feature_columns, "label",
        {
            NESTED_CANDIDATE_NAMES[0]: BaselineModel,
            NESTED_CANDIDATE_NAMES[1]: lambda: DummyClassifier(strategy="most_frequent"),
        },
        min_train=min_train, validation_size=validation_size,
        test_size=test_size, step_size=step_size, purge_bars=horizon,
    )
    bars_per_year, annualization = _estimate_observed_bars_per_year(frame, dataset)
    # Causal regimes are aligned to the exact cleaned dataset rows.
    regime_frame = classify_market_regime(frame)
    regime_by_row = regime_frame.loc[dataset.index, "regime"].reset_index(drop=True)
    performance_folds = []
    candidate_return_streams = {
        name: {"timestamps": [], "net_return_differentials": []}
        for name in report["candidate_summary"]
    }
    for fold in report["folds"]:
        start, end = fold["boundaries"]["test_start"], fold["boundaries"]["test_end"]
        fold_regimes = regime_by_row.iloc[start:end]
        regime_counts = fold_regimes.value_counts().to_dict()
        regime_total = int(len(fold_regimes))
        regime_diagnostics = {
            "bar_count": regime_total,
            "counts": {str(name): int(count) for name, count in regime_counts.items()},
            "fractions": {str(name): float(count / regime_total) for name, count in regime_counts.items()},
        }
        test_timestamps = pd.DatetimeIndex(pd.to_datetime(dataset["timestamp"].iloc[start:end], utc=True))
        close_test = pd.Series(dataset["close"].iloc[start:end].to_numpy(), index=test_timestamps, dtype=float)
        actual_test = pd.Series(fold["test_actual"], index=test_timestamps, dtype=int)
        prediction_test = pd.Series(fold["test_predictions"], index=test_timestamps, dtype=int)
        fold_funding_events = [event for event in (funding_events or [])
                               if test_timestamps[0] <= pd.Timestamp(event["timestamp"]) <= test_timestamps[-1]]
        evaluation = evaluate_predictions(
            close_test, actual_test, prediction_test,
            fee_bps=fee_bps, slippage_bps=slippage_bps,
            funding_bps_per_bar=funding_bps_per_bar,
            funding_events=fold_funding_events,
            allow_short=allow_short, bars_per_year=bars_per_year,
        )
        # Candidate-specific outer-test streams are net of the same costs and
        # expressed relative to the same no-trade benchmark.
        for candidate_name, candidate_metrics in fold["test_candidate_metrics"].items():
            candidate_prediction = pd.Series(
                candidate_metrics["test_predictions"], index=test_timestamps, dtype=int
            )
            candidate_evaluation = evaluate_predictions(
                close_test, actual_test, candidate_prediction,
                fee_bps=fee_bps, slippage_bps=slippage_bps,
                funding_bps_per_bar=funding_bps_per_bar,
                funding_events=fold_funding_events,
                allow_short=allow_short, bars_per_year=bars_per_year,
            )
            candidate_net = candidate_evaluation["trading"]["model"]["bar_returns"]
            benchmark_net = candidate_evaluation["trading"]["no_trade"]["bar_returns"]
            stream = candidate_return_streams[candidate_name]
            stream["timestamps"].extend(timestamp.isoformat() for timestamp in test_timestamps)
            stream["net_return_differentials"].extend(
                float(value - benchmark)
                for value, benchmark in zip(candidate_net, benchmark_net)
            )
        evaluation["timestamps"] = [timestamp.isoformat() for timestamp in test_timestamps]
        performance_folds.append({
            "fold": fold["fold"],
            "selected_candidate": fold["selected_candidate"],
            "market_regimes": regime_diagnostics,
            **evaluation,
        })
    report["candidate_oos_returns"] = {
        name: {
            "timestamps": stream["timestamps"],
            "net_return_differentials": stream["net_return_differentials"],
            "benchmark_id": "no_trade",
            "ledger_compatible": (
                len(stream["timestamps"]) == len(set(stream["timestamps"]))
                and len(stream["timestamps"]) >= 8
            ),
        }
        for name, stream in candidate_return_streams.items()
    }

    # Re-run the same frozen outer-test predictions under harsher execution costs.
    # These scenarios are sensitivity diagnostics, not additional model selection.
    cost_sensitivity = []
    for multiplier in (1.0, 1.5, 2.0):
        scenario_folds = []
        for fold in report["folds"]:
            start = fold["boundaries"]["test_start"]
            end = fold["boundaries"]["test_end"]
            test_timestamps = pd.DatetimeIndex(pd.to_datetime(dataset["timestamp"].iloc[start:end], utc=True))
            close_test = pd.Series(dataset["close"].iloc[start:end].to_numpy(), index=test_timestamps, dtype=float)
            actual_test = pd.Series(fold["test_actual"], index=test_timestamps, dtype=int)
            prediction_test = pd.Series(fold["test_predictions"], index=test_timestamps, dtype=int)
            fold_funding_events = [dict(event, rate_bps=float(event["rate_bps"]) * multiplier)
                                   for event in (funding_events or [])
                                   if test_timestamps[0] <= pd.Timestamp(event["timestamp"]) <= test_timestamps[-1]]
            scenario = evaluate_predictions(
                close_test, actual_test, prediction_test,
                fee_bps=fee_bps * multiplier,
                slippage_bps=slippage_bps * multiplier,
                funding_bps_per_bar=funding_bps_per_bar * multiplier,
                funding_events=fold_funding_events,
                allow_short=allow_short, bars_per_year=bars_per_year,
            )
            scenario_folds.append(scenario["trading"])
        model_scenario = [item["model"] for item in scenario_folds]
        buy_hold_scenario = [item["buy_and_hold"] for item in scenario_folds]
        no_trade_scenario = [item["no_trade"] for item in scenario_folds]
        cost_sensitivity.append({
            "cost_multiplier": multiplier,
            "fee_bps": float(fee_bps * multiplier),
            "slippage_bps": float(slippage_bps * multiplier),
            "mean_net_return": float(np.mean([item["total_return"] for item in model_scenario])),
            "median_net_return": float(np.median([item["total_return"] for item in model_scenario])),
            "positive_return_folds": int(sum(item["total_return"] > 0 for item in model_scenario)),
            "folds_beating_buy_and_hold": int(sum(
                model["total_return"] > benchmark["total_return"]
                for model, benchmark in zip(model_scenario, buy_hold_scenario)
            )),
            "folds_beating_no_trade": int(sum(
                model["total_return"] > benchmark["total_return"]
                for model, benchmark in zip(model_scenario, no_trade_scenario)
            )),
            "mean_max_drawdown": float(np.mean([item["max_drawdown"] for item in model_scenario])),
            "mean_sharpe_annualized": float(np.mean([item["sharpe_annualized"] for item in model_scenario])),
            "fold_count": len(model_scenario),
        })
    def mean_metric(strategy_name: str, metric_name: str) -> float:
        values = [item["trading"][strategy_name][metric_name] for item in performance_folds]
        return float(np.mean(values))
    model_returns = [item["trading"]["model"]["total_return"] for item in performance_folds]
    buy_hold_returns = [item["trading"]["buy_and_hold"]["total_return"] for item in performance_folds]
    no_trade_returns = [item["trading"]["no_trade"]["total_return"] for item in performance_folds]
    report["trading_performance"] = {
        "folds": performance_folds,
        "summary": {
            "mean_net_return": mean_metric("model", "total_return"),
            "mean_hypothetical_long_short_return": float(np.mean([
                item["trading"]["model_long_short"]["total_return"] for item in performance_folds
            ])),
            "mean_max_drawdown": mean_metric("model", "max_drawdown"),
            "mean_sharpe_annualized": mean_metric("model", "sharpe_annualized"),
            "mean_sortino_annualized": mean_metric("model", "sortino_annualized"),
            "mean_trade_count": mean_metric("model", "trade_count"),
            "mean_win_rate": mean_metric("model", "win_rate"),
            "mean_buy_and_hold_return": float(np.mean(buy_hold_returns)),
            "mean_no_trade_return": float(np.mean(no_trade_returns)),
            "folds_beating_buy_and_hold": int(sum(a > b for a, b in zip(model_returns, buy_hold_returns))),
            "folds_beating_no_trade": int(sum(a > b for a, b in zip(model_returns, no_trade_returns))),
            "fold_count": len(performance_folds),
        },
        "warning": "Fold metrics are averaged descriptively; overlapping test windows are dependent. Costs are illustrative assumptions, not exchange-specific quotes.",
    }
    report["cost_sensitivity"] = {
        "scenarios": cost_sensitivity,
        "note": "The same untouched outer-test predictions are re-simulated at 1.0x, 1.5x and 2.0x fee/slippage assumptions. This is a robustness diagnostic, not a forecast of actual execution costs.",
    }
    report.update({"rows": len(dataset), "horizon": horizon, "timeframe": timeframe,
                   "annualization": annualization,
                   "cost_assumptions": {"fee_bps": float(fee_bps), "slippage_bps": float(slippage_bps),
                                        "funding_bps_per_bar": float(funding_bps_per_bar),
                                        "funding_event_count": len(funding_events or []),
                                        "funding_model": "timestamped_settlement_events" if funding_events else "none_or_legacy_per_bar",
                                        "allow_short": bool(allow_short)},
                   "training_performed": True,
                   "execution_enabled": False,
                   "compute_budget": {
                       "estimated_model_fit_calls": estimated_model_fits,
                       "max_model_fit_calls": max_model_fits,
                       "fold_count": len(planned_folds),
                       "candidate_count": 2,
                       "note": "Fit-call count is a compute proxy, not a Kaggle dollar-cost estimate."
                   },
                   "model_note": "HistGradientBoostingClassifier is fitted from scratch within each fold; majority-class is a naive comparator."})
    training_rows = sorted({index for fold in planned_folds for index in range(fold.train_end)})
    evaluation_rows = sorted({
        index for fold in planned_folds for index in range(fold.test_start, fold.test_end)
    })
    def _rows_sha256(indices: list[int]) -> str:
        payload = pd.util.hash_pandas_object(dataset.iloc[indices], index=True).to_numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()
    report["trial_lineage"] = {
        "training_data_sha256": _rows_sha256(training_rows),
        "evaluation_data_sha256": _rows_sha256(evaluation_rows),
        "training_row_count": len(training_rows),
        "evaluation_row_count": len(evaluation_rows),
        "evaluation_protocol": "nested_purged_chronological_outer_test",
    }
    return report


def run_multi_symbol_experiment(frames: dict[str, pd.DataFrame], horizon: int = 6,
                                min_train: int = 500, validation_size: int = 100,
                                test_size: int = 100, step_size: int = 100,
                                timeframe: str = "1h",
                                max_model_fits: int | None = None) -> dict:
    """Run independent nested experiments per symbol; no cross-market leakage."""
    if not frames:
        raise ValueError("At least one symbol dataset is required")
    planned_fits = 0
    for frame in frames.values():
        clean_frame = validate_ohlcv(frame)
        feature_frame = build_features(clean_frame)
        feature_cols = feature_columns_for_frame(clean_frame)
        labels = make_labels(clean_frame, horizon=horizon)
        usable = feature_frame.join(labels.rename("label")).dropna(subset=feature_cols + ["label"])
        planned_fits += estimate_nested_model_fits_from_rows(
            len(usable), horizon, min_train, validation_size, test_size, step_size
        )
    if max_model_fits is not None and planned_fits > max_model_fits:
        raise ValueError(
            f"Multi-market training stopped before fitting: estimated {planned_fits} model fits "
            f"exceed the configured cap of {max_model_fits}."
        )
    reports = {}
    for symbol, frame in sorted(frames.items()):
        reports[symbol] = run_nested_experiment(
            frame, horizon, min_train, validation_size, test_size, step_size,
            timeframe=timeframe)
    return {"symbol_count": len(reports), "symbols": reports,
            "compute_budget": {
                "estimated_model_fit_calls": planned_fits,
                "max_model_fit_calls": max_model_fits,
                "note": "Fit-call count is a compute proxy, not a Kaggle dollar-cost estimate."
            },
            "warning": "Each market is evaluated independently. These results are not a portfolio backtest."}


def _timeframe_seconds(timeframe: str) -> float:
    """Parse common fixed-duration CCXT timeframes for annualized metrics."""
    try:
        import ccxt
        seconds = float(ccxt.Exchange.parse_timeframe(timeframe))
    except (ImportError, AttributeError, TypeError, ValueError):
        import re
        match = re.fullmatch(r"(\d+)([smhdw])", timeframe)
        if not match:
            raise ValueError(f"Unsupported fixed-duration timeframe: {timeframe}")
        amount, unit = int(match.group(1)), match.group(2)
        seconds = amount * {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
    if seconds <= 0:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    return seconds


def _reserve_final_export_holdout(frame: pd.DataFrame, save_model: bool) -> tuple[pd.DataFrame, dict]:
    """Keep the final chronological 20% out of all research evaluation when exporting."""
    if not save_model:
        return frame, {
            "enabled": False, "evaluation_rows": int(len(frame)),
            "reserved_rows": 0,
        }
    if len(frame) < 10:
        raise ValueError("Insufficient rows to reserve an independent final export holdout")
    cutoff = int(len(frame) * 0.80)
    if cutoff < 1 or cutoff >= len(frame):
        raise ValueError("Unable to construct a final chronological export holdout")
    return frame.iloc[:cutoff].copy(), {
        "enabled": True, "evaluation_rows": int(cutoff),
        "reserved_rows": int(len(frame) - cutoff),
        "reserved_fraction": float((len(frame) - cutoff) / len(frame)),
        "policy": "final_chronological_20_percent_excluded_from_all_prior_research_evaluation",
    }


def main():
    parser = argparse.ArgumentParser(description="Run ETI research-only baseline")
    parser.add_argument("--exchange", default="auto", choices=["auto", "binance", "mexc", "coinbaseexchange", "kraken"])
    parser.add_argument("--symbol", default="BTC/USDT", help="Single symbol")
    parser.add_argument("--symbols", help="Comma-separated symbols for independent nested evaluation")
    parser.add_argument("--evaluation", choices=["holdout", "nested"], default="holdout", help="nested performs repeated from-scratch candidate fitting")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--min-train", type=int, default=400)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--step-size", type=int, default=200)
    parser.add_argument("--max-model-fits", type=int, default=32,
                        help="Pre-fit cap for nested model fits; a compute proxy, not a dollar estimate")
    parser.add_argument("--csv", type=Path, help="Use a local OHLCV CSV instead of downloading")
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline_metrics.json"))
    parser.add_argument("--save-model", action="store_true",
                        help="explicitly fit and save one deployable model artifact after evaluation")
    parser.add_argument("--model-id", default="eti-baseline")
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument("--scan-universe", action="store_true",
                        help="discover a bounded multi-asset universe and check data readiness without training")
    parser.add_argument("--asset-classes", default="crypto,forex,equity,commodity,index",
                        help="comma-separated asset classes for --scan-universe")
    parser.add_argument("--market-types", default="spot",
                        help="comma-separated market types for --scan-universe")
    parser.add_argument("--max-per-asset-class", type=int, default=5,
                        help="candidate cap per asset class for --scan-universe")
    parser.add_argument("--max-per-venue", type=int, default=2,
                        help="candidate cap per venue within each asset class")
    parser.add_argument("--min-market-rows", type=int, default=662,
                        help="minimum valid candles required for a data-ready universe candidate")
    args = parser.parse_args()
    if args.save_model and args.max_model_fits < 2:
        parser.error("--save-model requires a fit cap of at least 2 (one final deployable fit plus evaluation)")
    nested_fit_cap = args.max_model_fits - (1 if args.save_model else 0)
    if args.scan_universe and (args.csv or args.symbols or args.save_model):
        parser.error("--scan-universe cannot be combined with --csv, --symbols or --save-model")
    if args.csv and args.symbols:
        parser.error("--csv cannot be combined with --symbols")
    if args.save_model and args.symbols:
        parser.error("--save-model currently requires one symbol or one CSV dataset")
    if args.scan_universe:
        asset_classes = [item.strip() for item in args.asset_classes.split(",") if item.strip()]
        market_types = [item.strip() for item in args.market_types.split(",") if item.strip()]
        metrics = scan_market_universe(
            asset_classes=asset_classes, market_types=market_types,
            timeframe=args.timeframe, limit=args.limit,
            min_rows=args.min_market_rows,
            max_per_asset_class=args.max_per_asset_class,
            max_per_venue=args.max_per_venue,
        )
    elif args.csv:
        frame = pd.read_csv(args.csv)
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        frame = drop_incomplete_candles(frame, int(_timeframe_seconds(args.timeframe) * 1000), now_ms)
        evaluation_frame, holdout_protocol = _reserve_final_export_holdout(frame, args.save_model)
        metrics = run_nested_experiment(
            evaluation_frame, horizon=args.horizon, min_train=args.min_train,
            validation_size=args.validation_size, test_size=args.test_size,
            step_size=args.step_size, timeframe=args.timeframe,
            max_model_fits=nested_fit_cap,
        ) if args.evaluation == "nested" else run_experiment(
            evaluation_frame, horizon=args.horizon, timeframe=args.timeframe
        )
        if args.save_model:
            metrics["independent_holdout_protocol"] = holdout_protocol
    elif args.symbols:
        symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
        if not symbols:
            parser.error("--symbols must contain at least one symbol")
        frames = {}
        source_records = {}
        for symbol in symbols:
            if args.exchange == "auto":
                frames[symbol], source_records[symbol] = fetch_ohlcv_auto(
                    symbol, args.timeframe, args.limit,
                    min_rows=args.min_train + args.validation_size + args.test_size + 2 * args.horizon + 50,
                )
            else:
                frames[symbol] = fetch_ohlcv(args.exchange, symbol, args.timeframe, args.limit)
                source_records[symbol] = {
                    "mode": "explicit", "provider": args.exchange,
                    "venue_symbol": symbol, "requested_symbol": symbol,
                    "returned_candles": int(len(frames[symbol])),
                }
        metrics = run_multi_symbol_experiment(
            frames, horizon=args.horizon, min_train=args.min_train,
            validation_size=args.validation_size, test_size=args.test_size,
            step_size=args.step_size, timeframe=args.timeframe,
            max_model_fits=nested_fit_cap,
        ) if args.evaluation == "nested" else {
            symbol: run_experiment(data, horizon=args.horizon, timeframe=args.timeframe)
            for symbol, data in frames.items()}
    else:
        if args.exchange == "auto":
            frame, source_record = fetch_ohlcv_auto(
                args.symbol, args.timeframe, args.limit,
                min_rows=args.min_train + args.validation_size + args.test_size + 2 * args.horizon + 50,
            )
        else:
            frame = fetch_ohlcv(args.exchange, args.symbol, args.timeframe, args.limit)
            source_record = {
                "mode": "explicit", "provider": args.exchange,
                "venue_symbol": args.symbol, "requested_symbol": args.symbol,
                "returned_candles": int(len(frame)),
            }
        evaluation_frame, holdout_protocol = _reserve_final_export_holdout(frame, args.save_model)
        metrics = run_nested_experiment(
            evaluation_frame, horizon=args.horizon, min_train=args.min_train,
            validation_size=args.validation_size, test_size=args.test_size,
            step_size=args.step_size, timeframe=args.timeframe,
            max_model_fits=nested_fit_cap,
        ) if args.evaluation == "nested" else run_experiment(
            evaluation_frame, horizon=args.horizon, timeframe=args.timeframe
        )
        if args.save_model:
            metrics["independent_holdout_protocol"] = holdout_protocol
    if not args.csv and not args.scan_universe:
        metrics["market_data"] = (
            {"mode": "multi_symbol", "sources": source_records}
            if args.symbols else source_record
        )
    if args.save_model:
        if args.csv:
            training_ref = "csv:" + args.csv.name
        else:
            actual_source = source_record if args.exchange == "auto" else {
                "provider": args.exchange, "venue_symbol": args.symbol
            }
            training_ref = (
                f"ccxt:{actual_source['provider']}:{actual_source['venue_symbol']}:"
                f"{args.timeframe}:{len(frame)}"
            )
        metrics["exported_model"] = export_deployable_model(
            frame, args.model_id, args.model_version, training_ref, horizon=args.horizon,
            timeframe=args.timeframe,
            holdout_reserved_from_prior_evaluation=True,
        )
        budget = metrics.setdefault("compute_budget", {})
        budget["deployable_model_fit_calls"] = 1
        budget["total_estimated_model_fit_calls"] = int(budget.get("estimated_model_fit_calls", 0)) + 1
        budget["max_model_fit_calls"] = args.max_model_fits
        metrics["warning"] = (
            str(metrics.get("warning", "")) +
            " Exported model is a candidate artifact only; it is not registered, "
            "approved, or enabled for autonomous trading."
        ).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
