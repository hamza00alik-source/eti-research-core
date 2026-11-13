"""End-to-end baseline experiment. Run: python -m eti.pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from .data import drop_incomplete_candles, fetch_ohlcv, validate_ohlcv
from .features import FEATURE_COLUMNS, build_features, make_labels
from .model import BaselineModel
from .artifacts import ModelArtifactStore
from .simulator import simulate_long_flat
from .performance import evaluate_predictions
from .regimes import classify_market_regime


def run_experiment(frame: pd.DataFrame, horizon: int = 6, timeframe: str = "1h") -> dict:
    frame = validate_ohlcv(frame)
    features = build_features(frame)
    labels = make_labels(frame, horizon=horizon)
    dataset = features.join(labels.rename("label")).dropna()
    if len(dataset) < 200:
        raise ValueError(f"Need at least 200 usable rows; got {len(dataset)}")
    x, y = dataset[FEATURE_COLUMNS], dataset["label"].astype(int)
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
    timeframe_seconds = _timeframe_seconds(timeframe)
    bars_per_year = 365.25 * 24 * 60 * 60 / timeframe_seconds
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
        "warning": "Single chronological holdout baseline with horizon purging. Not evidence of profitability or future performance.",
    }



def fit_deployable_model(frame: pd.DataFrame, horizon: int = 6) -> tuple[BaselineModel, dict]:
    """Fit on earlier rows and calibrate only on a later, held-out time segment.

    The final export is opt-in. The calibration window is never used to fit the
    classifier, preventing in-sample confidence estimates from being treated as
    calibration evidence.
    """
    frame = validate_ohlcv(frame)
    features = build_features(frame)
    labels = make_labels(frame, horizon=horizon)
    dataset = features.join(labels.rename("label")).dropna()
    if len(dataset) < 250:
        raise ValueError(
            f"Need at least 250 usable rows for a 200-row fit and 30-row calibration; got {len(dataset)}"
        )
    split = int(len(dataset) * 0.8)
    training_end = split - int(horizon)
    training = dataset.iloc[:training_end]
    calibration = dataset.iloc[split:]
    purged_rows = split - training_end
    if len(training) < 200 or len(calibration) < 30:
        raise ValueError(
            "Export split must contain at least 200 training rows and 30 calibration rows "
            "after purging the label horizon"
        )

    x_train = training[FEATURE_COLUMNS]
    y_train = training["label"].astype(int)
    model = BaselineModel().fit(x_train, y_train)
    calibration_report = model.fit_temperature_calibration(
        calibration[FEATURE_COLUMNS], calibration["label"].astype(int)
    )
    dataset_sha256 = hashlib.sha256(
        pd.util.hash_pandas_object(dataset, index=True).values.tobytes()
    ).hexdigest()
    schema = {
        "features": list(FEATURE_COLUMNS),
        "dataset_sha256": dataset_sha256,
        "feature_version": "eti.features.v1",
        "training_rows": int(len(training)),
        "calibration_rows": int(len(calibration)),
        "purged_rows": int(purged_rows),
        "training_end": training.index[-1].isoformat() if hasattr(training.index[-1], "isoformat") else str(training.index[-1]),
        "calibration_start": calibration.index[0].isoformat() if hasattr(calibration.index[0], "isoformat") else str(calibration.index[0]),
        "calibration_end": calibration.index[-1].isoformat() if hasattr(calibration.index[-1], "isoformat") else str(calibration.index[-1]),
        "label_horizon_bars": int(horizon),
        "calibration_status": model.calibration_status,
        "calibration_report": calibration_report,
    }
    return model, schema


def export_deployable_model(
    frame: pd.DataFrame, model_id: str, version: str, training_data_ref: str,
    horizon: int = 6,
) -> dict:
    """Explicitly fit and store an immutable artifact; never called at import/startup."""
    model, input_schema = fit_deployable_model(frame, horizon=horizon)
    artifact = ModelArtifactStore().save_model(model)
    return {
        "model_id": model_id,
        "version": version,
        **artifact,
        "input_schema": input_schema,
        "supported_classes": ["crypto_spot"],
        "training_data_ref": training_data_ref,
        "calibration_status": model.calibration_status,
        "calibration_report": model.calibration_report,
        "note": "Candidate artifact only. Review evaluation and calibration evidence, then obtain independent approval before inference.",
    }


def run_nested_experiment(frame: pd.DataFrame, horizon: int = 6,
                          min_train: int = 500, validation_size: int = 100,
                          test_size: int = 100, step_size: int = 100,
                          timeframe: str = "1h", allow_short: bool = False,
                          funding_bps_per_bar: float = 0.0,
                          fee_bps: float = 6.0, slippage_bps: float = 2.0) -> dict:
    """Train candidates inside purged folds; configure long/short and explicit derivative cost assumptions."""
    from sklearn.dummy import DummyClassifier
    from .evaluation import run_nested_walk_forward
    frame = validate_ohlcv(frame)
    features = build_features(frame)
    labels = make_labels(frame, horizon=horizon)
    dataset = features.join(labels.rename("label")).dropna()
    dataset["close"] = frame.loc[dataset.index, "close"].astype(float)
    if len(dataset) < min_train + validation_size + test_size + 2 * horizon:
        raise ValueError("Insufficient completed data for nested walk-forward training")
    def fit_baseline(train_x, train_y, eval_x):
        return BaselineModel().fit(train_x, train_y).predict(eval_x)
    def fit_majority(train_x, train_y, eval_x):
        return DummyClassifier(strategy="most_frequent").fit(train_x, train_y).predict(eval_x)
    report = run_nested_walk_forward(
        dataset, FEATURE_COLUMNS, "label",
        {"hist_gradient_boosting": fit_baseline, "majority_class": fit_majority},
        min_train=min_train, validation_size=validation_size,
        test_size=test_size, step_size=step_size, purge_bars=horizon)
    bars_per_year = 365.25 * 24 * 60 * 60 / _timeframe_seconds(timeframe)
    # Causal regimes are aligned to the exact cleaned dataset rows.
    regime_frame = classify_market_regime(frame)
    regime_by_row = regime_frame.loc[dataset.index, "regime"].reset_index(drop=True)
    performance_folds = []
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
        close_test = dataset["close"].iloc[start:end].reset_index(drop=True)
        actual_test = pd.Series(fold["test_actual"], dtype=int)
        prediction_test = pd.Series(fold["test_predictions"], dtype=int)
        performance_folds.append({
            "fold": fold["fold"],
            "selected_candidate": fold["selected_candidate"],
            "market_regimes": regime_diagnostics,
            **evaluate_predictions(
                close_test, actual_test, prediction_test,
                fee_bps=fee_bps, slippage_bps=slippage_bps,
                funding_bps_per_bar=funding_bps_per_bar,
                allow_short=allow_short, bars_per_year=bars_per_year,
            ),
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
    report.update({"rows": len(dataset), "horizon": horizon, "timeframe": timeframe,
                   "cost_assumptions": {"fee_bps": float(fee_bps), "slippage_bps": float(slippage_bps),
                                        "funding_bps_per_bar": float(funding_bps_per_bar),
                                        "allow_short": bool(allow_short)},
                   "training_performed": True,
                   "model_note": "HistGradientBoostingClassifier is fitted from scratch within each fold; majority-class is a naive comparator."})
    return report


def run_multi_symbol_experiment(frames: dict[str, pd.DataFrame], horizon: int = 6,
                                min_train: int = 500, validation_size: int = 100,
                                test_size: int = 100, step_size: int = 100,
                                timeframe: str = "1h") -> dict:
    """Run independent nested experiments per symbol; no cross-market leakage."""
    if not frames:
        raise ValueError("At least one symbol dataset is required")
    reports = {}
    for symbol, frame in sorted(frames.items()):
        reports[symbol] = run_nested_experiment(
            frame, horizon, min_train, validation_size, test_size, step_size,
            timeframe=timeframe)
    return {"symbol_count": len(reports), "symbols": reports,
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


def main():
    parser = argparse.ArgumentParser(description="Run ETI research-only baseline")
    parser.add_argument("--exchange", default="binance", choices=["binance", "mexc"])
    parser.add_argument("--symbol", default="BTC/USDT", help="Single symbol")
    parser.add_argument("--symbols", help="Comma-separated symbols for independent nested evaluation")
    parser.add_argument("--evaluation", choices=["holdout", "nested"], default="holdout", help="nested performs repeated from-scratch candidate fitting")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--csv", type=Path, help="Use a local OHLCV CSV instead of downloading")
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline_metrics.json"))
    parser.add_argument("--save-model", action="store_true",
                        help="explicitly fit and save one deployable model artifact after evaluation")
    parser.add_argument("--model-id", default="eti-baseline")
    parser.add_argument("--model-version", default="0.1.0")
    args = parser.parse_args()
    if args.csv and args.symbols:
        parser.error("--csv cannot be combined with --symbols")
    if args.save_model and args.symbols:
        parser.error("--save-model currently requires one symbol or one CSV dataset")
    if args.csv:
        frame = pd.read_csv(args.csv)
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        frame = drop_incomplete_candles(frame, int(_timeframe_seconds(args.timeframe) * 1000), now_ms)
        metrics = run_nested_experiment(frame, timeframe=args.timeframe) if args.evaluation == "nested" else run_experiment(frame, timeframe=args.timeframe)
    elif args.symbols:
        symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
        if not symbols:
            parser.error("--symbols must contain at least one symbol")
        frames = {symbol: fetch_ohlcv(args.exchange, symbol, args.timeframe, args.limit)
                  for symbol in symbols}
        metrics = run_multi_symbol_experiment(frames, timeframe=args.timeframe) if args.evaluation == "nested" else {
            symbol: run_experiment(data, timeframe=args.timeframe)
            for symbol, data in frames.items()}
    else:
        frame = fetch_ohlcv(args.exchange, args.symbol, args.timeframe, args.limit)
        metrics = run_nested_experiment(frame) if args.evaluation == "nested" else run_experiment(frame, timeframe=args.timeframe)
    if args.save_model:
        if args.csv:
            training_ref = "csv:" + args.csv.name
        else:
            training_ref = f"ccxt:{args.exchange}:{args.symbol}:{args.timeframe}:{args.limit}"
        metrics["exported_model"] = export_deployable_model(
            frame, args.model_id, args.model_version, training_ref
        )
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
