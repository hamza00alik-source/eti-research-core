"""End-to-end baseline experiment. Run: python -m eti.pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from .data import drop_incomplete_candles, fetch_ohlcv, validate_ohlcv
from .features import FEATURE_COLUMNS, build_features, make_labels
from .model import BaselineModel
from .simulator import simulate_long_flat
from .performance import evaluate_predictions


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



def run_nested_experiment(frame: pd.DataFrame, horizon: int = 6,
                          min_train: int = 500, validation_size: int = 100,
                          test_size: int = 100, step_size: int = 100,
                          timeframe: str = "1h") -> dict:
    """Train-from-scratch candidate models inside purged nested walk-forward folds."""
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
    performance_folds = []
    for fold in report["folds"]:
        start, end = fold["boundaries"]["test_start"], fold["boundaries"]["test_end"]
        close_test = dataset["close"].iloc[start:end].reset_index(drop=True)
        actual_test = pd.Series(fold["test_actual"], dtype=int)
        prediction_test = pd.Series(fold["test_predictions"], dtype=int)
        performance_folds.append({
            "fold": fold["fold"],
            "selected_candidate": fold["selected_candidate"],
            **evaluate_predictions(
                close_test, actual_test, prediction_test,
                fee_bps=6.0, slippage_bps=2.0, funding_bps_per_bar=0.0,
                allow_short=True, bars_per_year=bars_per_year,
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
    args = parser.parse_args()
    if args.csv and args.symbols:
        parser.error("--csv cannot be combined with --symbols")
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
