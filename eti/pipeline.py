"""End-to-end baseline experiment. Run: python -m eti.pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from .data import fetch_ohlcv, validate_ohlcv
from .features import FEATURE_COLUMNS, build_features, make_labels
from .model import BaselineModel
from .simulator import simulate_long_flat


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
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--csv", type=Path, help="Use a local OHLCV CSV instead of downloading")
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline_metrics.json"))
    args = parser.parse_args()
    frame = pd.read_csv(args.csv) if args.csv else fetch_ohlcv(args.exchange, args.symbol, args.timeframe, args.limit)
    metrics = run_experiment(frame, timeframe=args.timeframe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
