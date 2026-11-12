"""Compare candidate signals on the same chronological price sample."""
from __future__ import annotations

import pandas as pd

from .backtest import backtest_positions
from .strategies import buy_and_hold_signal, momentum_signal, moving_average_signal


def compare_baselines(close: pd.Series) -> list[dict]:
    strategies = {
        "buy_and_hold": buy_and_hold_signal(close),
        "ma_10_30": moving_average_signal(close, 10, 30),
        "momentum_12": momentum_signal(close, 12),
    }
    rows = []
    for name, signal in strategies.items():
        metrics = backtest_positions(close, signal)
        metrics.pop("equity_curve", None)
        rows.append({"strategy": name, **metrics})
    return sorted(rows, key=lambda x: x["total_return"], reverse=True)
