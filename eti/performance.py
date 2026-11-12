"""Out-of-sample trading and classification diagnostics for research reports."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

from .backtest import backtest_positions


def _finite_nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _trade_statistics(position: pd.Series, bar_returns: pd.Series) -> dict[str, Any]:
    """Summarize contiguous directional holding episodes, including exit costs."""
    values = position.to_numpy(dtype=float)
    returns = bar_returns.to_numpy(dtype=float)
    trades: list[float] = []
    i = 0
    while i < len(values):
        if values[i] == 0:
            i += 1
            continue
        direction = values[i]
        start = i
        while i + 1 < len(values) and values[i + 1] == direction:
            i += 1
        end = i
        # Include the following bar's turnover cost when the position closes.
        stop = min(end + 2, len(returns))
        trade_return = float(np.prod(1.0 + returns[start:stop]) - 1.0)
        trades.append(trade_return)
        i += 1
    wins = [value for value in trades if value > 0]
    losses = [value for value in trades if value < 0]
    gross_profit = float(sum(wins))
    gross_loss = float(-sum(losses))
    return {
        "trade_count": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": float(len(wins) / len(trades)) if trades else 0.0,
        "average_trade_return": float(np.mean(trades)) if trades else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else (None if gross_profit > 0 else 0.0),
    }


def _strategy_metrics(close: pd.Series, signals: pd.Series, fee_bps: float,
                      slippage_bps: float, funding_bps_per_bar: float,
                      allow_short: bool, bars_per_year: float) -> dict[str, Any]:
    result = backtest_positions(
        close, signals, fee_bps=fee_bps, slippage_bps=slippage_bps,
        funding_bps_per_bar=funding_bps_per_bar, allow_short=allow_short,
    )
    equity = result.pop("equity_curve")
    position = signals.astype(float).shift(1).fillna(0.0)
    returns = close.astype(float).pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    cost = (fee_bps + slippage_bps) / 10000.0
    bar_returns = position * returns - turnover * cost - position.abs() * funding_bps_per_bar / 10000.0
    std = float(bar_returns.std(ddof=1))
    result["sharpe_annualized"] = float(bar_returns.mean() / std * math.sqrt(bars_per_year)) if std > 0 else 0.0
    result["sortino_annualized"] = float(
        bar_returns.mean() / float(np.sqrt(np.mean(np.minimum(bar_returns, 0.0) ** 2))) * math.sqrt(bars_per_year)
    ) if np.any(bar_returns.to_numpy() < 0) and np.mean(np.minimum(bar_returns, 0.0) ** 2) > 0 else 0.0
    result["turnover_fraction"] = float(turnover.sum())
    result["long_exposure"] = float(position.clip(lower=0).mean())
    result["short_exposure"] = float((-position.clip(upper=0)).mean())
    result.update(_trade_statistics(position, bar_returns))
    result["equity_curve"] = [float(value) for value in equity.to_numpy()]
    return result


def evaluate_predictions(
    close: pd.Series,
    actual: pd.Series,
    predictions: pd.Series,
    *,
    fee_bps: float = 6.0,
    slippage_bps: float = 2.0,
    funding_bps_per_bar: float = 0.0,
    allow_short: bool = True,
    bars_per_year: float = 365.25 * 24,
) -> dict[str, Any]:
    """Evaluate class decisions and net strategy performance on one untouched test window.

    Labels are 0=short, 1=flat, 2=long. Decisions at close t are applied
    from the next bar. Baselines use the exact same price window and costs.
    """
    fee_bps = _finite_nonnegative(fee_bps, "fee_bps")
    slippage_bps = _finite_nonnegative(slippage_bps, "slippage_bps")
    funding_bps_per_bar = _finite_nonnegative(funding_bps_per_bar, "funding_bps_per_bar")
    if isinstance(bars_per_year, bool) or not isinstance(bars_per_year, (int, float)) or not math.isfinite(bars_per_year) or bars_per_year <= 0:
        raise ValueError("bars_per_year must be finite and positive")
    if not isinstance(allow_short, bool):
        raise ValueError("allow_short must be a boolean")
    if not close.index.equals(actual.index) or not close.index.equals(predictions.index):
        raise ValueError("Close, actual labels and predictions indexes must match exactly")
    data = pd.concat([close.rename("close"), actual.rename("actual"), predictions.rename("prediction")], axis=1)
    if len(data) < 2 or data.isna().any().any():
        raise ValueError("At least two aligned, non-missing test observations are required")
    if not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError("Test index must be unique and chronological")
    prices = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=float)
    y_true = pd.to_numeric(data["actual"], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(data["prediction"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Test prices must be finite and positive")
    if not np.isin(y_true, [0, 1, 2]).all() or not np.isin(y_pred, [0, 1, 2]).all():
        raise ValueError("Actual labels and predictions must be class labels 0, 1 or 2")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true.astype(int), y_pred.astype(int), labels=[0, 1, 2], zero_division=0
    )
    confusion = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=[0, 1, 2])
    close_series = data["close"].astype(float)
    actual_series = data["actual"].astype(int)
    prediction_series = data["prediction"].astype(int)
    predicted_positions = prediction_series.astype(float) - 1.0
    actual_positions = actual_series.astype(float) - 1.0
    if not allow_short:
        predicted_positions = predicted_positions.clip(lower=0)
        actual_positions = actual_positions.clip(lower=0)
    strategy = _strategy_metrics(
        close_series, predicted_positions, fee_bps, slippage_bps,
        funding_bps_per_bar, allow_short, float(bars_per_year),
    )
    buy_hold = _strategy_metrics(
        close_series, pd.Series(1.0, index=close_series.index), fee_bps,
        slippage_bps, funding_bps_per_bar, False, float(bars_per_year),
    )
    no_trade = _strategy_metrics(
        close_series, pd.Series(0.0, index=close_series.index), fee_bps,
        slippage_bps, 0.0, False, float(bars_per_year),
    )
    actual_oracle = _strategy_metrics(
        close_series, actual_positions, fee_bps, slippage_bps,
        funding_bps_per_bar, allow_short, float(bars_per_year),
    )
    return {
        "classification": {
            "accuracy": float(np.mean(y_true == y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true.astype(int), y_pred.astype(int))),
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)),
            "per_class": {
                name: {"precision": float(precision[i]), "recall": float(recall[i]),
                       "f1": float(f1[i]), "support": int(support[i])}
                for i, name in enumerate(["short", "flat", "long"])
            },
            "confusion_matrix_labels": ["short", "flat", "long"],
            "confusion_matrix": confusion.astype(int).tolist(),
        },
        "trading": {
            "model": strategy,
            "buy_and_hold": buy_hold,
            "no_trade": no_trade,
            "label_oracle_upper_bound": actual_oracle,
            "cost_assumptions_bps": {
                "fee_per_unit_turnover": fee_bps,
                "slippage_per_unit_turnover": slippage_bps,
                "funding_per_bar": funding_bps_per_bar,
            },
            "note": "Label-oracle is a diagnostic upper bound, not an executable strategy. Bar-based fills omit queue position, partial fills and venue-specific mechanics.",
        },
        "warning": "Out-of-sample research metrics only; they do not establish future profitability.",
    }
