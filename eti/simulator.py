"""Deliberately simple long-only paper simulator; not exchange-grade execution."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def simulate_long_flat(close: pd.Series, predictions: pd.Series,
    fee_bps: float = 6.0, slippage_bps: float = 2.0,
    bars_per_year: float = 365.25 * 24) -> dict:
    """Apply close-time decisions on the next bar and charge turnover costs."""
    numeric = (fee_bps, slippage_bps, bars_per_year)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in numeric):
        raise ValueError("Costs and bars_per_year must be finite numbers")
    if fee_bps < 0 or slippage_bps < 0 or bars_per_year <= 0:
        raise ValueError("Costs cannot be negative and bars_per_year must be positive")
    if not close.index.equals(predictions.index):
        raise ValueError("Close and predictions indexes must match exactly")
    aligned = pd.concat([close.rename("close"), predictions.rename("prediction")], axis=1)
    if len(aligned) < 2 or aligned.isna().any().any():
        raise ValueError("At least two aligned, non-missing observations are required")
    prices = pd.to_numeric(aligned["close"], errors="coerce").to_numpy(dtype=float)
    raw_predictions = pd.to_numeric(aligned["prediction"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Close prices must be finite and positive")
    if not np.isfinite(raw_predictions).all() or not np.isin(raw_predictions, [0, 1, 2]).all():
        raise ValueError("Predictions must be class labels 0, 1 or 2")
    desired = (raw_predictions == 2).astype(float)
    position = pd.Series(desired, index=aligned.index).shift(1).fillna(0.0)
    asset_returns = pd.Series(prices, index=aligned.index).pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    cost_rate = (fee_bps + slippage_bps) / 10000.0
    strategy_returns = position * asset_returns - turnover * cost_rate
    if (strategy_returns <= -1).any():
        raise ValueError("A bar loss of 100% or more makes equity non-positive")
    equity = (1.0 + strategy_returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    bar_std = strategy_returns.std(ddof=1)
    sharpe = float(strategy_returns.mean() / bar_std * np.sqrt(bars_per_year)) if bar_std > 0 else 0.0
    return {"bars": int(len(aligned)), "total_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float(drawdown.min()), "exposure": float(position.mean()),
        "position_changes": int((turnover > 0).sum()), "bars_per_year": float(bars_per_year),
        "sharpe_annualized": sharpe, "equity_curve": equity}
