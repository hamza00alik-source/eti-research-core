"""Chronological long/short/flat research backtester with explicit proportional costs."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def backtest_positions(
    close: pd.Series,
    signals: pd.Series,
    fee_bps: float = 6.0,
    slippage_bps: float = 2.0,
    funding_bps_per_bar: float = 0.0,
    allow_short: bool = False,
    initial_equity: float = 1.0,
) -> dict:
    costs_input = (fee_bps, slippage_bps, funding_bps_per_bar, initial_equity)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in costs_input):
        raise ValueError("Costs and initial equity must be finite numbers")
    if min(fee_bps, slippage_bps, funding_bps_per_bar) < 0 or initial_equity <= 0:
        raise ValueError("Costs must be non-negative and initial equity positive")
    if not isinstance(allow_short, bool):
        raise ValueError("allow_short must be a boolean")
    if not close.index.equals(signals.index):
        raise ValueError("Close and signal indexes must match exactly")
    data = pd.concat([close.rename("close"), signals.rename("signal")], axis=1)
    if data.isna().any().any() or len(data) < 2:
        raise ValueError("Need at least two aligned, non-missing close prices and signals")
    if not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError("Backtest index must be unique and chronological")
    data = data.astype(float)
    if not np.isfinite(data.to_numpy()).all() or (data.close <= 0).any():
        raise ValueError("Prices and signals must be finite; prices must be positive")
    if (data.signal.abs() > 1).any():
        raise ValueError("Signals must be in [-1, 1]")
    signal = data.signal.clip(-1, 1)
    if not allow_short:
        signal = signal.clip(0, 1)
    # A signal formed on bar t is first exposed to the return from t to t+1.
    position = signal.shift(1).fillna(0)
    returns = data.close.pct_change().fillna(0)
    turnover = position.diff().abs().fillna(position.abs())
    trading_cost = (fee_bps + slippage_bps) / 10000 * turnover
    funding_cost = position.abs() * funding_bps_per_bar / 10000
    strategy_returns = position * returns - trading_cost - funding_cost
    if (strategy_returns <= -1).any():
        raise ValueError("A bar loss of 100% or more makes equity non-positive")
    equity = initial_equity * (1 + strategy_returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1
    volatility = strategy_returns.std(ddof=1)
    return {
        "bars": int(len(data)),
        "initial_equity": float(initial_equity),
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / initial_equity - 1),
        "max_drawdown": float(drawdown.min()),
        "exposure": float(position.abs().mean()),
        "position_changes": int((turnover > 0).sum()),
        "trading_cost_total": float(trading_cost.sum()),
        "funding_cost_total": float(funding_cost.sum()),
        "sharpe_per_bar": float(strategy_returns.mean() / volatility) if volatility > 0 else 0.0,
        "equity_curve": equity,
    }
