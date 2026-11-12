"""Chronological long/short/flat research backtester with explicit costs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_positions(
    close: pd.Series,
    signals: pd.Series,
    fee_bps: float = 6.0,
    slippage_bps: float = 2.0,
    funding_bps_per_bar: float = 0.0,
    allow_short: bool = False,
) -> dict:
    if min(fee_bps, slippage_bps, funding_bps_per_bar) < 0:
        raise ValueError("Costs must be non-negative")
    data = pd.concat([close.rename("close"), signals.rename("signal")], axis=1).dropna()
    if len(data) < 2 or (data.close <= 0).any():
        raise ValueError("Need at least two positive close prices")
    signal = data.signal.astype(float).clip(-1, 1)
    if not allow_short:
        signal = signal.clip(0, 1)
    position = signal.shift(1).fillna(0)
    returns = data.close.pct_change().fillna(0)
    turnover = position.diff().abs().fillna(position.abs())
    costs = (fee_bps + slippage_bps) / 10000 * turnover
    funding = position.abs() * funding_bps_per_bar / 10000
    strategy_returns = position * returns - costs - funding
    equity = (1 + strategy_returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    vol = strategy_returns.std()
    return {
        "bars": int(len(data)), "total_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float(dd.min()), "exposure": float(position.abs().mean()),
        "position_changes": int((turnover > 0).sum()),
        "sharpe_per_bar": float(strategy_returns.mean() / vol) if vol > 0 else 0.0,
        "equity_curve": equity,
    }
