"""Deliberately simple long-only paper simulator; not exchange-grade execution."""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_long_flat(
    close: pd.Series,
    predictions: pd.Series,
    fee_bps: float = 6.0,
    slippage_bps: float = 2.0,
) -> dict:
    """Apply close-time decisions on the next bar and charge turnover costs.

    Prediction 2 means long; 0/1 means flat in this baseline. Returns are
    close-to-close and therefore approximate, not tick-accurate fills.
    """
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("Costs cannot be negative")
    aligned = pd.concat([close.rename("close"), predictions.rename("prediction")], axis=1).dropna()
    if len(aligned) < 2:
        raise ValueError("At least two aligned observations are required")
    desired = (aligned["prediction"].astype(int) == 2).astype(float)
    position = desired.shift(1).fillna(0.0)
    asset_returns = aligned["close"].pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    cost_rate = (fee_bps + slippage_bps) / 10000.0
    strategy_returns = position * asset_returns - turnover * cost_rate
    equity = (1.0 + strategy_returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    years = max(len(strategy_returns) / (365.25 * 24), 1 / (365.25 * 24))
    total_return = float(equity.iloc[-1] - 1.0)
    return {
        "bars": int(len(aligned)),
        "total_return": total_return,
        "max_drawdown": float(drawdown.min()),
        "exposure": float(position.mean()),
        "trades": int((turnover > 0).sum()),
        "sharpe_bar_annualized": float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(365.25*24)) if strategy_returns.std() > 0 else 0.0,
        "equity_curve": equity,
    }
