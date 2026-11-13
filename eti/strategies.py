"""Transparent long-only baseline strategies for research and comparison."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def moving_average_signal(close: pd.Series, fast: int = 10, slow: int = 30) -> pd.Series:
    if fast < 2 or slow <= fast:
        raise ValueError("Require 2 <= fast < slow")
    f, s = close.rolling(fast).mean(), close.rolling(slow).mean()
    return pd.Series((f > s).astype(int), index=close.index).where(s.notna(), 0)


def momentum_signal(close: pd.Series, lookback: int = 12) -> pd.Series:
    if lookback < 1:
        raise ValueError("lookback must be positive")
    return (close.pct_change(lookback) > 0).astype(int).where(close.shift(lookback).notna(), 0)


def donchian_breakout_signal(close: pd.Series, window: int = 20) -> pd.Series:
    """Enter above the prior-window high; exit below the prior-window low.

    Both channel boundaries exclude the current bar. The backtester shifts the
    resulting close-time signal by one bar before applying market returns.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer >= 2")
    upper = close.shift(1).rolling(window, min_periods=window).max()
    lower = close.shift(1).rolling(window, min_periods=window).min()
    values = close.to_numpy(dtype=float)
    highs, lows = upper.to_numpy(dtype=float), lower.to_numpy(dtype=float)
    signal = np.zeros(len(close), dtype=float)
    active = False
    for i in range(len(close)):
        if not math.isfinite(highs[i]) or not math.isfinite(lows[i]):
            continue
        if active and values[i] < lows[i]:
            active = False
        elif not active and values[i] > highs[i]:
            active = True
        signal[i] = 1.0 if active else 0.0
    return pd.Series(signal, index=close.index)


def mean_reversion_signal(
    close: pd.Series, window: int = 20, entry_z: float = 1.5, exit_z: float = 0.0
) -> pd.Series:
    """Long when price is sufficiently below its rolling mean; exit at mean."""
    if isinstance(window, bool) or not isinstance(window, int) or window < 3:
        raise ValueError("window must be an integer >= 3")
    if (isinstance(entry_z, bool) or not isinstance(entry_z, (int, float))
            or not math.isfinite(entry_z) or entry_z <= 0):
        raise ValueError("entry_z must be finite and positive")
    if (isinstance(exit_z, bool) or not isinstance(exit_z, (int, float))
            or not math.isfinite(exit_z) or exit_z >= entry_z):
        raise ValueError("exit_z must be finite and less than entry_z")
    mean = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0).replace(0, np.nan)
    zscore = (close - mean) / std
    signal = np.zeros(len(close), dtype=float)
    active = False
    for i, z in enumerate(zscore.to_numpy(dtype=float)):
        if not math.isfinite(z):
            continue
        if active and z >= exit_z:
            active = False
        elif not active and z <= -entry_z:
            active = True
        signal[i] = 1.0 if active else 0.0
    return pd.Series(signal, index=close.index)


def buy_and_hold_signal(close: pd.Series) -> pd.Series:
    return pd.Series(1, index=close.index, dtype="int64")
