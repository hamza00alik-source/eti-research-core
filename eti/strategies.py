"""Small transparent baseline strategies used for benchmarking, not recommendations."""
from __future__ import annotations

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


def buy_and_hold_signal(close: pd.Series) -> pd.Series:
    return pd.Series(1, index=close.index, dtype="int64")
