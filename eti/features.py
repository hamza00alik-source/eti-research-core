"""Causal feature engineering and forward-return classification labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "volatility_20", "sma_gap_10",
    "sma_gap_30", "rsi_14", "range_pct", "body_pct",
    "close_location", "volume_z20", "atr_pct",
]


def feature_columns_for_frame(frame: pd.DataFrame) -> list[str]:
    """Select only features backed by data actually present in this market frame."""
    columns = list(BASE_FEATURE_COLUMNS)
    if "funding_rate" in frame.columns:
        columns.append("funding_rate")
    if "open_interest" in frame.columns:
        columns.append("open_interest_change")
    return columns


# Backwards-compatible alias for callers that explicitly require the base spot schema.
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute features using current/past candles only; preserves input index."""
    required = {"open", "high", "low", "close", "volume"}
    if required - set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required-set(frame.columns))}")
    d = frame.copy()
    close = pd.to_numeric(d["close"], errors="coerce")
    high, low = d["high"].astype(float), d["low"].astype(float)
    opn, vol = d["open"].astype(float), d["volume"].astype(float)
    logret = np.log(close).diff()
    sma10, sma30 = close.rolling(10).mean(), close.rolling(30).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((gain == 0) & (loss > 0), 0.0)
    rsi = rsi.mask((gain == 0) & (loss == 0), 50.0)
    prev_close = close.shift(1)
    true_range = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    vol_mean, vol_std = vol.rolling(20).mean(), vol.rolling(20).std()
    span = (high-low).replace(0, np.nan)
    out = pd.DataFrame(index=d.index)
    out["ret_1"] = close.pct_change()
    out["ret_3"] = close.pct_change(3)
    out["ret_6"] = close.pct_change(6)
    out["volatility_20"] = logret.rolling(20).std()
    out["sma_gap_10"] = close / sma10 - 1
    out["sma_gap_30"] = close / sma30 - 1
    out["rsi_14"] = rsi
    out["range_pct"] = (high-low) / close
    out["body_pct"] = (close-opn) / opn
    out["close_location"] = (close-low) / span
    out["volume_z20"] = (vol-vol_mean) / vol_std.replace(0, np.nan)
    out["atr_pct"] = atr / close
    if "funding_rate" in d.columns:
        out["funding_rate"] = pd.to_numeric(d["funding_rate"], errors="coerce")
    if "open_interest" in d.columns:
        open_interest = pd.to_numeric(d["open_interest"], errors="coerce")
        out["open_interest_change"] = open_interest.pct_change().replace([np.inf, -np.inf], np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def make_labels(frame: pd.DataFrame, horizon: int = 6, threshold: float = 0.5) -> pd.Series:
    """Labels: 0=short, 1=flat, 2=long based on forward return vs past volatility."""
    if horizon < 1 or threshold < 0:
        raise ValueError("horizon must be >=1 and threshold must be >=0")
    close = frame["close"].astype(float)
    vol = np.log(close).diff().rolling(20).std()
    forward_return = close.shift(-horizon) / close - 1
    barrier = threshold * vol * np.sqrt(horizon)
    labels = pd.Series(1, index=frame.index, dtype="float64")
    labels.loc[forward_return > barrier] = 2
    labels.loc[forward_return < -barrier] = 0
    labels.loc[forward_return.isna() | barrier.isna()] = np.nan
    return labels
