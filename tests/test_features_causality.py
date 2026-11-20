"""Causality and prefix-invariance checks for ETI's feature pipeline."""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from eti.features import build_features


def _ohlcv(rows=180):
    rng = np.random.default_rng(20261119)
    returns = rng.normal(0.0002, 0.008, rows)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.004, 0.002, rows))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(7.0, 0.25, rows)
    index = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=index)


def test_future_candle_perturbations_do_not_change_prior_features():
    frame = _ohlcv()
    cutoff = 120
    original = build_features(frame)
    changed = frame.copy()
    for column in ("open", "high", "low", "close"):
        changed.iloc[cutoff:, changed.columns.get_loc(column)] *= 1.35
    changed_features = build_features(changed)
    assert_frame_equal(original.iloc[:cutoff], changed_features.iloc[:cutoff], check_exact=True)


def test_full_history_and_prefix_feature_values_match_at_prefix_end():
    frame = _ohlcv()
    cutoff = 130
    full = build_features(frame)
    prefix = build_features(frame.iloc[:cutoff])
    assert_frame_equal(full.iloc[:cutoff], prefix, check_exact=True)
