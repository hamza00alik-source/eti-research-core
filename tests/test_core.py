import numpy as np
import pandas as pd
import pytest

from eti.data import validate_ohlcv
from eti.features import build_features, make_labels
from eti.simulator import simulate_long_flat


def sample_ohlcv(n=400):
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = close * 0.003
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "open": open_, "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread, "close": close,
        "volume": rng.uniform(10, 100, n),
    })


def test_validation_sorts_and_deduplicates():
    raw = sample_ohlcv()
    raw = pd.concat([raw.iloc[[2]], raw.iloc[[0]], raw.iloc[[2]]], ignore_index=True)
    clean = validate_ohlcv(raw)
    assert len(clean) == 2
    assert clean.timestamp.is_monotonic_increasing


def test_features_are_causal_at_current_row():
    raw = sample_ohlcv()
    base = build_features(raw)
    changed = raw.copy()
    changed.loc[250:, ["open", "high", "low", "close", "volume"]] *= 1.5
    altered = build_features(changed)
    pd.testing.assert_series_equal(base.loc[249], altered.loc[249], check_names=False)


def test_labels_have_expected_classes_and_tail_nan():
    raw = sample_ohlcv()
    labels = make_labels(raw, horizon=5)
    assert labels.iloc[-5:].isna().all()
    assert set(labels.dropna().unique()).issubset({0, 1, 2})


def test_simulator_applies_signal_next_bar_and_costs():
    close = pd.Series([100.0, 110.0, 121.0, 121.0])
    predictions = pd.Series([2, 2, 1, 1])
    result = simulate_long_flat(close, predictions, fee_bps=0, slippage_bps=0)
    assert result["exposure"] == pytest.approx(0.75)
    assert result["total_return"] == pytest.approx(0.21)
