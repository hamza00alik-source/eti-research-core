import numpy as np
import pandas as pd
import pytest

from eti.strategies import donchian_breakout_signal, mean_reversion_signal


def test_donchian_channel_uses_only_prior_bars():
    close = pd.Series([10, 11, 10, 10, 12, 13, 11, 10, 9, 11], dtype=float)
    signal = donchian_breakout_signal(close, window=3)
    # A fresh close above the previous three-bar high enters on that close.
    assert signal.iloc[4] == 1.0
    # The signal is bounded and index-aligned.
    assert signal.index.equals(close.index)
    assert set(signal.unique()).issubset({0.0, 1.0})


def test_donchian_signal_does_not_look_ahead():
    close = pd.Series([10, 10, 10, 10, 11, 11, 11, 12], dtype=float)
    baseline = donchian_breakout_signal(close, window=3)
    extended = pd.concat([close, pd.Series([1000.0])], ignore_index=True)
    changed_future = donchian_breakout_signal(extended, window=3)
    assert np.array_equal(baseline.to_numpy(), changed_future.iloc[:len(close)].to_numpy())


def test_mean_reversion_signal_is_long_only_and_validates_parameters():
    close = pd.Series(np.r_[np.full(30, 100.0), np.linspace(100, 80, 15), np.linspace(80, 105, 20)])
    signal = mean_reversion_signal(close, window=10, entry_z=1.5, exit_z=0.0)
    assert set(signal.unique()).issubset({0.0, 1.0})
    with pytest.raises(ValueError, match="entry_z"):
        mean_reversion_signal(close, entry_z=-1)
