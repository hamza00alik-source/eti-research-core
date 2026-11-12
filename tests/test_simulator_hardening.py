import pandas as pd
import pytest

from eti.backtest import backtest_positions
from eti.simulator import simulate_long_flat


def test_backtest_rejects_duplicate_or_nonchronological_index():
    close = pd.Series([100.0, 101.0, 102.0], index=[0, 0, 2])
    signals = pd.Series([0.0, 1.0, 1.0], index=[0, 0, 2])
    with pytest.raises(ValueError):
        backtest_positions(close, signals)


def test_backtest_rejects_non_boolean_short_flag():
    with pytest.raises(ValueError):
        backtest_positions(pd.Series([100.0, 101.0]), pd.Series([0.0, 1.0]),
                           allow_short="yes")


def test_simulator_rejects_invalid_predictions_and_prices():
    with pytest.raises(ValueError):
        simulate_long_flat(pd.Series([100.0, 101.0]), pd.Series([2, 4]))
    with pytest.raises(ValueError):
        simulate_long_flat(pd.Series([100.0, 0.0]), pd.Series([0, 2]))
