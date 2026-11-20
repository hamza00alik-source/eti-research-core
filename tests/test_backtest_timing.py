"""Tests that historical backtests cannot consume same-bar future returns."""
import pandas as pd

from eti.backtest import backtest_positions


def test_signal_is_exposed_only_to_the_following_bar_return():
    close = pd.Series([100.0, 110.0, 121.0, 121.0],
                      index=pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"))
    signals = pd.Series([1.0, 0.0, 0.0, 0.0], index=close.index)
    result = backtest_positions(close, signals, fee_bps=0.0, slippage_bps=0.0)
    assert result["final_equity"] == 1.1


def test_future_signal_changes_cannot_rewrite_earlier_equity():
    index = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
    close = pd.Series([100.0, 101.0, 103.0, 99.0, 105.0], index=index)
    base = pd.Series([0.0, 1.0, 0.0, 0.0, 0.0], index=index)
    changed = base.copy()
    changed.iloc[3:] = [1.0, 1.0]
    a = backtest_positions(close, base, fee_bps=0, slippage_bps=0)
    b = backtest_positions(close, changed, fee_bps=0, slippage_bps=0)
    assert a["equity_curve"].iloc[:3].equals(b["equity_curve"].iloc[:3])
