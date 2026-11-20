"""Temporal-integrity and deterministic-replay acceptance tests for Phase 1."""
import pandas as pd
import pytest

from eti.backtest import backtest_positions
from eti.data import drop_incomplete_candles


def _ohlcv_rows(open_times_ms):
    return pd.DataFrame({
        "timestamp": open_times_ms,
        "open": [100.0] * len(open_times_ms),
        "high": [102.0] * len(open_times_ms),
        "low": [99.0] * len(open_times_ms),
        "close": [101.0] * len(open_times_ms),
        "volume": [1000.0] * len(open_times_ms),
    })


def test_incomplete_current_candle_is_excluded_at_exact_close_boundary():
    hour = 60 * 60 * 1000
    frame = _ohlcv_rows([0, hour, 2 * hour])

    at_two = drop_incomplete_candles(frame, hour, now_ms=2 * hour)
    just_after_two = drop_incomplete_candles(frame, hour, now_ms=2 * hour + 1)

    assert at_two["timestamp"].tolist() == [
        pd.Timestamp(0, unit="ms", tz="UTC"),
        pd.Timestamp(hour, unit="ms", tz="UTC"),
    ]
    assert just_after_two["timestamp"].tolist() == [
        pd.Timestamp(0, unit="ms", tz="UTC"),
        pd.Timestamp(hour, unit="ms", tz="UTC"),
    ]


def test_completed_candle_is_included_only_after_its_full_interval():
    hour = 60 * 60 * 1000
    frame = _ohlcv_rows([0, hour, 2 * hour])

    result = drop_incomplete_candles(frame, hour, now_ms=3 * hour)

    assert result["timestamp"].tolist() == [
        pd.Timestamp(0, unit="ms", tz="UTC"),
        pd.Timestamp(hour, unit="ms", tz="UTC"),
        pd.Timestamp(2 * hour, unit="ms", tz="UTC"),
    ]


def test_temporal_validation_sorts_and_deduplicates_before_completion_filter():
    hour = 60 * 60 * 1000
    frame = _ohlcv_rows([2 * hour, hour, 0, hour])

    result = drop_incomplete_candles(frame, hour, now_ms=3 * hour)

    assert result["timestamp"].is_monotonic_increasing
    assert result["timestamp"].is_unique
    assert len(result) == 3


def test_backtest_replay_is_deterministic_for_identical_inputs():
    index = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
    close = pd.Series(
        [100.0, 101.0, 99.0, 102.0, 103.0, 100.0, 98.0, 101.0, 104.0, 102.0, 105.0, 106.0],
        index=index,
        name="close",
    )
    signals = pd.Series(
        [0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        index=index,
        name="signal",
    )

    first = backtest_positions(
        close, signals, fee_bps=6.0, slippage_bps=8.0, allow_short=True
    )
    replay = backtest_positions(
        close.copy(), signals.copy(), fee_bps=6.0, slippage_bps=8.0, allow_short=True
    )

    assert first["final_equity"] == pytest.approx(replay["final_equity"], abs=0.0)
    assert first["equity_curve"].equals(replay["equity_curve"])
    assert first["position_changes"] == replay["position_changes"]
    assert first["max_drawdown"] == pytest.approx(replay["max_drawdown"], abs=0.0)
