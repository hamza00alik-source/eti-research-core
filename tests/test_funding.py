import pandas as pd
import pytest

from eti.backtest import backtest_positions
from eti.funding import funding_cost_series
from eti.performance import evaluate_predictions


def _times():
    return pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")


def test_funding_event_is_charged_once_at_exact_settlement_bar():
    index = _times()
    close = pd.Series([100.0] * 4, index=index)
    signals = pd.Series([1.0, 1.0, 1.0, 1.0], index=index)
    result = backtest_positions(
        close, signals, fee_bps=0, slippage_bps=0, allow_short=True,
        funding_events=[{"timestamp": index[2].isoformat(), "rate_bps": 10.0}],
    )
    # Signals are delayed one bar: exposure at settlement is long.
    assert result["funding_cost_total"] == pytest.approx(0.001)
    assert result["total_return"] == pytest.approx(-0.001)


def test_positive_and_negative_funding_have_correct_long_short_sign():
    index = _times()
    position = pd.Series([0.0, 1.0, -1.0, -1.0], index=index)
    positive = funding_cost_series(
        position, [{"timestamp": index[1].isoformat(), "rate_bps": 10}]
    )
    negative = funding_cost_series(
        position, [{"timestamp": index[2].isoformat(), "rate_bps": -10}]
    )
    assert positive.sum() == pytest.approx(0.001)
    assert negative.sum() == pytest.approx(0.001)


def test_funding_event_requires_exact_timezone_aware_bar_timestamp():
    index = _times()
    position = pd.Series([0.0, 1.0, 1.0, 1.0], index=index)
    with pytest.raises(ValueError, match="does not align"):
        funding_cost_series(
            position, [{"timestamp": "2026-01-01T02:30:00Z", "rate_bps": 10}]
        )
    with pytest.raises(ValueError, match="timezone"):
        funding_cost_series(
            position, [{"timestamp": "2026-01-01T02:00:00", "rate_bps": 10}]
        )


def test_duplicate_funding_settlement_is_rejected():
    index = _times()
    position = pd.Series([0.0, 1.0, 1.0, 1.0], index=index)
    event = {"timestamp": index[1].isoformat(), "rate_bps": 10}
    with pytest.raises(ValueError, match="duplicate"):
        funding_cost_series(position, [event, event])


def test_performance_report_uses_event_funding_not_every_bar():
    index = _times()
    close = pd.Series([100.0] * 4, index=index)
    labels = pd.Series([2, 2, 2, 2], index=index)
    report = evaluate_predictions(
        close, labels, labels, fee_bps=0, slippage_bps=0,
        funding_events=[{"timestamp": index[2].isoformat(), "rate_bps": 10}],
    )
    model = report["trading"]["model"]
    assert model["funding_cost_total"] == pytest.approx(0.001)
    assert model["total_return"] == pytest.approx(-0.001)


def test_empty_funding_schedule_is_a_noop_even_without_datetime_index():
    position = pd.Series([0.0, 1.0, -1.0], index=[0, 1, 2])
    result = funding_cost_series(position, [])
    assert result.tolist() == [0.0, 0.0, 0.0]


def test_nonfinite_position_is_rejected_when_funding_event_is_present():
    index = _times()
    position = pd.Series([0.0, float("nan"), 1.0, 1.0], index=index)
    with pytest.raises(ValueError, match="positions must be finite"):
        funding_cost_series(
            position, [{"timestamp": index[1].isoformat(), "rate_bps": 10}]
        )
