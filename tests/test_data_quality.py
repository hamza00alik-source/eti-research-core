import pandas as pd
import pytest

from eti.data import assess_ohlcv_quality


def sample_frame():
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    return pd.DataFrame({
        "timestamp": [start, start + pd.Timedelta(hours=1), start + pd.Timedelta(hours=4)],
        "open": [100.0, 101.0, 103.0],
        "high": [102.0, 103.0, 105.0],
        "low": [99.0, 100.0, 102.0],
        "close": [101.0, 102.0, 104.0],
        "volume": [10.0, 11.0, 12.0],
    })


def test_assess_ohlcv_quality_reports_gaps_and_calendar_coverage():
    frame = sample_frame()
    now = pd.Timestamp("2026-01-01T05:01:00Z")

    report = assess_ohlcv_quality(frame, "1h", now=now)

    assert report["status"] == "fresh"
    assert report["observed_candles"] == 3
    assert report["expected_slots_in_observed_span"] == 5
    assert report["calendar_coverage_ratio"] == 0.6
    assert report["gap_count"] == 1
    assert report["largest_gap_intervals"] == 3.0
    assert report["latest_candle_age_seconds"] == 60.0


def test_assess_ohlcv_quality_flags_old_latest_candle():
    frame = sample_frame()
    now = pd.Timestamp("2026-01-05T00:00:00Z")

    report = assess_ohlcv_quality(frame, "1h", now=now)

    assert report["status"] == "stale"


def test_assess_ohlcv_quality_rejects_unknown_timeframe():
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        assess_ohlcv_quality(sample_frame(), "1M")
