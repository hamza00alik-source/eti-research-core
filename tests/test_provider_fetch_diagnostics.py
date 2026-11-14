import pandas as pd

from eti.agents import _ohlcv_fetch_diagnostics


class FakeProvider:
    provider_name = "fake-public"
    venue = "test-venue"


def sample_frame(count=3):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    return pd.DataFrame({
        "timestamp": [start + pd.Timedelta(hours=i) for i in range(count)],
        "open": [100.0 + i for i in range(count)],
        "high": [101.0 + i for i in range(count)],
        "low": [99.0 + i for i in range(count)],
        "close": [100.5 + i for i in range(count)],
        "volume": [10.0] * count,
    })


def test_provider_diagnostics_report_shortfall_and_identity():
    report = _ohlcv_fetch_diagnostics(FakeProvider(), sample_frame(), "1h", 5)

    assert report["provider"] == "fake-public"
    assert report["provider_class"] == "FakeProvider"
    assert report["venue"] == "test-venue"
    assert report["requested_candles"] == 5
    assert report["returned_candles"] == 3
    assert report["request_shortfall_candles"] == 2
    assert report["request_fulfillment_ratio"] == 0.6
    assert report["first_returned_candle"] == "2026-01-01T00:00:00+00:00"
    assert report["latest_returned_candle"] == "2026-01-01T02:00:00+00:00"


def test_provider_diagnostics_cap_ratio_and_do_not_claim_market_coverage():
    report = _ohlcv_fetch_diagnostics(FakeProvider(), sample_frame(6), "1h", 5)

    assert report["request_shortfall_candles"] == 0
    assert report["request_fulfillment_ratio"] == 1.0
    assert "not a measure" in report["note"]


def test_provider_diagnostics_handles_empty_response():
    report = _ohlcv_fetch_diagnostics(FakeProvider(), sample_frame(0), "1h", 5)

    assert report["returned_candles"] == 0
    assert report["request_shortfall_candles"] == 5
    assert report["request_fulfillment_ratio"] == 0.0
    assert report["first_returned_candle"] is None
    assert report["latest_returned_candle"] is None
