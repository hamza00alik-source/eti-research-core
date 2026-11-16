import pandas as pd

from eti.provider_smoke import probe_provider, run_provider_smoke_check


class FakeProvider:
    def __init__(self, frame=None, error=None):
        self.frame = frame
        self.error = error

    def fetch_ohlcv(self, symbol, timeframe, limit):
        if self.error:
            raise self.error
        return self.frame.tail(limit).reset_index(drop=True)


def frame():
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=100, freq="h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [100.0] * 100, "high": [101.0] * 100,
        "low": [99.0] * 100, "close": [100.5] * 100,
        "volume": [10.0] * 100,
    })


def test_provider_probe_reports_validated_fresh_completed_candles():
    data = frame()
    now_ms = int((data.timestamp.iloc[-1] + pd.Timedelta(hours=1, seconds=1)).timestamp() * 1000)
    result = probe_provider("coinbaseexchange", FakeProvider(data), now_ms=now_ms)
    assert result["status"] == "healthy"
    assert result["rows"] == 100
    assert result["execution_capability"] is False


def test_provider_probe_rejects_stale_data():
    data = frame()
    now_ms = int((data.timestamp.iloc[-1] + pd.Timedelta(hours=5)).timestamp() * 1000)
    import pytest
    with pytest.raises(ValueError, match="stale"):
        probe_provider("kraken", FakeProvider(data), now_ms=now_ms)


def test_provider_probe_rejects_future_dated_data():
    data = frame()
    now_ms = int(data.timestamp.iloc[-1].timestamp() * 1000) - 1
    import pytest
    with pytest.raises(ValueError, match="future-dated"):
        probe_provider("kraken", FakeProvider(data), now_ms=now_ms)


def test_provider_failover_health_is_degraded_when_one_venue_fails():
    data = frame()
    report = run_provider_smoke_check({
        "coinbaseexchange": FakeProvider(data),
        "kraken": FakeProvider(error=TimeoutError("offline")),
    }, now_ms=int((data.timestamp.iloc[-1] + pd.Timedelta(hours=1, seconds=1)).timestamp() * 1000))
    assert report["status"] == "degraded"
    assert report["healthy_provider_count"] == 1
    assert report["training_performed"] is False
    assert report["execution_enabled"] is False


def test_provider_health_is_unavailable_when_both_venues_fail():
    report = run_provider_smoke_check({
        "coinbaseexchange": FakeProvider(error=TimeoutError("offline")),
        "kraken": FakeProvider(error=TimeoutError("offline")),
    })
    assert report["status"] == "unavailable"
    assert report["healthy_provider_count"] == 0



def test_provider_probe_rejects_incomplete_last_candle():
    data = frame()
    now_ms = int((data.timestamp.iloc[-1] + pd.Timedelta(minutes=30)).timestamp() * 1000)
    import pytest
    with pytest.raises(ValueError, match="incomplete"):
        probe_provider("kraken", FakeProvider(data), now_ms=now_ms)
