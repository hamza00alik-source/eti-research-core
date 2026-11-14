import sys
from types import SimpleNamespace

import pandas as pd

from eti.agents import _provider_source_metadata
from eti.data import fetch_ohlcv


def test_ccxt_fetch_reports_requested_and_returned_candle_metadata(monkeypatch):
    now_ms = 1_760_000_000_000
    rows = [
        [now_ms - 4 * 3_600_000, 100, 102, 99, 101, 10],
        [now_ms - 3 * 3_600_000, 101, 103, 100, 102, 11],
        [now_ms - 2 * 3_600_000, 102, 104, 101, 103, 12],
    ]

    class FakeExchange:
        has = {"fetchOHLCV": True}

        def __init__(self, _config):
            self.calls = 0

        def load_markets(self):
            return {"BTC/USDT": {"spot": True}}

        def parse_timeframe(self, _timeframe):
            return 3600

        def milliseconds(self):
            return now_ms

        def fetch_ohlcv(self, _symbol, timeframe, since, limit):
            assert timeframe == "1h"
            assert since is not None
            assert limit > 0
            self.calls += 1
            return rows if self.calls == 1 else []

    exchange = FakeExchange({})
    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(binance=lambda config: exchange))

    frame = fetch_ohlcv("binance", "BTC/USDT", "1h", 5)
    metadata = frame.attrs["source_metadata"]

    assert len(frame) == 3
    assert metadata["provider"] == "ccxt-public"
    assert metadata["exchange"] == "binance"
    assert metadata["requested_candles"] == 5
    assert metadata["raw_candles_received"] == 3
    assert metadata["completed_candles_returned"] == 3
    assert metadata["fetch_pages"] == 1


def test_agent_provider_metadata_uses_frame_details_and_safe_fallbacks():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
        "open": [1.0, 2.0, 3.0], "high": [2.0, 3.0, 4.0],
        "low": [0.5, 1.5, 2.5], "close": [1.5, 2.5, 3.5],
        "volume": [10.0, 11.0, 12.0],
    })
    frame.attrs["source_metadata"] = {
        "provider": "test-provider",
        "completed_candles_returned": 3,
    }

    metadata = _provider_source_metadata(
        frame, "test-venue", "equity", "spot", "TEST", "1h", 100
    )

    assert metadata["provider"] == "test-provider"
    assert metadata["venue"] == "test-venue"
    assert metadata["asset_class"] == "equity"
    assert metadata["symbol"] == "TEST"
    assert metadata["requested_candles"] == 100
    assert metadata["completed_candles_returned"] == 3
    assert metadata["first_returned_candle"] == frame.iloc[0]["timestamp"].isoformat()
    assert metadata["last_returned_candle"] == frame.iloc[-1]["timestamp"].isoformat()
