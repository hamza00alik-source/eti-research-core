import pandas as pd
import pytest

import eti.data as market_data


def sample_frame(count=700):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    return pd.DataFrame({
        "timestamp": [start + pd.Timedelta(hours=i) for i in range(count)],
        "open": [100.0 + i for i in range(count)],
        "high": [102.0 + i for i in range(count)],
        "low": [99.0 + i for i in range(count)],
        "close": [101.0 + i for i in range(count)],
        "volume": [10.0] * count,
    })


def test_auto_provider_falls_back_and_records_actual_usd_market(monkeypatch):
    calls = []

    def fake_fetch(exchange, symbol, timeframe, limit, **kwargs):
        calls.append((exchange, symbol, timeframe, limit))
        if exchange == "coinbaseexchange":
            raise RuntimeError("regional API block")
        return sample_frame()

    monkeypatch.setattr(market_data, "fetch_ohlcv", fake_fetch)
    frame, source = market_data.fetch_ohlcv_auto(
        "BTC/USDT", "1h", 2000, min_rows=650
    )

    assert len(frame) == 700
    assert calls == [
        ("coinbaseexchange", "BTC/USD", "1h", 2000),
        ("kraken", "BTC/USD", "1h", 2000),
    ]
    assert source["provider"] == "kraken"
    assert source["venue_symbol"] == "BTC/USD"
    assert source["requested_symbol"] == "BTC/USDT"
    assert source["quote_mapping"] == "USDT_to_USD"
    assert "regional API block" in source["fallback_failures_before_success"][0]


def test_auto_provider_rejects_short_history_and_reports_all_failures(monkeypatch):
    monkeypatch.setattr(
        market_data, "fetch_ohlcv",
        lambda *args, **kwargs: sample_frame(500),
    )
    with pytest.raises(RuntimeError, match="insufficient history"):
        market_data.fetch_ohlcv_auto("BTC/USDT", "1h", 2000, min_rows=650)
