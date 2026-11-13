import pandas as pd
import pytest

from eti.providers import CcxtPublicProvider, CsvOHLCVProvider


def test_ccxt_provider_accepts_only_explicit_public_market_types(monkeypatch):
    import eti.providers as providers
    captured = {}

    def fake_fetch(exchange_id, symbol, timeframe, limit, market_type="spot"):
        captured.update(exchange_id=exchange_id, symbol=symbol, timeframe=timeframe,
                        limit=limit, market_type=market_type)
        return pd.DataFrame({"timestamp": [], "open": [], "high": [],
                             "low": [], "close": [], "volume": []})

    monkeypatch.setattr(providers, "fetch_ohlcv", fake_fetch)
    provider = CcxtPublicProvider("binance", "swap")
    provider.fetch_ohlcv("BTC/USDT:USDT", "1h", 1500)
    assert captured == {"exchange_id": "binance", "symbol": "BTC/USDT:USDT",
                        "timeframe": "1h", "limit": 1500, "market_type": "swap"}
    with pytest.raises(ValueError, match="market type"):
        CcxtPublicProvider("binance", "option")


def test_csv_provider_remains_available_for_non_crypto_assets(tmp_path):
    path = tmp_path / "eurusd.csv"
    pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
        "open": [1.1, 1.2, 1.3], "high": [1.11, 1.21, 1.31],
        "low": [1.09, 1.19, 1.29], "close": [1.1, 1.2, 1.3],
        "volume": [100, 110, 120],
    }).to_csv(path, index=False)
    frame = CsvOHLCVProvider(path).fetch_ohlcv("EUR/USD", "1h", 3)
    assert len(frame) == 3
    assert frame.iloc[-1]["close"] == pytest.approx(1.3)


def test_yahoo_provider_normalizes_public_ohlcv(monkeypatch):
    import sys
    from types import SimpleNamespace
    from eti.providers import YahooFinanceProvider

    index = pd.date_range("2026-01-01", periods=60, freq="h", tz="UTC")
    source = pd.DataFrame({
        "Open": [100.0] * 60, "High": [102.0] * 60,
        "Low": [99.0] * 60, "Close": [101.0] * 60,
        "Volume": [1000.0] * 60,
    }, index=index)
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(
        download=lambda *args, **kwargs: source
    ))
    import eti.providers as providers
    original_drop = providers.drop_incomplete_candles
    captured = {}
    def checked_drop(frame, timeframe_ms, now_ms):
        captured["timeframe_ms"] = timeframe_ms
        captured["now_ms"] = now_ms
        return original_drop(frame, timeframe_ms, now_ms)
    monkeypatch.setattr(providers, "drop_incomplete_candles", checked_drop)
    frame = YahooFinanceProvider("equity").fetch_ohlcv("AAPL", "1h", 50)
    assert captured["timeframe_ms"] == 3_600_000
    assert captured["now_ms"] > 0
    assert len(frame) == 50
    assert list(frame.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert frame.iloc[-1]["close"] == pytest.approx(101.0)
