import pytest

from eti.live_market import MarketDataUnavailable, capture_public_spot_snapshot


NOW = 1_800_000_210_000  # 2027-01-15T08:03:30Z


class FakeExchange:
    def __init__(self, ticker=None, market=None, candles=None):
        self.ticker = ticker or {
            "timestamp": NOW - 10_000, "bid": 99.9, "ask": 100.1, "last": 100.0,
        }
        self.market = market or {"spot": True, "active": True}
        self.candles = candles or [
            [NOW - 210_000, 99, 101, 98, 100, 12],
            [NOW - 150_000, 100, 102, 99, 101, 13],
            [NOW - 90_000, 101, 103, 100, 102, 14],
            [NOW - 30_000, 102, 104, 101, 103, 15],  # still-open 1m candle
        ]
        self.closed = False

    def load_markets(self):
        return {"BTC/USDT": self.market}

    def milliseconds(self):
        return NOW

    def parse_timeframe(self, timeframe):
        return {"1m": 60, "5m": 300}[timeframe]

    def fetch_ohlcv(self, symbol, timeframe, limit):
        return self.candles[-limit:]

    def fetch_ticker(self, symbol):
        return self.ticker

    def close(self):
        self.closed = True


def factory_for(exchange):
    return lambda venue: exchange


def test_capture_fresh_public_spot_snapshot_uses_only_completed_candle():
    exchange = FakeExchange()
    result = capture_public_spot_snapshot(
        "binance", "BTC/USDT", "1m",
        exchange_factory=factory_for(exchange), now_ms=NOW,
    )
    assert result["quality"]["paper_eligible"] is True
    assert result["market_type"] == "spot"
    assert result["latest_completed_candle"]["close"] == 102.0
    assert result["quote"]["bid"] == 99.9
    assert result["quote"]["ask"] == 100.1
    assert result["quote"]["spread_bps"] > 0
    assert len(result["snapshot_sha256"]) == 64
    assert exchange.closed is True


def test_capture_rejects_stale_quote():
    exchange = FakeExchange(ticker={
        "timestamp": NOW - 60_000, "bid": 99.9, "ask": 100.1, "last": 100.0,
    })
    with pytest.raises(MarketDataUnavailable, match="quote is stale"):
        capture_public_spot_snapshot(
            "binance", "BTC/USDT", "1m",
            exchange_factory=factory_for(exchange), now_ms=NOW,
        )


def test_capture_rejects_quote_without_exchange_timestamp():
    exchange = FakeExchange(ticker={"bid": 99.9, "ask": 100.1, "last": 100.0})
    with pytest.raises(MarketDataUnavailable, match="timestamp is unavailable"):
        capture_public_spot_snapshot(
            "binance", "BTC/USDT", "1m",
            exchange_factory=factory_for(exchange), now_ms=NOW,
        )


def test_capture_rejects_inactive_or_non_spot_market():
    exchange = FakeExchange(market={"spot": False, "active": True})
    with pytest.raises(ValueError, match="active public spot market"):
        capture_public_spot_snapshot(
            "binance", "BTC/USDT", "1m",
            exchange_factory=factory_for(exchange), now_ms=NOW,
        )


def test_capture_rejects_crossed_or_invalid_quotes():
    exchange = FakeExchange(ticker={
        "timestamp": NOW - 1_000, "bid": 100.2, "ask": 100.1, "last": 100.0,
    })
    with pytest.raises(MarketDataUnavailable, match="ask is below bid"):
        capture_public_spot_snapshot(
            "binance", "BTC/USDT", "1m",
            exchange_factory=factory_for(exchange), now_ms=NOW,
        )


def test_capture_rejects_stale_completed_candle():
    old = NOW - 600_000
    exchange = FakeExchange(candles=[
        [old, 99, 101, 98, 100, 12],
        [old + 60_000, 100, 102, 99, 101, 13],
    ])
    with pytest.raises(MarketDataUnavailable, match="completed candle is stale"):
        capture_public_spot_snapshot(
            "binance", "BTC/USDT", "1m",
            exchange_factory=factory_for(exchange), now_ms=NOW,
        )


def test_capture_rejects_recent_completed_candle_gap():
    exchange = FakeExchange(candles=[
        [NOW - 210_000, 99, 101, 98, 100, 12],
        [NOW - 90_000, 101, 103, 100, 102, 14],
    ])
    with pytest.raises(MarketDataUnavailable, match="data gap"):
        capture_public_spot_snapshot(
            "binance", "BTC/USDT", "1m",
            exchange_factory=factory_for(exchange), now_ms=NOW,
        )
