import json

import pytest

from eti.live_market_data import BinanceSpotKlineFeed

OPEN_MS = 1_800_000_000_000
INTERVAL_MS = 3_600_000
EVENT_MS = OPEN_MS + INTERVAL_MS


def message(*, open_ms=OPEN_MS, event_ms=EVENT_MS,
            closed=True, symbol="BTCUSDT", interval="1h", close="101"):
    return {
        "e": "kline", "E": event_ms, "s": symbol,
        "k": {"t": open_ms, "T": open_ms + INTERVAL_MS - 1, "s": symbol,
              "i": interval, "x": closed, "o": "100", "h": "102",
              "l": "99", "c": close, "v": "12.5"},
    }


def test_parse_completed_spot_kline_and_normalize_timestamp():
    feed = BinanceSpotKlineFeed("BTC/USDT", "1h")
    candle = feed.parse(json.dumps(message()), now_ms=EVENT_MS + 1)
    assert candle is not None
    assert candle.venue == "binance"
    assert candle.close == 101
    assert candle.as_record()["timestamp"].isoformat() == "2027-01-15T08:00:00+00:00"


def test_in_progress_kline_is_ignored():
    feed = BinanceSpotKlineFeed("BTC/USDT", "1h")
    assert feed.parse(message(closed=False), now_ms=EVENT_MS) is None


@pytest.mark.parametrize("symbol,interval", [("ETHUSDT", "1h"), ("BTCUSDT", "4h")])
def test_wrong_stream_identity_is_rejected(symbol, interval):
    feed = BinanceSpotKlineFeed("BTC/USDT", "1h")
    with pytest.raises(ValueError, match="does not match"):
        feed.parse(message(symbol=symbol, interval=interval), now_ms=EVENT_MS + 1)


def test_stale_and_future_events_are_rejected():
    feed = BinanceSpotKlineFeed("BTC/USDT", "1h", max_event_age_ms=100)
    with pytest.raises(ValueError, match="stale"):
        feed.parse(message(), now_ms=EVENT_MS + 101)
    with pytest.raises(ValueError, match="future"):
        feed.parse(message(event_ms=EVENT_MS + 10_000), now_ms=EVENT_MS)


@pytest.mark.parametrize("field,value", [
    ("c", "NaN"), ("v", "-1"), ("h", "98"),
])
def test_invalid_market_values_are_rejected(field, value):
    payload = message()
    payload["k"][field] = value
    feed = BinanceSpotKlineFeed("BTC/USDT", "1h")
    with pytest.raises(ValueError):
        feed.parse(payload, now_ms=EVENT_MS + 1)


def test_invalid_symbol_and_non_spot_intervals_are_rejected():
    with pytest.raises(ValueError, match="BASE/QUOTE"):
        BinanceSpotKlineFeed("BTCUSDT", "1h")
    with pytest.raises(ValueError, match="interval"):
        BinanceSpotKlineFeed("BTC/USDT", "1M")


def test_sequence_guard_deduplicates_and_rejects_gaps_and_reordering():
    feed = BinanceSpotKlineFeed("BTC/USDT", "1h")
    first = feed.parse(message(), now_ms=EVENT_MS + 1)
    assert feed.accept_sequence(first) is True
    assert feed.accept_sequence(first) is False

    second_open = OPEN_MS + INTERVAL_MS
    second_event = second_open + INTERVAL_MS
    second_payload = message(open_ms=second_open, event_ms=second_event)
    second = feed.parse(second_payload, now_ms=second_event + 1)
    assert feed.accept_sequence(second) is True
    with pytest.raises(ValueError, match="out-of-order"):
        feed.accept_sequence(first)

    gap_open = OPEN_MS + 3 * INTERVAL_MS
    gap_event = gap_open + INTERVAL_MS
    gap_payload = message(open_ms=gap_open, event_ms=gap_event)
    gap = feed.parse(gap_payload, now_ms=gap_event + 1)
    with pytest.raises(ValueError, match="gap"):
        feed.accept_sequence(gap)


def test_url_is_public_spot_stream():
    feed = BinanceSpotKlineFeed("BTC/USDT", "1m")
    assert feed.url == "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"


def test_receive_refreshes_clock_for_each_message():
    feed = BinanceSpotKlineFeed("BTC/USDT", "1h")
    first = message(closed=False)
    second = message()

    class FakeSocket:
        items = iter([json.dumps(first), json.dumps(second)])

        def recv(self):
            return next(self.items)

    clocks = iter([EVENT_MS, EVENT_MS + 1])
    candle = feed.receive_closed_candle(FakeSocket(), clock_ms=lambda: next(clocks))
    assert candle.close == 101


def test_sequence_guard_rejects_conflicting_duplicate_candle():
    feed = BinanceSpotKlineFeed("BTC/USDT", "1h")
    first = feed.parse(message(), now_ms=EVENT_MS + 1)
    assert feed.accept_sequence(first) is True
    retransmit = feed.parse(
        message(event_ms=EVENT_MS + 1), now_ms=EVENT_MS + 1
    )
    assert feed.accept_sequence(retransmit) is False
    changed = feed.parse(message(close="100.5"), now_ms=EVENT_MS + 1)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        feed.accept_sequence(changed)
