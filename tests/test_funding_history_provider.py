import sys
from types import SimpleNamespace

import pytest

from eti.providers import CcxtPublicProvider


class FakeExchange:
    def __init__(self, *, rows, has_history=True):
        self.rows = list(rows)
        self.has = {"fetchFundingRateHistory": has_history}
        self.urls = {"api": {"public": "https://api.example.test/public"}}
        self.calls = []
        self.closed = False

    def load_markets(self):
        return {
            "BTC/USDT:USDT": {
                "symbol": "BTC/USDT:USDT", "swap": True, "base": "BTC",
                "quote": "USDT", "settle": "USDT", "linear": True,
                "inverse": False, "contractSize": 0.001, "active": True,
            }
        }

    def milliseconds(self):
        return 5000

    def fetch_funding_rate_history(self, symbol, since, limit):
        self.calls.append((symbol, since, limit))
        rows = self.rows.pop(0) if self.rows else []
        return rows

    def close(self):
        self.closed = True


def row(timestamp, rate):
    return {"timestamp": timestamp, "fundingRate": rate}


def test_paginated_funding_history_preserves_events_and_marks_completeness_unverified(monkeypatch):
    exchange = FakeExchange(rows=[
        [row(1000, 0.0001), row(2000, -0.0002)],
        [row(3000, 0.0003)],
        [],
    ])
    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(binance=lambda config: exchange))
    provider = CcxtPublicProvider("binance", "swap")
    result = provider.fetch_funding_rate_history(
        "BTC/USDT:USDT", 1000, 3000, limit=10, page_size=2,
    )
    assert result["event_count"] == 3
    assert [event["rate_bps"] for event in result["events"]] == pytest.approx([1, -2, 3])
    assert result["history_completeness"] == "unverified"
    assert result["evidence_eligible"] is False
    assert result["endpoint_uri"] == "https://api.example.test/public"
    assert result["contract_metadata"]["contract_size"] == 0.001
    assert result["page_count"] == 3
    assert exchange.calls == [
        ("BTC/USDT:USDT", 1000, 2),
        ("BTC/USDT:USDT", 2001, 2),
        ("BTC/USDT:USDT", 3001, 2),
    ]
    assert exchange.closed is True


def test_funding_history_requires_derivative_market_type():
    with pytest.raises(ValueError, match="only for swap and future"):
        CcxtPublicProvider("binance", "spot").fetch_funding_rate_history(
            "BTC/USDT", 1000, 3000,
        )


def test_funding_history_rejects_exchange_without_history_capability(monkeypatch):
    exchange = FakeExchange(rows=[], has_history=False)
    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(binance=lambda config: exchange))
    with pytest.raises(NotImplementedError, match="does not expose"):
        CcxtPublicProvider("binance", "swap").fetch_funding_rate_history(
            "BTC/USDT:USDT", 1000, 3000,
        )
    assert exchange.closed is True


def test_funding_history_rejects_conflicting_duplicate_timestamp(monkeypatch):
    exchange = FakeExchange(rows=[
        [row(1000, 0.0001), row(2000, 0.0002)],
        [row(2000, 0.0003)],
    ])
    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(binance=lambda config: exchange))
    # A conflicting duplicate is rejected rather than silently overwriting history.
    with pytest.raises(ValueError, match="conflicting rates"):
        CcxtPublicProvider("binance", "swap").fetch_funding_rate_history(
            "BTC/USDT:USDT", 1000, 3000, limit=10, page_size=2,
        )
    assert exchange.closed is True


def test_funding_history_rejects_future_coverage_before_api_call(monkeypatch):
    exchange = FakeExchange(rows=[])
    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(binance=lambda config: exchange))
    with pytest.raises(ValueError, match="cannot be in the future"):
        CcxtPublicProvider("binance", "swap").fetch_funding_rate_history(
            "BTC/USDT:USDT", 1000, 6000,
        )
    assert exchange.closed is True



def test_funding_history_reports_when_requested_event_limit_truncates_interval(monkeypatch):
    exchange = FakeExchange(rows=[
        [row(1000, 0.0001), row(2000, 0.0002)],
    ])
    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(binance=lambda config: exchange))
    result = CcxtPublicProvider("binance", "swap").fetch_funding_rate_history(
        "BTC/USDT:USDT", 1000, 3000, limit=2, page_size=2,
    )
    assert result["event_count"] == 2
    assert result["limit_reached"] is True
    assert result["history_completeness"] == "unverified"
    assert result["evidence_eligible"] is False
