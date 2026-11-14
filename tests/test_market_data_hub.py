import pandas as pd
import pytest

from eti.adapters import AdapterCapabilities
from eti.instruments import InstrumentSpec
from eti.market_data_hub import MarketDataHub


def candles(rows=5, start="2026-01-01T00:00:00Z"):
    index = pd.date_range(start, periods=rows, freq="h")
    return pd.DataFrame({
        "timestamp": index, "open": [100.0] * rows, "high": [102.0] * rows,
        "low": [99.0] * rows, "close": [101.0] * rows, "volume": [10.0] * rows,
    })


class FakeAdapter:
    def __init__(self, venue, frame=None, error=None, asset="crypto", market="spot"):
        self.capabilities = AdapterCapabilities(
            venue=venue, asset_classes=(asset,), market_types=(market,),
            public_market_data=True, order_execution=False,
        )
        self.frame, self.error = frame, error
        self.calls = 0

    def discover_instruments(self):
        return [InstrumentSpec(
            symbol="BTC/USDT" if self.capabilities.venue != "second" else "BTC/USD",
            venue=self.capabilities.venue, asset_class="crypto", market_type="spot",
            base_currency="BTC", quote_currency="USDT" if self.capabilities.venue != "second" else "USD",
        )]

    def fetch_ohlcv(self, symbol, timeframe, limit):
        self.calls += 1
        if self.error:
            raise self.error
        return self.frame.tail(limit)


def test_discovery_collects_across_venues_and_keeps_venue_identity():
    hub = MarketDataHub([FakeAdapter("first"), FakeAdapter("second")])
    result = hub.discover(asset_class="crypto", market_type="spot")
    assert result["count"] == 2
    assert {item.venue for item in result["instruments"]} == {"first", "second"}
    assert result["source_errors"] == []


def test_fetch_validates_rows_and_reports_provenance_and_gaps():
    frame = candles(5)
    frame = pd.concat([frame.iloc[:2], frame.iloc[3:]], ignore_index=True)
    adapter = FakeAdapter("first", frame=frame)
    hub = MarketDataHub([adapter])
    instrument = adapter.discover_instruments()[0]
    result = hub.fetch(instrument, "1h", 10, min_rows=4)
    assert result.provenance["venue"] == "first"
    assert result.provenance["returned_rows"] == 4
    assert result.quality["ohlcv_validated"] is True
    assert result.quality["missing_interval_estimate"] == 1


def test_fallback_uses_next_explicit_candidate_and_never_merges_series():
    first = FakeAdapter("first", error=RuntimeError("provider blocked"))
    second = FakeAdapter("second", frame=candles(6))
    hub = MarketDataHub([first, second])
    a, b = first.discover_instruments()[0], second.discover_instruments()[0]
    result = hub.fetch_first_available([a, b], "1h", 6, min_rows=5)
    assert result.instrument.venue == "second"
    assert result.provenance["fallback_count"] == 1
    assert result.provenance["fallback_failures_before_success"][0]["venue"] == "first"
    assert len(result.frame) == 6
    assert first.calls == 1 and second.calls == 1


def test_fallback_rejects_short_history_and_fails_closed():
    first = FakeAdapter("first", frame=candles(2))
    hub = MarketDataHub([first])
    instrument = first.discover_instruments()[0]
    with pytest.raises(RuntimeError, match="No candidate data source succeeded"):
        hub.fetch_first_available([instrument], "1h", 10, min_rows=5)


def test_market_hub_does_not_register_duplicate_venue_adapters():
    with pytest.raises(ValueError, match="already registered"):
        MarketDataHub([FakeAdapter("same"), FakeAdapter("same")])


def test_fetch_rejects_incompatible_timeframe_for_quality_calculation():
    adapter = FakeAdapter("first", frame=candles(5))
    hub = MarketDataHub([adapter])
    with pytest.raises(ValueError, match="fixed-duration"):
        hub.fetch(adapter.discover_instruments()[0], "1M", 5)


def test_default_public_hub_registers_independent_crypto_and_reference_sources():
    from eti.market_data_hub import build_public_market_hub

    hub = build_public_market_hub()
    sources = {row["venue"]: row for row in hub.available_sources()}
    assert {"binance", "mexc", "coinbaseexchange", "kraken", "yahoo"} == set(sources)
    assert sources["coinbaseexchange"]["market_types"] == ["spot"]
    assert sources["kraken"]["market_types"] == ["spot"]
    assert sources["yahoo"]["asset_classes"] == ["commodity", "equity", "forex", "index"]
    assert all(row["order_execution"] is False for row in sources.values())


def test_ccxt_discovery_keeps_spot_when_one_derivative_market_type_fails(monkeypatch):
    import eti.market_data_hub as hub_module

    class Provider:
        def __init__(self, venue, market_type):
            self.market_type = market_type
        def discover_markets(self, limit=1000):
            if self.market_type == "future":
                raise RuntimeError("futures endpoint unavailable")
            return [{
                "symbol": "BTC/USDT", "base": "BTC", "quote": "USDT",
                "active": True, "contract_size": 1,
            }]

    monkeypatch.setattr(hub_module, "CcxtPublicProvider", Provider)
    adapter = hub_module.CcxtMarketAdapter("binance", ("spot", "future"))
    result = hub_module.MarketDataHub([adapter]).discover(asset_class="crypto")
    assert result["count"] == 1
    assert result["instruments"][0].market_type == "spot"
    assert result["source_errors"][0]["venue"] == "binance"
    assert "futures endpoint unavailable" in result["source_errors"][0]["message"]


def test_yahoo_discovery_is_scoped_and_keeps_partial_asset_class_results(monkeypatch):
    import eti.market_data_hub as hub_module

    calls = []
    class Provider:
        def __init__(self, asset_class):
            self.asset_class = asset_class
        def discover_markets(self, limit=500, market_type="spot"):
            calls.append((self.asset_class, market_type))
            if self.asset_class == "equity":
                raise RuntimeError("equity reference endpoint unavailable")
            return [{"symbol": "EURUSD=X"}]

    monkeypatch.setattr(hub_module, "YahooFinanceProvider", Provider)
    adapter = hub_module.YahooFinanceMarketAdapter()
    result = hub_module.MarketDataHub([adapter]).discover(
        asset_class="forex", market_type="spot"
    )
    assert calls == [("forex", "spot"), ("equity", "spot"), ("index", "spot")]
    assert result["count"] == 1
    assert result["instruments"][0].asset_class == "forex"
    assert len(result["source_errors"]) == 1
    assert "equity reference endpoint unavailable" in result["source_errors"][0]["message"]
