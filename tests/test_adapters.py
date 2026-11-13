import pandas as pd
import pytest

from eti.adapters import AdapterCapabilities, AdapterRegistry
from eti.instruments import InstrumentSpec


class FakeAdapter:
    capabilities = AdapterCapabilities(
        venue="example", asset_classes=("crypto", "commodity"),
        market_types=("spot", "future"), order_execution=False,
    )

    def discover_instruments(self):
        return [InstrumentSpec(
            symbol="PAXG/USDT", venue="example", asset_class="commodity",
            market_type="spot", base_currency="PAXG", quote_currency="USDT",
        )]

    def fetch_ohlcv(self, symbol, timeframe, limit):
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_adapter_registry_resolves_supported_combination():
    registry = AdapterRegistry()
    adapter = FakeAdapter()
    registry.register(adapter)
    assert registry.get("EXAMPLE", asset_class="commodity", market_type="spot") is adapter


def test_adapter_registry_rejects_unsupported_asset_class():
    registry = AdapterRegistry()
    registry.register(FakeAdapter())
    with pytest.raises(LookupError, match="does not support asset class"):
        registry.get("example", asset_class="equity", market_type="spot")


def test_adapter_registry_rejects_unsupported_market_type():
    registry = AdapterRegistry()
    registry.register(FakeAdapter())
    with pytest.raises(LookupError, match="does not support market type"):
        registry.get("example", asset_class="commodity", market_type="cfd")


def test_adapter_registry_rejects_duplicate_venue():
    registry = AdapterRegistry()
    registry.register(FakeAdapter())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(FakeAdapter())


def test_adapter_capabilities_are_normalized_and_read_only():
    caps = FakeAdapter.capabilities
    assert caps.venue == "example"
    assert caps.asset_classes == ("commodity", "crypto")
    assert caps.order_execution is False
