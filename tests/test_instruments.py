import pytest

from eti.instruments import InstrumentCatalog, InstrumentSpec


def spot():
    return InstrumentSpec(
        symbol="btc/usdt", venue="MEXC", asset_class="CRYPTO",
        market_type="spot", base_currency="BTC", quote_currency="USDT",
    )


def test_instrument_normalizes_identity_and_asset_fields():
    item = spot()
    assert item.key == ("mexc", "spot", "BTC/USDT")
    assert item.asset_class == "crypto"


def test_catalog_registers_and_retrieves_by_venue_market_and_symbol():
    catalog = InstrumentCatalog([spot()])
    assert catalog.get("MEXC", "SPOT", "btc/usdt") == spot()


def test_catalog_rejects_duplicate_without_explicit_replace():
    catalog = InstrumentCatalog([spot()])
    with pytest.raises(ValueError, match="already registered"):
        catalog.register(spot())


def test_catalog_can_filter_by_asset_market_and_venue():
    future = InstrumentSpec(
        symbol="BTC/USDT:USDT", venue="binance", asset_class="crypto",
        market_type="swap", max_leverage=50, supports_short=True,
        supports_funding=True, settlement_currency="USDT",
    )
    catalog = InstrumentCatalog([spot(), future])
    assert catalog.list(asset_class="crypto", market_type="swap")[0] == future
    assert catalog.list(venue="MEXC") == [spot()]


def test_inactive_instruments_are_hidden_by_default():
    inactive = InstrumentSpec(
        symbol="OLD/USDT", venue="mexc", asset_class="crypto",
        market_type="spot", active=False,
    )
    catalog = InstrumentCatalog([inactive])
    assert catalog.list() == []
    assert catalog.list(active_only=False) == [inactive]


@pytest.mark.parametrize("kwargs", [
    {"asset_class": "unknown"},
    {"market_type": "perpetual"},
    {"contract_multiplier": 0},
    {"contract_multiplier": float("nan")},
    {"tick_size": 0},
    {"tick_size": float("inf")},
    {"max_leverage": 2},
    {"supports_funding": True},
])
def test_invalid_spot_metadata_fails_closed(kwargs):
    values = dict(symbol="BTC/USDT", venue="mexc", asset_class="crypto", market_type="spot")
    values.update(kwargs)
    with pytest.raises(ValueError):
        InstrumentSpec(**values)
