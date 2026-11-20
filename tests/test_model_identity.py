import pytest

from eti.model_identity import legacy_spot_model_id, model_id_for_instrument


def test_canonical_model_identity_includes_all_instrument_dimensions():
    assert model_id_for_instrument(
        venue="Binance", asset_class="Crypto", market_type="Spot",
        symbol="BTC/USDT", timeframe="1h",
    ) == "eti-binance-crypto-spot-btc-usdt-1h"


def test_canonical_model_identity_separates_market_types_and_asset_classes():
    common = {"venue": "binance", "symbol": "BTC/USDT", "timeframe": "1h"}
    spot = model_id_for_instrument(**common, asset_class="crypto", market_type="spot")
    swap = model_id_for_instrument(**common, asset_class="crypto", market_type="swap")
    equity = model_id_for_instrument(**common, asset_class="equity", market_type="spot")
    assert len({spot, swap, equity}) == 3


def test_canonical_model_identity_rejects_missing_or_unsafe_components():
    with pytest.raises(ValueError, match="asset_class"):
        model_id_for_instrument(
            venue="binance", asset_class="", market_type="spot",
            symbol="BTC/USDT", timeframe="1h",
        )
    with pytest.raises(ValueError, match="symbol"):
        model_id_for_instrument(
            venue="binance", asset_class="crypto", market_type="spot",
            symbol="---", timeframe="1h",
        )


def test_legacy_spot_identity_is_explicitly_separate_from_canonical_identity():
    assert legacy_spot_model_id("binance", "BTC/USDT", "1h") == "eti-binance-btc-usdt-1h"
    assert legacy_spot_model_id("binance", "BTC/USDT", "1h") != model_id_for_instrument(
        venue="binance", asset_class="crypto", market_type="spot",
        symbol="BTC/USDT", timeframe="1h",
    )
