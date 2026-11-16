import pytest

from eti.fill_costs import measure_market_order_fill_cost


def test_buy_fill_measures_adverse_shortfall_and_fee():
    result = measure_market_order_fill_cost(
        side="buy", arrival_mid_price=100,
        fills=[
            {"quantity": 2, "price": 100.1, "fee_quote": 0.04},
            {"quantity": 1, "price": 100.2, "fee_quote": 0.02},
        ],
    )
    assert result["vwap"] == pytest.approx((2 * 100.1 + 100.2) / 3)
    assert result["implementation_shortfall_bps"] > 0
    assert result["fee_bps"] == pytest.approx(0.06 / (2 * 100.1 + 100.2) * 10_000)
    assert result["all_in_observed_cost_bps"] == pytest.approx(
        result["implementation_shortfall_bps"] + result["fee_bps"]
    )


def test_sell_fill_measures_adverse_shortfall_with_opposite_sign():
    result = measure_market_order_fill_cost(
        side="sell", arrival_mid_price=100,
        fills=[{"quantity": 1, "price": 99.9, "fee_quote": 0.01}],
    )
    assert result["implementation_shortfall_bps"] > 0


def test_price_improvement_is_negative_shortfall():
    result = measure_market_order_fill_cost(
        side="buy", arrival_mid_price=100,
        fills=[{"quantity": 1, "price": 99.9, "fee_quote": 0}],
    )
    assert result["implementation_shortfall_bps"] < 0


def test_rebate_is_a_negative_fee_cost():
    result = measure_market_order_fill_cost(
        side="sell", arrival_mid_price=100,
        fills=[{"quantity": 1, "price": 100, "fee_quote": -0.02}],
    )
    assert result["fee_bps"] < 0


@pytest.mark.parametrize("kwargs", [
    {"side": "hold", "arrival_mid_price": 100, "fills": [{"quantity": 1, "price": 100, "fee_quote": 0}]},
    {"side": "buy", "arrival_mid_price": 0, "fills": [{"quantity": 1, "price": 100, "fee_quote": 0}]},
    {"side": "buy", "arrival_mid_price": 100, "fills": []},
    {"side": "buy", "arrival_mid_price": 100, "fills": [{"quantity": 0, "price": 100, "fee_quote": 0}]},
    {"side": "buy", "arrival_mid_price": 100, "fills": [{"quantity": 1, "price": float("nan"), "fee_quote": 0}]},
])
def test_invalid_fill_measurements_fail_closed(kwargs):
    with pytest.raises(ValueError):
        measure_market_order_fill_cost(**kwargs)
