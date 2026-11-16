import pytest

from eti.derivative_accounting import calculate_contract_funding_payment


def calc(**overrides):
    values = {
        "position_contracts": 2.0,
        "mark_price": 50_000.0,
        "funding_rate_bps": 10.0,
        "contract_size": 0.001,
        "contract_kind": "linear",
        "base_currency": "BTC",
        "quote_currency": "USDT",
        "settlement_currency": "USDT",
    }
    values.update(overrides)
    return calculate_contract_funding_payment(**values)


def test_linear_contract_funding_uses_base_quantity_times_mark_price():
    result = calc()
    assert result["base_notional"] == pytest.approx(0.002)
    assert result["quote_notional"] == pytest.approx(100.0)
    assert result["settlement_funding_cost"] == pytest.approx(0.1)


def test_inverse_contract_funding_uses_quote_value_per_contract():
    result = calc(
        position_contracts=2,
        contract_size=100.0,
        contract_kind="inverse",
        settlement_currency="BTC",
    )
    assert result["quote_notional"] == pytest.approx(200.0)
    assert result["base_notional"] == pytest.approx(0.004)
    assert result["quote_funding_cost"] == pytest.approx(0.2)
    assert result["settlement_funding_cost"] == pytest.approx(0.2 / 50_000)


def test_short_position_receives_positive_funding():
    result = calc(position_contracts=-2)
    assert result["settlement_funding_cost"] == pytest.approx(-0.1)


def test_negative_funding_rate_reverses_long_payment():
    result = calc(funding_rate_bps=-10)
    assert result["settlement_funding_cost"] == pytest.approx(-0.1)


def test_quote_settlement_for_inverse_contract():
    result = calc(
        position_contracts=2, contract_size=100, contract_kind="inverse",
        settlement_currency="USDT",
    )
    assert result["settlement_funding_cost"] == pytest.approx(0.2)


@pytest.mark.parametrize("overrides", [
    {"position_contracts": 0},
    {"position_contracts": float("nan")},
    {"mark_price": 0},
    {"mark_price": float("inf")},
    {"contract_size": -1},
    {"funding_rate_bps": float("nan")},
    {"contract_kind": "quanto"},
    {"settlement_currency": "EUR"},
])
def test_invalid_contract_funding_inputs_fail_closed(overrides):
    with pytest.raises(ValueError):
        calc(**overrides)



@pytest.mark.parametrize("overrides", [
    {"base_currency": None},
    {"quote_currency": ""},
    {"settlement_currency": "   "},
    {"base_currency": "BTC", "quote_currency": "BTC", "settlement_currency": "BTC"},
])
def test_contract_funding_requires_explicit_distinct_currency_codes(overrides):
    with pytest.raises(ValueError, match="currency"):
        calc(**overrides)
