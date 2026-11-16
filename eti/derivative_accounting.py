"""Contract-level derivative funding arithmetic.

This module is a pure accounting primitive. It is deliberately not wired into
ETI's normalized-exposure backtester or live execution path until actual
position sizing, mark-price history and settlement-currency equity accounting
are available end to end.
"""
from __future__ import annotations

import math
from typing import Any


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _currency(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be an explicit non-empty currency code")
    return value.strip().upper()


def calculate_contract_funding_payment(
    *,
    position_contracts: float,
    mark_price: float,
    funding_rate_bps: float,
    contract_size: float,
    contract_kind: str,
    base_currency: str,
    quote_currency: str,
    settlement_currency: str,
) -> dict[str, float | str]:
    """Calculate signed funding cost for one linear or inverse contract position.

    Positive cost means the account pays funding; negative cost means it receives
    funding. Positive funding rates charge longs and credit shorts. For linear
    contracts, contract_size is base units per contract. For inverse contracts,
    contract_size is quote-currency value per contract. Settlement is supported
    only in the instrument's base or quote currency; other currencies require an
    explicit FX conversion model and are rejected.
    """
    contracts = _finite("position_contracts", position_contracts)
    price = _finite("mark_price", mark_price)
    rate_bps = _finite("funding_rate_bps", funding_rate_bps)
    size = _finite("contract_size", contract_size)
    if price <= 0 or size <= 0:
        raise ValueError("mark_price and contract_size must be positive")
    if contracts == 0:
        raise ValueError("position_contracts must be non-zero")
    kind = str(contract_kind).strip().lower()
    if kind not in {"linear", "inverse"}:
        raise ValueError("contract_kind must be 'linear' or 'inverse'")
    base = _currency("base_currency", base_currency)
    quote = _currency("quote_currency", quote_currency)
    settlement = _currency("settlement_currency", settlement_currency)
    if base == quote:
        raise ValueError("base and quote currencies must be distinct")
    if settlement not in {base, quote}:
        raise ValueError("settlement_currency must be base or quote; FX conversion is unsupported")

    absolute_contracts = abs(contracts)
    if kind == "linear":
        base_notional = absolute_contracts * size
        quote_notional = base_notional * price
    else:
        quote_notional = absolute_contracts * size
        base_notional = quote_notional / price

    # Positive rate: longs pay, shorts receive. Negative rate reverses direction.
    quote_cost = math.copysign(1.0, contracts) * (rate_bps / 10_000.0) * quote_notional
    settlement_cost = quote_cost if settlement == quote else quote_cost / price
    values = (base_notional, quote_notional, quote_cost, settlement_cost)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("computed funding amount is not finite")
    return {
        "contract_kind": kind,
        "position_contracts": contracts,
        "mark_price": price,
        "funding_rate_bps": rate_bps,
        "contract_size": size,
        "base_notional": base_notional,
        "quote_notional": quote_notional,
        "quote_funding_cost": quote_cost,
        "settlement_currency": settlement,
        "settlement_funding_cost": settlement_cost,
    }
