"""Measure realized market-order implementation shortfall from recorded fills.

Fees must already be converted into the instrument quote currency by the
caller. This module measures observed outcomes; it does not predict future
slippage or split spread from market impact without order-book snapshots.
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


def measure_market_order_fill_cost(
    *,
    side: str,
    arrival_mid_price: float,
    fills: list[dict[str, Any]],
) -> dict[str, float | int | str]:
    """Measure side-adjusted shortfall and fee in bps for one completed order.

    A positive implementation_shortfall_bps is adverse. A negative value is
    price improvement. fee_quote is signed and already converted to quote
    currency; positive means paid, negative means rebate/credit.
    """
    direction = str(side).strip().lower()
    if direction not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")
    mid = _finite("arrival_mid_price", arrival_mid_price)
    if mid <= 0:
        raise ValueError("arrival_mid_price must be positive")
    if not isinstance(fills, list) or not fills:
        raise ValueError("at least one fill is required")

    total_quantity = 0.0
    total_notional = 0.0
    total_fee = 0.0
    for index, fill in enumerate(fills):
        if not isinstance(fill, dict):
            raise ValueError(f"fills[{index}] must be an object")
        quantity = _finite(f"fills[{index}].quantity", fill.get("quantity"))
        price = _finite(f"fills[{index}].price", fill.get("price"))
        fee = _finite(f"fills[{index}].fee_quote", fill.get("fee_quote"))
        if quantity <= 0 or price <= 0:
            raise ValueError(f"fills[{index}] quantity and price must be positive")
        total_quantity += quantity
        total_notional += quantity * price
        total_fee += fee

    if not all(math.isfinite(value) for value in (total_quantity, total_notional, total_fee)):
        raise ValueError("aggregated fill values are not finite")
    if total_notional <= 0:
        raise ValueError("filled notional must be positive")
    vwap = total_notional / total_quantity
    raw_move = (vwap / mid - 1.0) * 10_000.0
    shortfall_bps = raw_move if direction == "buy" else -raw_move
    fee_bps = total_fee / total_notional * 10_000.0
    return {
        "side": direction,
        "fill_count": len(fills),
        "filled_quantity": total_quantity,
        "filled_notional_quote": total_notional,
        "vwap": vwap,
        "arrival_mid_price": mid,
        "implementation_shortfall_bps": shortfall_bps,
        "fee_quote": total_fee,
        "fee_bps": fee_bps,
        "all_in_observed_cost_bps": shortfall_bps + fee_bps,
    }
