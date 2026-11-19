"""Canonical model identity helpers shared by ETI training and inference.

New model IDs encode every instrument dimension that affects interpretation:
venue, asset class, market type, symbol and timeframe. IDs are stable,
lowercase, URL/database-safe identifiers.
"""
from __future__ import annotations

import re


def _component(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not normalized:
        raise ValueError(f"{name} must contain at least one letter or digit")
    return normalized


def model_id_for_instrument(
    *, venue: str, asset_class: str, market_type: str,
    symbol: str, timeframe: str,
) -> str:
    """Return ETI's canonical, dimension-complete model identifier."""
    parts = (
        _component(venue, "venue"),
        _component(asset_class, "asset_class"),
        _component(market_type, "market_type"),
        _component(symbol, "symbol"),
        _component(timeframe, "timeframe"),
    )
    return "eti-" + "-".join(parts)


def legacy_spot_model_id(venue: str, symbol: str, timeframe: str) -> str:
    """Return the pre-canonical ID used by legacy crypto-spot registrations."""
    parts = (
        _component(venue, "venue"),
        _component(symbol, "symbol"),
        _component(timeframe, "timeframe"),
    )
    return "eti-" + "-".join(parts)
