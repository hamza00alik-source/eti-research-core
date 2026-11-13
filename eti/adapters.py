"""Exchange-independent adapter capability registry.

Adapters normalize venue-specific APIs into ETI's canonical instrument and OHLCV
contracts. This registry does not provide order placement or bypass risk gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from .instruments import ASSET_CLASSES, MARKET_TYPES, InstrumentSpec


@dataclass(frozen=True)
class AdapterCapabilities:
    venue: str
    asset_classes: tuple[str, ...]
    market_types: tuple[str, ...]
    public_market_data: bool = True
    authenticated_market_data: bool = False
    order_execution: bool = False

    def __post_init__(self) -> None:
        if not self.venue or not self.venue.strip():
            raise ValueError("venue must be a non-empty string")
        assets = tuple(sorted({item.strip().lower() for item in self.asset_classes}))
        markets = tuple(sorted({item.strip().lower() for item in self.market_types}))
        if not assets or not set(assets).issubset(ASSET_CLASSES):
            raise ValueError("asset_classes must contain supported ETI asset classes")
        if not markets or not set(markets).issubset(MARKET_TYPES):
            raise ValueError("market_types must contain supported ETI market types")
        for name in ("public_market_data", "authenticated_market_data", "order_execution"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        object.__setattr__(self, "venue", self.venue.strip().lower())
        object.__setattr__(self, "asset_classes", assets)
        object.__setattr__(self, "market_types", markets)


@runtime_checkable
class MarketAdapter(Protocol):
    """Minimal read-side adapter contract; no order methods are defined."""

    capabilities: AdapterCapabilities

    def discover_instruments(self) -> list[InstrumentSpec]: ...

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame: ...


class AdapterRegistry:
    """Register adapters by venue; resolve only explicitly supported combinations."""

    def __init__(self) -> None:
        self._adapters: dict[str, MarketAdapter] = {}

    def register(self, adapter: MarketAdapter, *, replace: bool = False) -> None:
        if not isinstance(adapter, MarketAdapter):
            raise TypeError("adapter must implement capabilities, discover_instruments and fetch_ohlcv")
        venue = adapter.capabilities.venue
        if venue in self._adapters and not replace:
            raise ValueError(f"Adapter already registered for venue: {venue}")
        self._adapters[venue] = adapter

    def get(self, venue: str, *, asset_class: str, market_type: str) -> MarketAdapter:
        key = venue.strip().lower()
        if key not in self._adapters:
            raise LookupError(f"No adapter registered for venue: {key}")
        adapter = self._adapters[key]
        capabilities = adapter.capabilities
        if asset_class.strip().lower() not in capabilities.asset_classes:
            raise LookupError(f"{key} does not support asset class: {asset_class}")
        if market_type.strip().lower() not in capabilities.market_types:
            raise LookupError(f"{key} does not support market type: {market_type}")
        return adapter

    def available(self) -> list[dict[str, object]]:
        return [
            {
                "venue": adapter.capabilities.venue,
                "asset_classes": list(adapter.capabilities.asset_classes),
                "market_types": list(adapter.capabilities.market_types),
                "public_market_data": adapter.capabilities.public_market_data,
                "authenticated_market_data": adapter.capabilities.authenticated_market_data,
                "order_execution": adapter.capabilities.order_execution,
            }
            for _, adapter in sorted(self._adapters.items())
        ]
