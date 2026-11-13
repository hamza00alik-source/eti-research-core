"""Canonical, venue-independent instrument metadata for ETI.

The catalog describes instruments; it does not imply that a venue is connected
or that an instrument is eligible to trade. Eligibility is validated separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


ASSET_CLASSES = frozenset({"crypto", "forex", "equity", "commodity", "index"})
MARKET_TYPES = frozenset({"spot", "margin", "swap", "future", "option", "cfd"})


@dataclass(frozen=True)
class InstrumentSpec:
    """Normalized instrument contract, independent of any broker API schema."""

    symbol: str
    venue: str
    asset_class: str
    market_type: str
    base_currency: str | None = None
    quote_currency: str | None = None
    settlement_currency: str | None = None
    contract_multiplier: float = 1.0
    tick_size: float | None = None
    lot_size: float | None = None
    min_quantity: float | None = None
    max_leverage: float = 1.0
    supports_short: bool = False
    supports_funding: bool = False
    active: bool = True

    def __post_init__(self) -> None:
        for field_name in ("symbol", "venue", "asset_class", "market_type"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        asset_class = self.asset_class.strip().lower()
        market_type = self.market_type.strip().lower()
        if asset_class not in ASSET_CLASSES:
            raise ValueError(f"Unsupported asset_class: {asset_class}")
        if market_type not in MARKET_TYPES:
            raise ValueError(f"Unsupported market_type: {market_type}")
        object.__setattr__(self, "asset_class", asset_class)
        object.__setattr__(self, "market_type", market_type)
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "venue", self.venue.strip().lower())

        for field_name in ("contract_multiplier", "max_leverage"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{field_name} must be finite and positive")
        for field_name in ("tick_size", "lot_size", "min_quantity"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{field_name} must be finite and positive when supplied")
        if market_type == "spot" and self.max_leverage != 1.0:
            raise ValueError("Spot instruments must use max_leverage=1.0 in the canonical catalog")
        if market_type == "spot" and self.supports_funding:
            raise ValueError("Spot instruments cannot declare perpetual funding")
        if not isinstance(self.supports_short, bool) or not isinstance(self.supports_funding, bool):
            raise ValueError("Instrument capability flags must be booleans")
        if not isinstance(self.active, bool):
            raise ValueError("active must be a boolean")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.venue, self.market_type, self.symbol)


class InstrumentCatalog:
    """In-memory canonical catalog; conflicting venue instruments fail closed."""

    def __init__(self, instruments: Iterable[InstrumentSpec] = ()) -> None:
        self._items: dict[tuple[str, str, str], InstrumentSpec] = {}
        for instrument in instruments:
            self.register(instrument)

    def register(self, instrument: InstrumentSpec, *, replace: bool = False) -> None:
        if not isinstance(instrument, InstrumentSpec):
            raise TypeError("instrument must be an InstrumentSpec")
        if instrument.key in self._items and not replace:
            raise ValueError(f"Instrument already registered: {instrument.key}")
        self._items[instrument.key] = instrument

    def get(self, venue: str, market_type: str, symbol: str) -> InstrumentSpec:
        key = (venue.strip().lower(), market_type.strip().lower(), symbol.strip().upper())
        try:
            return self._items[key]
        except KeyError as exc:
            raise LookupError(f"Unknown instrument: {key}") from exc

    def list(
        self,
        *,
        asset_class: str | None = None,
        market_type: str | None = None,
        venue: str | None = None,
        active_only: bool = True,
    ) -> list[InstrumentSpec]:
        items = self._items.values()
        if asset_class is not None:
            items = (x for x in items if x.asset_class == asset_class.strip().lower())
        if market_type is not None:
            items = (x for x in items if x.market_type == market_type.strip().lower())
        if venue is not None:
            items = (x for x in items if x.venue == venue.strip().lower())
        if active_only:
            items = (x for x in items if x.active)
        return sorted(items, key=lambda x: x.key)

    def __len__(self) -> int:
        return len(self._items)
