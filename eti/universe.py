"""Read-only market-universe scanning for ETI research candidate discovery."""
from __future__ import annotations

from typing import Iterable, Any

from .instruments import ASSET_CLASSES, InstrumentSpec
from .market_data_hub import MarketDataHub, build_public_market_hub


def scan_market_universe(
    *,
    asset_classes: Iterable[str] = ("crypto", "forex", "equity", "commodity", "index"),
    market_types: Iterable[str] = ("spot",),
    timeframe: str = "1h",
    limit: int = 2000,
    min_rows: int = 662,
    max_per_asset_class: int = 5,
    max_per_venue: int = 2,
    hub: MarketDataHub | None = None,
) -> dict[str, Any]:
    """Discover and test data readiness across asset classes without training.

    Selection is bounded and venue-diverse. A candidate is eligible only when
    one source supplies the requested minimum history and passes OHLCV checks.
    This scan deliberately emits no trade signal and never enables execution.
    """
    assets = tuple(dict.fromkeys(str(x).strip().lower() for x in asset_classes))
    types = tuple(dict.fromkeys(str(x).strip().lower() for x in market_types))
    if not assets or not set(assets).issubset(ASSET_CLASSES):
        raise ValueError(f"asset_classes must be selected from {sorted(ASSET_CLASSES)}")
    if not types or not set(types).issubset({"spot", "margin", "swap", "future", "option", "cfd"}):
        raise ValueError("market_types contains an unsupported ETI market type")
    for name, value in (
        ("limit", limit), ("min_rows", min_rows),
        ("max_per_asset_class", max_per_asset_class), ("max_per_venue", max_per_venue),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if min_rows > limit:
        raise ValueError("min_rows cannot exceed limit")

    data_hub = hub or build_public_market_hub()
    market_type_rank = {name: index for index, name in enumerate(types)}
    discovery: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []

    for asset in assets:
        results = [
            data_hub.discover(asset_class=asset, market_type=market_type)
            for market_type in types
        ]
        discovered = [
            instrument for result in results for instrument in result["instruments"]
        ]
        source_errors = [
            error for result in results for error in result["source_errors"]
        ]
        sources_checked = sorted({
            venue for result in results for venue in result["sources_checked"]
        })
        discovery[asset] = {
            "discovered_count": len({item.key: item for item in discovered}),
            "sources_checked": sources_checked,
            "source_errors": source_errors,
        }
        by_venue: dict[str, list[InstrumentSpec]] = {}
        for instrument in discovered:
            if instrument.market_type in types and instrument.active:
                by_venue.setdefault(instrument.venue, []).append(instrument)
        for venue_items in by_venue.values():
            venue_items.sort(key=lambda item: (
                market_type_rank.get(item.market_type, len(types)), item.symbol
            ))

        # Round-robin venues so a large catalog from one exchange cannot consume
        # the entire asset-class budget before other independent providers appear.
        selected: list[InstrumentSpec] = []
        venues = sorted(by_venue)
        for offset in range(max_per_venue):
            for venue in venues:
                items = by_venue[venue]
                if offset < len(items) and len(selected) < max_per_asset_class:
                    selected.append(items[offset])
            if len(selected) >= max_per_asset_class:
                break

        discovery[asset]["selected_count"] = len(selected)
        discovery[asset]["selection_policy"] = (
            "active instruments; requested market types; round-robin venues; "
            f"maximum {max_per_asset_class} instruments per asset class and "
            f"{max_per_venue} per venue"
        )
        for instrument in selected:
            record: dict[str, Any] = {
                "instrument": {
                    "venue": instrument.venue, "symbol": instrument.symbol,
                    "asset_class": instrument.asset_class,
                    "market_type": instrument.market_type,
                    "base_currency": instrument.base_currency,
                    "quote_currency": instrument.quote_currency,
                    "settlement_currency": instrument.settlement_currency,
                    "contract_multiplier": instrument.contract_multiplier,
                    "supports_short": instrument.supports_short,
                    "supports_funding": instrument.supports_funding,
                },
                "status": "unavailable",
            }
            try:
                snapshot = data_hub.fetch(
                    instrument, timeframe, limit, min_rows=min_rows
                )
                record.update({
                    "status": "data_ready",
                    "provenance": snapshot.provenance,
                    "quality": snapshot.quality,
                })
            except Exception as exc:
                record["error"] = {
                    "type": type(exc).__name__, "message": str(exc)[:500]
                }
            candidates.append(record)

    ready = sum(item["status"] == "data_ready" for item in candidates)
    return {
        "schema_version": "eti.market_universe.v1",
        "scan_only": True,
        "training_performed": False,
        "execution_enabled": False,
        "config": {
            "asset_classes": list(assets), "market_types": list(types),
            "timeframe": timeframe, "requested_rows": limit,
            "minimum_valid_rows": min_rows,
            "max_per_asset_class": max_per_asset_class,
            "max_per_venue": max_per_venue,
        },
        "discovery": discovery,
        "summary": {
            "candidate_count": len(candidates),
            "data_ready_count": ready,
            "unavailable_count": len(candidates) - ready,
            "asset_classes_with_ready_data": sorted({
                item["instrument"]["asset_class"] for item in candidates
                if item["status"] == "data_ready"
            }),
        },
        "candidates": candidates,
        "warning": (
            "Data readiness is not a trading signal, model result, or proof of "
            "profitability. Each source is independently validated; price series "
            "are never merged across venues. No model training or order execution occurred."
        ),
    }
