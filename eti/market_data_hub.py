"""Unified multi-market discovery and public historical-data orchestration.

The hub coordinates independent read-only adapters. It never merges prices from
different venues: failover is explicit and returns one source's validated series
with full provenance. Execution is deliberately outside this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Any

import pandas as pd

from .adapters import AdapterCapabilities, AdapterRegistry, MarketAdapter
from .data import validate_ohlcv
from .instruments import InstrumentCatalog, InstrumentSpec
from .providers import CcxtPublicProvider, YahooFinanceProvider


@dataclass(frozen=True)
class MarketDataSnapshot:
    """One provider's validated historical series and its source-quality record."""

    instrument: InstrumentSpec
    timeframe: str
    frame: pd.DataFrame
    provenance: dict[str, Any]
    quality: dict[str, Any]


class CcxtMarketAdapter:
    """One CCXT venue adapter spanning supported crypto market types."""

    def __init__(self, venue: str, market_types: tuple[str, ...] = ("spot", "swap", "future")):
        allowed = {"spot", "swap", "future"}
        self.venue = venue.strip().lower()
        if self.venue in {"coinbaseexchange", "kraken"}:
            allowed = {"spot"}
        if not market_types or not set(market_types).issubset(allowed):
            raise ValueError(
                f"{self.venue} market_types must be a non-empty subset of {sorted(allowed)}"
            )
        self.market_types = tuple(market_types)
        self.capabilities = AdapterCapabilities(
            venue=self.venue, asset_classes=("crypto",), market_types=self.market_types,
            public_market_data=True, authenticated_market_data=False, order_execution=False,
        )
        self._providers = {
            market_type: CcxtPublicProvider(self.venue, market_type)
            for market_type in self.market_types
        }

    def _normalize_markets(self, market_type: str, rows: list[dict]) -> list[InstrumentSpec]:
        found: list[InstrumentSpec] = []
        for row in rows:
            found.append(InstrumentSpec(
                symbol=row["symbol"], venue=self.venue, asset_class="crypto",
                market_type=market_type, base_currency=row.get("base"),
                quote_currency=row.get("quote"), settlement_currency=row.get("settle"),
                contract_multiplier=float(row.get("contract_size") or 1.0),
                max_leverage=1.0, supports_short=market_type in {"swap", "future"},
                supports_funding=market_type == "swap", active=bool(row.get("active", True)),
            ))
        return found

    def discover_instruments_for(self, market_type: str) -> list[InstrumentSpec]:
        """Discover only one requested market type to avoid unnecessary venue calls."""
        try:
            provider = self._providers[market_type]
        except KeyError as exc:
            raise LookupError(f"{self.venue} does not support {market_type} discovery") from exc
        return self._normalize_markets(
            market_type, provider.discover_markets(limit=1000)
        )

    def discover_instruments(self) -> list[InstrumentSpec]:
        found: list[InstrumentSpec] = []
        self.discovery_errors: list[dict[str, str]] = []
        for market_type in self.market_types:
            try:
                found.extend(self.discover_instruments_for(market_type))
            except Exception as exc:
                self.discovery_errors.append({
                    "market_type": market_type, "error_type": type(exc).__name__,
                    "message": str(exc)[:500],
                })
        return found

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Protocol-compatible spot default; prefer fetch_instrument_ohlcv in the hub."""
        if "spot" not in self._providers:
            raise ValueError(f"{self.venue} adapter has no default spot market")
        return self._providers["spot"].fetch_ohlcv(symbol, timeframe, limit)

    def fetch_instrument_ohlcv(
        self, instrument: InstrumentSpec, timeframe: str, limit: int
    ) -> pd.DataFrame:
        try:
            provider = self._providers[instrument.market_type]
        except KeyError as exc:
            raise LookupError(
                f"{self.venue} does not support {instrument.market_type} data"
            ) from exc
        return provider.fetch_ohlcv(instrument.symbol, timeframe, limit)


class YahooFinanceMarketAdapter:
    """One read-only adapter for Yahoo reference universes across non-crypto assets."""

    _MARKET_TYPES = {
        "forex": ("spot",),
        "equity": ("spot",),
        "commodity": ("future",),
        "index": ("spot", "future"),
    }

    def __init__(self):
        assets = tuple(self._MARKET_TYPES)
        market_types = tuple(sorted({m for values in self._MARKET_TYPES.values() for m in values}))
        self.capabilities = AdapterCapabilities(
            venue="yahoo", asset_classes=assets, market_types=market_types,
            public_market_data=True, authenticated_market_data=False, order_execution=False,
        )
        self._providers = {
            asset: YahooFinanceProvider(asset) for asset in assets
        }

    def discover_instruments_for(self, market_type: str) -> list[InstrumentSpec]:
        """Discover only asset classes that support the requested market type."""
        found: list[InstrumentSpec] = []
        self.discovery_errors: list[dict[str, str]] = []
        for asset_class, market_types in self._MARKET_TYPES.items():
            if market_type not in market_types:
                continue
            try:
                provider = self._providers[asset_class]
                rows = provider.discover_markets(limit=500, market_type=market_type)
                for row in rows:
                    found.append(InstrumentSpec(
                        symbol=row["symbol"], venue="yahoo", asset_class=asset_class,
                        market_type=market_type, active=True,
                    ))
            except Exception as exc:
                self.discovery_errors.append({
                    "asset_class": asset_class, "error_type": type(exc).__name__,
                    "message": str(exc)[:500],
                })
        return found

    def discover_instruments(self) -> list[InstrumentSpec]:
        found: list[InstrumentSpec] = []
        errors: list[dict[str, str]] = []
        for market_type in sorted({m for values in self._MARKET_TYPES.values() for m in values}):
            found.extend(self.discover_instruments_for(market_type))
            errors.extend(getattr(self, "discovery_errors", []))
        self.discovery_errors = errors
        return found

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        matches = [
            asset for asset, provider in self._providers.items()
            if any(row["symbol"] == symbol for market_type in self._MARKET_TYPES[asset]
                   for row in provider.discover_markets(limit=500, market_type=market_type))
        ]
        if len(matches) != 1:
            raise LookupError(f"Yahoo symbol must resolve to exactly one asset class: {symbol}")
        return self._providers[matches[0]].fetch_ohlcv(symbol, timeframe, limit)

    def fetch_instrument_ohlcv(
        self, instrument: InstrumentSpec, timeframe: str, limit: int
    ) -> pd.DataFrame:
        try:
            provider = self._providers[instrument.asset_class]
            allowed_types = self._MARKET_TYPES[instrument.asset_class]
        except KeyError as exc:
            raise LookupError(f"Yahoo does not support {instrument.asset_class}") from exc
        if instrument.market_type not in allowed_types:
            raise LookupError(
                f"Yahoo does not support {instrument.asset_class}/{instrument.market_type}"
            )
        return provider.fetch_ohlcv(instrument.symbol, timeframe, limit)


class MarketDataHub:
    """Discover instruments and fetch validated data with explicit source failover."""

    def __init__(self, adapters: Iterable[MarketAdapter] = ()):
        self.registry = AdapterRegistry()
        self.catalog = InstrumentCatalog()
        self._adapters: dict[str, MarketAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: MarketAdapter) -> None:
        self.registry.register(adapter)
        self._adapters[adapter.capabilities.venue] = adapter

    def available_sources(self) -> list[dict[str, object]]:
        return self.registry.available()

    def discover(
        self, *, asset_class: str | None = None, market_type: str | None = None
    ) -> dict[str, Any]:
        """Discover across every compatible adapter; preserve per-source failures."""
        instruments: list[InstrumentSpec] = []
        errors: list[dict[str, str]] = []
        for venue, adapter in sorted(self._adapters.items()):
            caps = adapter.capabilities
            if asset_class and asset_class.lower() not in caps.asset_classes:
                continue
            if market_type and market_type.lower() not in caps.market_types:
                continue
            try:
                discover_one = getattr(adapter, "discover_instruments_for", None)
                if market_type and callable(discover_one):
                    discovered = discover_one(market_type.lower())
                    item_errors = getattr(adapter, "discovery_errors", [])
                else:
                    discovered = adapter.discover_instruments()
                    item_errors = getattr(adapter, "discovery_errors", [])
                for item_error in item_errors:
                    dimension = item_error.get("market_type", item_error.get("asset_class", "unknown"))
                    errors.append({
                        "venue": venue,
                        "error_type": item_error.get("error_type", "DiscoveryError"),
                        "message": f"{dimension}: {item_error.get('message', '')}"[:500],
                    })
                for instrument in discovered:
                    if asset_class and instrument.asset_class != asset_class.lower():
                        continue
                    if market_type and instrument.market_type != market_type.lower():
                        continue
                    try:
                        self.catalog.register(instrument)
                    except ValueError:
                        existing = self.catalog.get(
                            instrument.venue, instrument.market_type, instrument.symbol
                        )
                        if existing != instrument:
                            raise
                    instruments.append(instrument)
            except Exception as exc:
                errors.append({
                    "venue": venue, "error_type": type(exc).__name__,
                    "message": str(exc)[:500],
                })
        unique = {item.key: item for item in instruments}
        return {
            "instruments": list(unique.values()),
            "count": len(unique),
            "sources_checked": [
                source["venue"] for source in self.available_sources()
                if (not asset_class or asset_class.lower() in source["asset_classes"])
                and (not market_type or market_type.lower() in source["market_types"])
            ],
            "source_errors": errors,
        }

    @staticmethod
    def _quality(frame: pd.DataFrame, timeframe: str) -> dict[str, Any]:
        try:
            interval = pd.Timedelta(timeframe)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported fixed-duration timeframe: {timeframe}") from exc
        interval_ms = int(interval.total_seconds() * 1000)
        if interval_ms <= 0:
            raise ValueError("timeframe must be a positive fixed duration")
        timestamps = frame["timestamp"]
        deltas = timestamps.diff().dropna().dt.total_seconds() * 1000
        missing = sum(max(0, round(float(delta) / interval_ms) - 1)
                      for delta in deltas if delta > interval_ms * 1.5)
        now = datetime.now(timezone.utc)
        last = timestamps.iloc[-1].to_pydatetime()
        age = max(0.0, (now - last).total_seconds())
        return {
            "rows": int(len(frame)),
            "start": timestamps.iloc[0].isoformat(),
            "end": timestamps.iloc[-1].isoformat(),
            "missing_interval_estimate": int(missing),
            "max_gap_seconds": float(deltas.max() / 1000) if len(deltas) else 0.0,
            "last_candle_age_seconds": age,
            "timestamp_order": "ascending",
            "ohlcv_validated": True,
        }

    def fetch(
        self, instrument: InstrumentSpec, timeframe: str, limit: int,
        *, min_rows: int = 1,
    ) -> MarketDataSnapshot:
        if isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if isinstance(min_rows, bool) or min_rows < 1 or min_rows > limit:
            raise ValueError("min_rows must be between 1 and limit")
        adapter = self.registry.get(
            instrument.venue, asset_class=instrument.asset_class,
            market_type=instrument.market_type,
        )
        method = getattr(adapter, "fetch_instrument_ohlcv", None)
        frame = (method(instrument, timeframe, limit) if callable(method)
                 else adapter.fetch_ohlcv(instrument.symbol, timeframe, limit))
        clean = validate_ohlcv(frame).tail(limit).reset_index(drop=True)
        if len(clean) < min_rows:
            raise ValueError(
                f"{instrument.venue} returned {len(clean)} valid candles; {min_rows} required"
            )
        quality = self._quality(clean, timeframe)
        provenance = {
            "venue": instrument.venue, "symbol": instrument.symbol,
            "asset_class": instrument.asset_class, "market_type": instrument.market_type,
            "timeframe": timeframe, "requested_rows": limit,
            "returned_rows": len(clean), "public_market_data": adapter.capabilities.public_market_data,
            "authenticated_market_data": adapter.capabilities.authenticated_market_data,
            "execution_capability": adapter.capabilities.order_execution,
        }
        return MarketDataSnapshot(instrument, timeframe, clean, provenance, quality)

    def fetch_first_available(
        self, candidates: Iterable[InstrumentSpec], timeframe: str, limit: int,
        *, min_rows: int = 1,
    ) -> MarketDataSnapshot:
        """Try caller-supplied equivalent instruments in priority order.

        The caller must explicitly map equivalent contracts across venues. The
        hub never assumes similarly named symbols are economically interchangeable.
        """
        failures: list[dict[str, str]] = []
        for instrument in candidates:
            try:
                snapshot = self.fetch(instrument, timeframe, limit, min_rows=min_rows)
                snapshot.provenance["fallback_failures_before_success"] = failures
                snapshot.provenance["fallback_count"] = len(failures)
                return snapshot
            except Exception as exc:
                failures.append({
                    "venue": instrument.venue, "symbol": instrument.symbol,
                    "error_type": type(exc).__name__, "message": str(exc)[:500],
                })
        raise RuntimeError(f"No candidate data source succeeded: {failures}")


def build_public_market_hub(
    crypto_venues: tuple[str, ...] = ("binance", "mexc", "coinbaseexchange", "kraken"),
    *, include_yahoo: bool = True,
) -> MarketDataHub:
    """Build ETI's default public-data hub without credentials or execution access."""
    adapters: list[MarketAdapter] = []
    for venue in crypto_venues:
        normalized = venue.strip().lower()
        market_types = (
            ("spot",) if normalized in {"coinbaseexchange", "kraken"}
            else ("spot", "swap", "future")
        )
        adapters.append(CcxtMarketAdapter(normalized, market_types))
    if include_yahoo:
        adapters.append(YahooFinanceMarketAdapter())
    return MarketDataHub(adapters)
