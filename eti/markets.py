"""Market and venue discovery metadata; discovery never submits orders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MarketDescriptor:
    venue: str
    symbol: str
    base: str
    quote: str
    asset_class: str
    market_type: str
    active: bool = True
    leverage_max: float = 1.0


SUPPORTED_VENUES = ("binance", "mexc")


def classify_symbol(symbol: str) -> tuple[str, str]:
    base, quote = symbol.split("/", 1)
    quote = quote.split(":", 1)[0]
    return base, quote


def normalize_markets(venue: str, markets: Iterable[dict]) -> list[MarketDescriptor]:
    """Normalize CCXT market dictionaries without assuming leverage eligibility."""
    normalized = []
    for market in markets:
        if not market.get("active", True) or not market.get("spot", False):
            continue
        symbol = market.get("symbol")
        if not symbol or "/" not in symbol:
            continue
        base, quote = classify_symbol(symbol)
        normalized.append(MarketDescriptor(
            venue=venue, symbol=symbol, base=base, quote=quote,
            asset_class="crypto", market_type="spot",
            active=bool(market.get("active", True)), leverage_max=1.0,
        ))
    return normalized


def discover_spot_markets(venues: Iterable[str] = SUPPORTED_VENUES) -> list[MarketDescriptor]:
    """Fetch public spot market metadata from CCXT. No keys or private endpoints."""
    try:
        import ccxt
    except ImportError as exc:
        raise RuntimeError("Install project requirements first") from exc
    found = []
    for venue in venues:
        if venue not in SUPPORTED_VENUES:
            raise ValueError(f"Unsupported venue: {venue}")
        exchange = getattr(ccxt, venue)({"enableRateLimit": True})
        found.extend(normalize_markets(venue, exchange.load_markets().values()))
    return found
