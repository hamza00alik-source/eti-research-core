"""Extensible market-data providers: crypto spot/derivatives and labelled local OHLCV files."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
import pandas as pd
from .data import drop_incomplete_candles, fetch_ohlcv, validate_ohlcv

@dataclass(frozen=True)
class Instrument:
    symbol: str
    asset_class: str
    venue: str
    quote_currency: str | None = None
    market_type: str = "spot"

@runtime_checkable
class MarketDataProvider(Protocol):
    provider_name: str
    supported_asset_classes: tuple[str, ...]
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame: ...

class CcxtPublicProvider:
    """Unauthenticated crypto OHLCV provider for explicitly selected CCXT market types."""
    provider_name = "ccxt-public"
    supported_asset_classes = ("crypto",)
    supported_market_types = ("spot", "swap", "future")

    def __init__(self, venue: str = "binance", market_type: str = "spot"):
        if venue not in ("binance", "mexc"):
            raise ValueError(f"Unsupported public venue: {venue}")
        if market_type not in self.supported_market_types:
            raise ValueError(f"Unsupported public market type: {market_type}")
        self.venue = venue
        self.market_type = market_type

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 1500) -> pd.DataFrame:
        return fetch_ohlcv(self.venue, symbol, timeframe, limit, market_type=self.market_type)

class CsvOHLCVProvider:
    """Local OHLCV adapter; the caller labels the instrument's asset class."""
    provider_name = "csv"
    supported_asset_classes = ("crypto", "forex", "equity", "commodity", "index")
    def __init__(self, path: str | Path, timeframe_ms: int | None = None):
        self.path, self.timeframe_ms = Path(path), timeframe_ms
    def fetch_ohlcv(self, symbol: str = "", timeframe: str = "", limit: int = 10000) -> pd.DataFrame:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if limit <= 0:
            raise ValueError("limit must be positive")
        frame = validate_ohlcv(pd.read_csv(self.path))
        if self.timeframe_ms is not None:
            now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
            frame = drop_incomplete_candles(frame, self.timeframe_ms, now_ms)
        return frame.tail(limit).reset_index(drop=True)

class ProviderRegistry:
    """Explicit registry; unconfigured asset classes fail closed."""
    def __init__(self):
        self._providers: dict[str, MarketDataProvider] = {}
    def register(self, asset_class: str, provider: MarketDataProvider) -> None:
        if not asset_class or not isinstance(provider, MarketDataProvider):
            raise TypeError("A named asset class and MarketDataProvider are required")
        self._providers[asset_class.lower()] = provider
    def get(self, asset_class: str) -> MarketDataProvider:
        key = asset_class.lower()
        if key not in self._providers:
            raise LookupError(f"No configured provider for asset class: {key}")
        return self._providers[key]
    def capabilities(self) -> dict[str, str]:
        return {key: getattr(provider, "provider_name", type(provider).__name__)
                for key, provider in sorted(self._providers.items())}
