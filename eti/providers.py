"""Provider interfaces keep market-data integrations separate from strategy logic."""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from .data import fetch_ohlcv


class MarketDataProvider(Protocol):
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame: ...


class CcxtPublicProvider:
    """Public OHLCV provider for supported CCXT exchanges; no private API methods."""
    def __init__(self, venue: str = "binance"):
        if venue not in ("binance", "mexc"):
            raise ValueError(f"Unsupported public venue: {venue}")
        self.venue = venue

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 1500) -> pd.DataFrame:
        return fetch_ohlcv(self.venue, symbol, timeframe, limit)
