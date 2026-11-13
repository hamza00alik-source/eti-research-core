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

    def discover_markets(
        self, limit: int = 100, quote_currencies: tuple[str, ...] = ("USDT", "USDC", "USD")
    ) -> list[dict]:
        """Discover active, linear, quote-filtered instruments from exchange metadata.

        Ranking uses public quote volume when the venue exposes bulk tickers;
        metadata-only deterministic ordering is used when it does not.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer from 1 to 1000")
        import ccxt
        exchange_class = getattr(ccxt, self.venue)
        exchange = exchange_class({
            "enableRateLimit": True,
            "options": {"defaultType": self.market_type},
        })
        markets = exchange.load_markets()
        selected = []
        for symbol, market in markets.items():
            if not market.get("active", True):
                continue
            if not market.get(self.market_type, False):
                continue
            quote = market.get("quote")
            if quote not in quote_currencies:
                continue
            if self.market_type in {"swap", "future"} and market.get("inverse", False):
                continue
            selected.append({
                "symbol": symbol, "venue": self.venue, "asset_class": "crypto",
                "market_type": self.market_type, "base": market.get("base"),
                "quote": quote, "settle": market.get("settle"),
                "linear": bool(market.get("linear", False)),
                "expiry": market.get("expiry"),
                "contract_size": market.get("contractSize"),
                "active": True, "quote_volume": None,
            })
        if not selected:
            return []
        if exchange.has.get("fetchTickers"):
            try:
                tickers = exchange.fetch_tickers([item["symbol"] for item in selected])
                for item in selected:
                    ticker = tickers.get(item["symbol"], {})
                    value = ticker.get("quoteVolume")
                    if isinstance(value, (int, float)) and value >= 0:
                        item["quote_volume"] = float(value)
                selected.sort(
                    key=lambda item: (
                        item["quote_volume"] is not None,
                        item["quote_volume"] or 0.0,
                        item["symbol"],
                    ), reverse=True,
                )
            except Exception:
                selected.sort(key=lambda item: item["symbol"])
        else:
            selected.sort(key=lambda item: item["symbol"])
        return selected[:limit]

    def fetch_derivatives_context(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> dict:
        """Fetch funding and open-interest history when supported; never fabricate missing data."""
        if self.market_type not in {"swap", "future"}:
            raise ValueError("derivatives context is available only for swap and future markets")
        import ccxt
        exchange = getattr(ccxt, self.venue)({
            "enableRateLimit": True, "options": {"defaultType": self.market_type},
        })
        markets = exchange.load_markets()
        market = markets.get(symbol)
        if not market or not market.get(self.market_type, False):
            raise ValueError(f"Unknown {self.market_type} market for {self.venue}: {symbol}")
        since = exchange.milliseconds() - exchange.parse_timeframe(timeframe) * 1000 * limit
        result = {
            "venue": self.venue, "symbol": symbol, "market_type": self.market_type,
            "funding_history": [], "open_interest_history": [],
            "current_funding": None, "current_open_interest": None,
            "liquidation_history": [],
            "availability": {
                "funding_history": bool(exchange.has.get("fetchFundingRateHistory")),
                "current_funding": bool(exchange.has.get("fetchFundingRate")),
                "open_interest_history": bool(exchange.has.get("fetchOpenInterestHistory")),
                "current_open_interest": bool(exchange.has.get("fetchOpenInterest")),
                "liquidation_history": bool(exchange.has.get("fetchLiquidations")),
            },
        }
        if result["availability"]["funding_history"]:
            try:
                rows = exchange.fetch_funding_rate_history(symbol, since=since, limit=limit)
                result["funding_history"] = [
                    {"timestamp": int(row["timestamp"]), "funding_rate": float(row["fundingRate"])}
                    for row in rows if row.get("timestamp") is not None and row.get("fundingRate") is not None
                ]
            except Exception as exc:
                result["availability"]["funding_history_error"] = type(exc).__name__
        if result["availability"]["current_funding"]:
            try:
                row = exchange.fetch_funding_rate(symbol)
                value = row.get("fundingRate")
                if value is not None:
                    result["current_funding"] = {
                        "funding_rate": float(value), "timestamp": row.get("timestamp"),
                        "next_funding_timestamp": row.get("nextFundingTimestamp"),
                    }
            except Exception as exc:
                result["availability"]["current_funding_error"] = type(exc).__name__
        if result["availability"]["open_interest_history"]:
            try:
                rows = exchange.fetch_open_interest_history(symbol, timeframe=timeframe, since=since, limit=limit)
                result["open_interest_history"] = [
                    {"timestamp": int(row["timestamp"]),
                     "open_interest": float(row.get("openInterestAmount") or row.get("openInterestValue"))}
                    for row in rows
                    if row.get("timestamp") is not None
                    and (row.get("openInterestAmount") is not None or row.get("openInterestValue") is not None)
                ]
            except Exception as exc:
                result["availability"]["open_interest_history_error"] = type(exc).__name__
        if result["availability"]["current_open_interest"]:
            try:
                row = exchange.fetch_open_interest(symbol)
                value = row.get("openInterestAmount") or row.get("openInterestValue")
                if value is not None:
                    result["current_open_interest"] = {
                        "value": float(value), "timestamp": row.get("timestamp"),
                        "unit": "contracts_or_quote_value_as_reported_by_venue",
                    }
            except Exception as exc:
                result["availability"]["current_open_interest_error"] = type(exc).__name__
        if result["availability"]["liquidation_history"]:
            try:
                rows = exchange.fetch_liquidations(symbol, since=since, limit=limit)
                result["liquidation_history"] = [
                    {"timestamp": int(row["timestamp"]), "side": row.get("side"),
                     "price": float(row["price"]) if row.get("price") is not None else None,
                     "amount": float(row["amount"]) if row.get("amount") is not None else None,
                     "cost": float(row["cost"]) if row.get("cost") is not None else None}
                    for row in rows if row.get("timestamp") is not None
                ]
            except Exception as exc:
                result["availability"]["liquidation_history_error"] = type(exc).__name__
        result["history_coverage"] = {
            "funding_observations": len(result["funding_history"]),
            "open_interest_observations": len(result["open_interest_history"]),
            "liquidation_observations": len(result["liquidation_history"]),
            "has_historical_funding": bool(result["funding_history"]),
            "has_historical_open_interest": bool(result["open_interest_history"]),
        }
        result["limitations"] = [
            "Funding and open-interest endpoint support varies by exchange, contract and history window.",
            "Open-interest units are venue-reported and must not be compared across contracts without normalization.",
            "Missing observations are not imputed; this response is research context, not an execution signal.",
        ]
        return result

def enrich_ohlcv_with_derivatives(
    frame: pd.DataFrame, context: dict, timeframe: str
) -> pd.DataFrame:
    """As-of join exchange funding/OI observations without forward-looking leakage."""
    base = frame.copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    timeframe_ms = int(pd.Timedelta(timeframe).total_seconds() * 1000)
    funding = context.get("funding_history") or []
    if funding:
        funding_frame = pd.DataFrame(funding)
        funding_frame["timestamp"] = pd.to_datetime(
            pd.to_numeric(funding_frame["timestamp"], errors="coerce"), unit="ms", utc=True
        )
        funding_frame["funding_rate"] = pd.to_numeric(
            funding_frame["funding_rate"], errors="coerce"
        )
        funding_frame = funding_frame.dropna().sort_values("timestamp").drop_duplicates("timestamp")
        base = pd.merge_asof(
            base.sort_values("timestamp"), funding_frame[["timestamp", "funding_rate"]],
            on="timestamp", direction="backward", tolerance=pd.Timedelta(hours=12),
        )
    oi = context.get("open_interest_history") or []
    if oi:
        oi_frame = pd.DataFrame(oi)
        oi_frame["timestamp"] = pd.to_datetime(
            pd.to_numeric(oi_frame["timestamp"], errors="coerce"), unit="ms", utc=True
        )
        oi_frame["open_interest"] = pd.to_numeric(
            oi_frame["open_interest"], errors="coerce"
        )
        oi_frame = oi_frame.dropna().sort_values("timestamp").drop_duplicates("timestamp")
        base = pd.merge_asof(
            base.sort_values("timestamp"), oi_frame[["timestamp", "open_interest"]],
            on="timestamp", direction="backward",
            tolerance=pd.Timedelta(milliseconds=max(timeframe_ms * 4, 60 * 60 * 1000)),
        )
    return base.reset_index(drop=True)


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


class YahooFinanceProvider:
    """Public Yahoo Finance OHLCV adapter for forex, equities, commodities and indices.

    Symbols use Yahoo Finance notation (for example EURUSD=X, AAPL, GC=F, ^GSPC).
    This is a public-data adapter, not an execution connection; availability and
    history depth depend on Yahoo's instrument and interval coverage.
    """
    provider_name = "yfinance-public"
    supported_asset_classes = ("forex", "equity", "commodity", "index")

    def __init__(self, asset_class: str):
        if asset_class not in self.supported_asset_classes:
            raise ValueError(f"Unsupported Yahoo Finance asset class: {asset_class}")
        self.asset_class = asset_class

    def discover_markets(self, limit: int = 100, market_type: str = "spot") -> list[dict]:
        """Return a curated cross-region public reference universe for Yahoo-supported assets."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be an integer from 1 to 500")
        universes = {
            "forex": {"spot": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X"]},
            "equity": {"spot": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "AVGO", "TSLA", "JPM", "LLY", "WMT", "V", "XOM", "ORCL", "COST", "MA", "NFLX", "AMD", "KO"]},
            "commodity": {"future": ["GC=F", "SI=F", "PL=F", "PA=F", "CL=F", "NG=F", "HG=F", "ZC=F", "ZS=F", "ZW=F", "KC=F", "SB=F"]},
            "index": {
                "spot": ["^GSPC", "^IXIC", "^DJI", "^RUT", "^FTSE", "^GDAXI", "^N225", "^HSI"],
                "future": ["ES=F", "NQ=F", "YM=F", "RTY=F"],
            },
        }
        symbols = universes.get(self.asset_class, {}).get(market_type)
        if symbols is None:
            raise ValueError(f"Unsupported market type {market_type} for {self.asset_class}")
        return [{
            "symbol": symbol, "venue": "yahoo", "asset_class": self.asset_class,
            "market_type": market_type, "quote": None, "active": None,
            "discovery_source": "curated_public_reference_universe",
        } for symbol in symbols[:limit]]

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 1500) -> pd.DataFrame:
        if not isinstance(symbol, str) or not symbol.strip() or len(symbol) > 40:
            raise ValueError("A valid Yahoo Finance symbol is required")
        if timeframe not in {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}:
            raise ValueError("Yahoo Finance timeframe must be 1m, 5m, 15m, 30m, 1h, 4h or 1d")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 50 <= limit <= 10000:
            raise ValueError("limit must be an integer from 50 to 10000")
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("Install yfinance to use the Yahoo Finance public-data adapter") from exc
        download_interval = "1h" if timeframe == "4h" else timeframe
        data = yf.download(
            symbol.strip(), period="max", interval=download_interval, auto_adjust=False,
            actions=False, progress=False, threads=False,
        )
        if data is None or data.empty:
            raise RuntimeError(f"No Yahoo Finance OHLCV history returned for {symbol}")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.rename(columns={column: str(column).lower() for column in data.columns})
        required = {"open", "high", "low", "close"}
        if not required.issubset(set(data.columns)):
            raise ValueError("Yahoo Finance response is missing OHLC price columns")
        volume = data["volume"] if "volume" in data.columns else pd.Series(0.0, index=data.index)
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(data.index, utc=True),
            "open": data["open"].to_numpy(), "high": data["high"].to_numpy(),
            "low": data["low"].to_numpy(), "close": data["close"].to_numpy(),
            "volume": volume.to_numpy(),
        })
        frame = validate_ohlcv(frame)
        if timeframe == "4h":
            frame = frame.set_index("timestamp").resample("4h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna().reset_index()
        timeframe_ms = {
            "1m": 60_000, "5m": 300_000, "15m": 900_000,
            "30m": 1_800_000, "1h": 3_600_000,
            "4h": 14_400_000, "1d": 86_400_000,
        }[timeframe]
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        completed = drop_incomplete_candles(frame, timeframe_ms, now_ms)
        return completed.tail(limit).reset_index(drop=True)
