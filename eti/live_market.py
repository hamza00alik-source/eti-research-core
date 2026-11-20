"""Read-only live public-market snapshots for ETI paper workflows.

This module polls public spot market data through CCXT. It never loads API keys,
creates authenticated clients, or submits orders. A snapshot is eligible for a
paper decision only when both the latest completed candle and quote are fresh.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Callable

import pandas as pd

from .data import drop_incomplete_candles, validate_ohlcv


def _finite_positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be a finite positive number")
    return number


def _timeframe_ms(exchange: Any, timeframe: str) -> int:
    if not isinstance(timeframe, str) or not timeframe or timeframe.endswith("M"):
        raise ValueError("timeframe must be a supported fixed-duration CCXT interval")
    seconds = float(exchange.parse_timeframe(timeframe))
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("exchange returned an invalid timeframe duration")
    return int(seconds * 1000)


def capture_public_spot_snapshot(
    venue: str,
    symbol: str,
    timeframe: str = "1m",
    *,
    exchange_factory: Callable[[str], Any] | None = None,
    now_ms: int | None = None,
    max_quote_age_ms: int = 30_000,
    max_candle_age_intervals: int = 2,
) -> dict[str, Any]:
    """Capture one fresh quote and the latest completed OHLCV candle.

    exchange_factory exists for deterministic tests; production defaults to
    a public CCXT client with rate limiting and no credentials.
    """
    if not isinstance(venue, str) or not venue.strip():
        raise ValueError("venue is required")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol is required")
    if isinstance(max_quote_age_ms, bool) or not isinstance(max_quote_age_ms, int) or max_quote_age_ms <= 0:
        raise ValueError("max_quote_age_ms must be a positive integer")
    if (isinstance(max_candle_age_intervals, bool)
            or not isinstance(max_candle_age_intervals, int)
            or max_candle_age_intervals < 1):
        raise ValueError("max_candle_age_intervals must be a positive integer")

    if exchange_factory is None:
        import ccxt
        exchange_factory = lambda exchange_id: getattr(ccxt, exchange_id)({"enableRateLimit": True})

    exchange = exchange_factory(venue)
    try:
        markets = exchange.load_markets()
        market = markets.get(symbol)
        if not isinstance(market, dict) or not market.get("spot", False) or market.get("active") is False:
            raise ValueError(f"{venue} {symbol} is not an active public spot market")

        current_ms = int(exchange.milliseconds() if now_ms is None else now_ms)
        interval_ms = _timeframe_ms(exchange, timeframe)
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=5)
        if not raw:
            raise ValueError("venue returned no OHLCV candles")
        frame = validate_ohlcv(pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        ))
        completed = drop_incomplete_candles(frame, interval_ms, current_ms)
        candle = completed.iloc[-1]
        candle_open_ms = int(pd.Timestamp(candle["timestamp"]).timestamp() * 1000)
        candle_close_ms = candle_open_ms + interval_ms
        candle_age_ms = current_ms - candle_close_ms
        if candle_age_ms < 0 or candle_age_ms > interval_ms * max_candle_age_intervals:
            raise ValueError("latest completed candle is stale")

        ticker = exchange.fetch_ticker(symbol)
        if not isinstance(ticker, dict):
            raise ValueError("venue returned an invalid ticker")
        quote_timestamp = ticker.get("timestamp")
        if (isinstance(quote_timestamp, bool)
                or not isinstance(quote_timestamp, (int, float))
                or not math.isfinite(quote_timestamp)):
            raise ValueError("ticker timestamp is unavailable; quote freshness cannot be verified")
        quote_timestamp = int(quote_timestamp)
        quote_age_ms = current_ms - quote_timestamp
        if quote_age_ms < 0 or quote_age_ms > max_quote_age_ms:
            raise ValueError("latest quote is stale or timestamped in the future")

        bid = _finite_positive(ticker.get("bid"), "bid")
        ask = _finite_positive(ticker.get("ask"), "ask")
        last = _finite_positive(ticker.get("last"), "last")
        if ask < bid:
            raise ValueError("ticker ask is below bid")
        mid = (bid + ask) / 2.0
        spread_bps = ((ask - bid) / mid) * 10_000.0

        snapshot = {
            "schema_version": "eti.live_public_market_snapshot.v1",
            "venue": venue,
            "market_type": "spot",
            "symbol": symbol,
            "timeframe": timeframe,
            "observed_at": datetime.fromtimestamp(current_ms / 1000, tz=timezone.utc).isoformat(),
            "quote": {
                "timestamp": datetime.fromtimestamp(quote_timestamp / 1000, tz=timezone.utc).isoformat(),
                "bid": bid, "ask": ask, "last": last, "mid": mid,
                "spread_bps": spread_bps, "age_ms": quote_age_ms,
            },
            "latest_completed_candle": {
                "open_time": pd.Timestamp(candle["timestamp"]).isoformat(),
                "close_time": datetime.fromtimestamp(candle_close_ms / 1000, tz=timezone.utc).isoformat(),
                "open": float(candle["open"]), "high": float(candle["high"]),
                "low": float(candle["low"]), "close": float(candle["close"]),
                "volume": float(candle["volume"]), "age_ms": candle_age_ms,
            },
            "quality": {
                "quote_fresh": True,
                "candle_fresh": True,
                "paper_eligible": True,
                "authenticated_data": False,
                "order_execution": False,
            },
        }
        canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
        snapshot["snapshot_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return snapshot
    finally:
        close = getattr(exchange, "close", None)
        if callable(close):
            close()
