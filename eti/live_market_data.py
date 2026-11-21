"""Read-only Binance spot kline WebSocket ingestion with fail-closed validation.

This module emits completed candles only. It never imports credentials, creates
an authenticated exchange client, or submits orders. Connection supervision,
durable event storage and paper-session integration are separate concerns.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import time
from typing import Any, Callable
from urllib.parse import quote

import pandas as pd

_INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000,
    "1w": 604_800_000,
}
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}/[A-Z0-9]{2,20}$")


@dataclass(frozen=True)
class ClosedKline:
    venue: str
    symbol: str
    timeframe: str
    open_time_ms: int
    close_time_ms: int
    event_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def as_record(self) -> dict[str, Any]:
        return {
            "timestamp": pd.to_datetime(self.open_time_ms, unit="ms", utc=True),
            "open": self.open, "high": self.high, "low": self.low,
            "close": self.close, "volume": self.volume,
        }


class BinanceSpotKlineFeed:
    """Public Binance spot kline stream; no credentials or trading methods."""

    base_url = "wss://stream.binance.com:9443/ws"

    def __init__(self, symbol: str, timeframe: str, *, max_event_age_ms: int = 120_000,
                 allowed_future_skew_ms: int = 5_000):
        if not isinstance(symbol, str) or not _SYMBOL_RE.fullmatch(symbol):
            raise ValueError("symbol must be a canonical BASE/QUOTE spot symbol")
        if timeframe not in _INTERVAL_MS:
            raise ValueError("unsupported Binance spot kline interval")
        for name, value in (("max_event_age_ms", max_event_age_ms),
                            ("allowed_future_skew_ms", allowed_future_skew_ms)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        self.symbol = symbol
        self.timeframe = timeframe
        self.interval_ms = _INTERVAL_MS[timeframe]
        self.max_event_age_ms = max_event_age_ms
        self.allowed_future_skew_ms = allowed_future_skew_ms
        self.last_open_time_ms: int | None = None
        self.last_candle: ClosedKline | None = None

    @property
    def url(self) -> str:
        stream = f"{self.symbol.replace('/', '').lower()}@kline_{self.timeframe}"
        return f"{self.base_url}/{quote(stream, safe='@_')}"

    def parse(self, message: str | bytes | dict[str, Any], *, now_ms: int) -> ClosedKline | None:
        """Return a validated completed candle; ignore in-progress kline updates."""
        if isinstance(now_ms, bool) or not isinstance(now_ms, int) or now_ms < 0:
            raise ValueError("now_ms must be a non-negative integer")
        try:
            payload = json.loads(message) if isinstance(message, (str, bytes)) else message
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("stream message is not valid JSON") from exc
        if not isinstance(payload, dict) or payload.get("e") != "kline":
            raise ValueError("unexpected Binance stream event")
        kline = payload.get("k")
        if not isinstance(kline, dict):
            raise ValueError("kline payload is missing")
        if payload.get("s") != self.symbol.replace("/", "") or kline.get("s") != self.symbol.replace("/", ""):
            raise ValueError("stream symbol does not match configured spot instrument")
        if kline.get("i") != self.timeframe:
            raise ValueError("stream interval does not match configured timeframe")
        if kline.get("x") is not True:
            return None

        def integer(value: Any, field: str) -> int:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
            return value

        event_ms = integer(payload.get("E"), "event time")
        open_ms = integer(kline.get("t"), "kline open time")
        close_ms = integer(kline.get("T"), "kline close time")
        if close_ms < open_ms or close_ms >= open_ms + self.interval_ms:
            raise ValueError("kline timestamps do not fit the configured interval")
        if event_ms < close_ms:
            raise ValueError("event timestamp precedes candle close")
        if event_ms > now_ms + self.allowed_future_skew_ms:
            raise ValueError("stream event timestamp is too far in the future")
        if now_ms - event_ms > self.max_event_age_ms:
            raise ValueError("stream event is stale")

        def number(value: Any, field: str) -> float:
            if isinstance(value, bool):
                raise ValueError(f"{field} must be finite numeric data")
            try:
                result = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{field} must be finite numeric data") from exc
            if not math.isfinite(result):
                raise ValueError(f"{field} must be finite numeric data")
            return result

        open_, high = number(kline.get("o"), "open"), number(kline.get("h"), "high")
        low, close = number(kline.get("l"), "low"), number(kline.get("c"), "close")
        volume = number(kline.get("v"), "volume")
        if min(open_, high, low, close) <= 0 or volume < 0:
            raise ValueError("OHLC prices must be positive and volume non-negative")
        if high < max(open_, close, low) or low > min(open_, close, high):
            raise ValueError("OHLC candle range is inconsistent")
        return ClosedKline(
            venue="binance", symbol=self.symbol, timeframe=self.timeframe,
            open_time_ms=open_ms, close_time_ms=close_ms, event_time_ms=event_ms,
            open=open_, high=high, low=low, close=close, volume=volume,
        )

    def accept_sequence(self, candle: ClosedKline) -> bool:
        """Reject out-of-order/gapped bars; return False for an exact duplicate."""
        if candle.symbol != self.symbol or candle.timeframe != self.timeframe:
            raise ValueError("candle identity does not match this feed")
        previous = self.last_open_time_ms
        if previous is None:
            self.last_open_time_ms = candle.open_time_ms
            self.last_candle = candle
            return True
        if candle.open_time_ms == previous:
            prior = self.last_candle
            same_values = prior is not None and (
                candle.close_time_ms == prior.close_time_ms
                and (candle.open, candle.high, candle.low, candle.close, candle.volume)
                    == (prior.open, prior.high, prior.low, prior.close, prior.volume)
            )
            if not same_values:
                raise ValueError("conflicting duplicate closed candle")
            return False
        if candle.open_time_ms < previous:
            raise ValueError("out-of-order closed candle")
        if candle.open_time_ms != previous + self.interval_ms:
            raise ValueError("closed-candle gap detected; resynchronization required")
        self.last_open_time_ms = candle.open_time_ms
        self.last_candle = candle
        return True

    def connect(self, *, timeout_seconds: float = 10.0):
        """Open a public read-only socket. Caller owns close/reconnect policy."""
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        try:
            import websocket
        except ImportError as exc:
            raise RuntimeError("Install websocket-client to use the live public feed") from exc
        return websocket.create_connection(self.url, timeout=float(timeout_seconds),
                                           enable_multithread=False)

    def receive_closed_candle(self, socket, *, clock_ms: Callable[[], int] | None = None) -> ClosedKline:
        """Read one completed candle using a fresh clock value for every message."""
        clock = clock_ms or (lambda: time.time_ns() // 1_000_000)
        while True:
            message = socket.recv()
            candle = self.parse(message, now_ms=clock())
            if candle is None:
                continue
            if self.accept_sequence(candle):
                return candle
