"""Allowlisted ETI agent capabilities.

Handlers are explicit server-side functions. User task payloads are data only and
are never interpreted as Python, shell commands, prompts or executable code.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .features import build_features
from .providers import CcxtPublicProvider


AgentHandler = Callable[[dict[str, Any]], dict[str, Any]]


class AgentCapabilityRegistry:
    """In-process allowlist of named, server-implemented capabilities."""

    def __init__(self):
        self._handlers: dict[str, AgentHandler] = {}

    def register(self, name: str, handler: AgentHandler) -> None:
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ValueError("capability name must contain 1-120 characters")
        if not callable(handler):
            raise TypeError("capability handler must be callable")
        if name in self._handlers:
            raise ValueError(f"capability already registered: {name}")
        self._handlers[name] = handler

    def execute(self, name: str, payload: Any) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            raise LookupError(f"unsupported agent capability: {name}")
        if not isinstance(payload, dict):
            raise ValueError("agent task input must be a JSON object")
        result = handler(payload)
        if not isinstance(result, dict):
            raise TypeError("agent capability must return a JSON object")
        # Reuse strict JSON encoding rules to reject NaN, infinity and oversized results.
        from .storage import _encode_json
        _encode_json(result)
        return result

    def capabilities(self) -> list[str]:
        return sorted(self._handlers)


def market_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Fetch completed public crypto spot candles and summarize causal indicators."""
    venue = payload.get("venue", "binance")
    symbol = payload.get("symbol")
    timeframe = payload.get("timeframe", "1h")
    limit = payload.get("limit", 200)
    if not isinstance(venue, str) or venue not in {"binance", "mexc"}:
        raise ValueError("venue must be binance or mexc")
    if not isinstance(symbol, str) or not 3 <= len(symbol) <= 40 or "/" not in symbol:
        raise ValueError("symbol must be a valid spot pair such as BTC/USDT")
    if not isinstance(timeframe, str) or timeframe not in {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}:
        raise ValueError("unsupported timeframe")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 50 <= limit <= 500:
        raise ValueError("limit must be an integer from 50 to 500")

    frame = CcxtPublicProvider(venue).fetch_ohlcv(symbol, timeframe, limit)
    if len(frame) < 40:
        raise ValueError("provider returned too few completed candles for indicator calculation")
    features = build_features(frame)
    latest = features.iloc[-1]
    indicator_names = (
        "ret_1", "ret_3", "ret_6", "volatility_20",
        "sma_gap_10", "sma_gap_30", "rsi_14", "range_pct", "atr_pct",
    )
    indicators: dict[str, float] = {}
    for name in indicator_names:
        value = float(latest[name])
        if not np.isfinite(value):
            raise ValueError(f"latest indicator is unavailable: {name}")
        indicators[name] = round(value, 8)
    short_gap, long_gap = indicators["sma_gap_10"], indicators["sma_gap_30"]
    trend = "bullish" if short_gap > 0 and long_gap > 0 else (
        "bearish" if short_gap < 0 and long_gap < 0 else "mixed"
    )
    return {
        "kind": "public_market_snapshot",
        "venue": venue,
        "asset_class": "crypto",
        "market_type": "spot",
        "symbol": symbol,
        "timeframe": timeframe,
        "completed_candles": len(frame),
        "latest_completed_candle": frame.iloc[-1]["timestamp"].isoformat(),
        "last_close": round(float(frame.iloc[-1]["close"]), 10),
        "trend_context": trend,
        "indicators": indicators,
        "limitations": [
            "Descriptive market research only; not a trade signal or recommendation.",
            "Public spot OHLCV only; no private account data or order execution.",
            "No model training is performed by this capability.",
        ],
    }


def default_capabilities() -> AgentCapabilityRegistry:
    registry = AgentCapabilityRegistry()
    registry.register("market.snapshot", market_snapshot)
    return registry
