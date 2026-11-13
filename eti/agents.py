"""Allowlisted ETI agent capabilities.

Handlers are explicit server-side functions. User task payloads are data only and
are never interpreted as Python, shell commands, prompts or executable code.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .features import build_features
from .providers import CcxtPublicProvider
from .opportunities import rank_spot_opportunity
from .strategy_evaluation import run_strategy_walk_forward


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
    venue = payload.get("venue", "mexc")
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
        "research_ranking": rank_spot_opportunity(indicators),
        "limitations": [
            "Descriptive market research and a transparent heuristic ranking only; not a forecast or trade signal.",
            "Public spot OHLCV only; no private account data or order execution.",
            "The ranking is not a trained AI model; model training is performed separately.",
        ],
    }


def strategy_walk_forward(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate fixed rule candidates on public candles; report is stored as task result."""
    venue = payload.get("venue", "mexc")
    symbol = payload.get("symbol")
    timeframe = payload.get("timeframe", "1h")
    limit = payload.get("limit", 500)
    if not isinstance(venue, str) or venue not in {"binance", "mexc"}:
        raise ValueError("venue must be binance or mexc")
    if not isinstance(symbol, str) or not 3 <= len(symbol) <= 40 or "/" not in symbol:
        raise ValueError("symbol must be a valid spot pair such as BTC/USDT")
    if not isinstance(timeframe, str) or timeframe not in {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}:
        raise ValueError("unsupported timeframe")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 300 <= limit <= 500:
        raise ValueError("strategy walk-forward limit must be from 300 to 500")
    options = {
        "min_history": payload.get("min_history", 120),
        "validation_size": payload.get("validation_size", 80),
        "test_size": payload.get("test_size", 80),
        "step_size": payload.get("step_size", 80),
        "purge_bars": payload.get("purge_bars", 2),
    }
    for name, value in options.items():
        minimum = 0 if name == "purge_bars" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} has an invalid value")
    fee_bps, slippage_bps = payload.get("fee_bps", 6.0), payload.get("slippage_bps", 2.0)
    frame = CcxtPublicProvider(venue).fetch_ohlcv(symbol, timeframe, limit)
    if len(frame) < options["min_history"] + options["validation_size"] + options["test_size"] + 2 * options["purge_bars"]:
        raise ValueError("provider returned insufficient completed candles for walk-forward evaluation")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any() or timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("provider timestamps must be valid, unique, and chronological")
    close = pd.Series(pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float),
                      index=pd.DatetimeIndex(timestamps), name="close")
    report = run_strategy_walk_forward(close, **options, fee_bps=fee_bps, slippage_bps=slippage_bps)
    return {
        "kind": "strategy_walk_forward_report", "venue": venue, "asset_class": "crypto",
        "market_type": "spot", "symbol": symbol, "timeframe": timeframe,
        "requested_candles": limit, "available_candles": len(frame),
        "data_start": timestamps.iloc[0].isoformat(), "data_end": timestamps.iloc[-1].isoformat(),
        **report,
        "limitations": [
            "Fixed rule-based moving-average and momentum candidates; not a trained AI model.",
            "Validation data selects candidates; test windows are held out from candidate selection.",
            "Historical results do not establish future profitability and are not trade signals.",
            "Public spot data only; no credentials, balances, orders, or live execution.",
        ],
    }


def default_capabilities() -> AgentCapabilityRegistry:
    registry = AgentCapabilityRegistry()
    registry.register("market.snapshot", market_snapshot)
    registry.register("strategy.walk_forward", strategy_walk_forward)
    return registry
