"""Transparent, deterministic research ranking for completed spot-market snapshots.

This is a screening heuristic, not a predictive model or trading signal.
"""
from __future__ import annotations

import math
from typing import Any


_REQUIRED = ("ret_3", "ret_6", "volatility_20", "sma_gap_10", "sma_gap_30", "rsi_14")


def rank_spot_opportunity(indicators: dict[str, Any]) -> dict[str, Any]:
    """Return an explainable 0-100 long-bias screening score for spot markets."""
    if not isinstance(indicators, dict):
        raise ValueError("indicators must be an object")
    values: dict[str, float] = {}
    for key in _REQUIRED:
        try:
            value = float(indicators[key])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"missing or invalid indicator: {key}") from None
        if not math.isfinite(value):
            raise ValueError(f"indicator must be finite: {key}")
        values[key] = value

    score = 50
    reasons: list[str] = []
    bullish = values["sma_gap_10"] > 0 and values["sma_gap_30"] > 0
    bearish = values["sma_gap_10"] < 0 and values["sma_gap_30"] < 0

    if bullish:
        score += 20
        reasons.append("Price is above both short- and medium-term moving averages")
    elif bearish:
        score -= 20
        reasons.append("Price is below both short- and medium-term moving averages")
    else:
        reasons.append("Moving-average trend is mixed")

    if values["ret_3"] > 0 and values["ret_6"] > 0:
        score += 10
        reasons.append("Recent 3- and 6-period momentum is positive")
    elif values["ret_3"] < 0 and values["ret_6"] < 0:
        score -= 10
        reasons.append("Recent 3- and 6-period momentum is negative")
    else:
        reasons.append("Recent momentum is mixed")

    rsi = values["rsi_14"]
    if 45 <= rsi <= 65:
        score += 10
        reasons.append("RSI is in a moderate positive-momentum range")
    elif 35 <= rsi <= 70:
        score += 5
        reasons.append("RSI is within a broad, non-extreme range")
    else:
        score -= 10
        reasons.append("RSI is outside the preferred screening range")

    volatility = values["volatility_20"]
    if volatility > 0.08:
        score -= 15
        reasons.append("Recent volatility is unusually high")
    elif 0.002 <= volatility <= 0.05:
        score += 5
        reasons.append("Recent volatility is within the screening band")
    else:
        reasons.append("Volatility is outside the preferred screening band")

    score = max(0, min(100, score))
    if score >= 70 and bullish:
        classification = "long_watch"
    elif score <= 35 or bearish:
        classification = "avoid"
    else:
        classification = "monitor"

    return {
        "score": score,
        "classification": classification,
        "method": "transparent_heuristic_v1",
        "is_predictive_model": False,
        "reasons": reasons,
        "disclaimer": "Research ranking only. Not a forecast, trade signal, or instruction to buy.",
    }
