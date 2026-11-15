"""Event-timed funding accounting; never infer settlement cadence from candles."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd


def funding_cost_series(position: pd.Series, events: list[dict[str, Any]] | None) -> pd.Series:
    """Return per-bar funding costs for events exactly aligned to UTC bar timestamps.

    A positive rate means longs pay shorts; a negative rate means shorts pay longs.
    Positions are already shifted by the caller to represent exposure held on each bar.
    """
    costs = pd.Series(0.0, index=position.index, dtype=float)
    if events is None:
        return costs
    if not isinstance(events, list):
        raise ValueError("funding_events must be a list")
    if not events:
        return costs
    index = position.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("event-timed funding requires a timezone-aware DatetimeIndex")
    utc_index = index.tz_convert("UTC")
    if not utc_index.is_unique or not utc_index.is_monotonic_increasing:
        raise ValueError("funding bar timestamps must be unique and chronological")
    lookup = {timestamp: i for i, timestamp in enumerate(utc_index)}
    seen = set()
    values = costs.to_numpy(copy=True)
    positions = position.to_numpy(dtype=float)
    if not all(math.isfinite(value) for value in positions):
        raise ValueError("funding positions must be finite")
    for i, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"funding_events[{i}] must be an object")
        raw_timestamp = event.get("timestamp")
        try:
            timestamp = pd.Timestamp(raw_timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"funding_events[{i}].timestamp must be ISO-8601") from exc
        if timestamp.tzinfo is None:
            raise ValueError(f"funding_events[{i}].timestamp must include a timezone")
        timestamp = timestamp.tz_convert("UTC")
        rate = event.get("rate_bps")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate):
            raise ValueError(f"funding_events[{i}].rate_bps must be finite and signed")
        if timestamp in seen:
            raise ValueError("funding_events contains duplicate timestamps")
        seen.add(timestamp)
        if timestamp not in lookup:
            raise ValueError(f"funding settlement {timestamp.isoformat()} does not align to an evaluation bar")
        row = lookup[timestamp]
        values[row] += positions[row] * float(rate) / 10000.0
    return pd.Series(values, index=position.index, dtype=float)
