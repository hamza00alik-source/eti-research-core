"""Quality checks for exchange funding and open-interest observations."""
from __future__ import annotations

import math
from statistics import median
from typing import Any


MIN_TIMESTAMP_MS = 946684800000       # 2000-01-01; reject seconds/malformed epochs
MAX_TIMESTAMP_MS = 4102444800000     # 2100-01-01; reject implausible future epochs


def assess_observation_history(
    rows: Any, *, value_key: str, allow_negative: bool
) -> tuple[list[dict], dict]:
    """Return sorted, de-duplicated valid observations and transparent diagnostics.

    This is a research-data filter, not an exchange-calendar validator. Raw order,
    duplicate timestamps, invalid values and malformed epochs remain visible in
    the diagnostics instead of being silently hidden by pandas drop_duplicates.
    """
    if not isinstance(rows, list):
        rows = []
    accepted: list[dict] = []
    invalid_timestamp = 0
    invalid_value = 0
    duplicate_timestamp = 0
    out_of_order = 0
    seen: set[int] = set()
    previous: int | None = None

    for raw in rows:
        if not isinstance(raw, dict):
            invalid_timestamp += 1
            continue
        timestamp = raw.get("timestamp")
        value = raw.get(value_key)
        try:
            numeric_timestamp = float(timestamp)
            if not math.isfinite(numeric_timestamp) or not numeric_timestamp.is_integer():
                raise ValueError
            ts = int(numeric_timestamp)
            if not MIN_TIMESTAMP_MS <= ts <= MAX_TIMESTAMP_MS:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            invalid_timestamp += 1
            continue

        if previous is not None and ts < previous:
            out_of_order += 1
        previous = ts
        if ts in seen:
            duplicate_timestamp += 1
            continue
        try:
            numeric_value = float(value)
            if not math.isfinite(numeric_value) or (not allow_negative and numeric_value < 0):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            invalid_value += 1
            continue
        seen.add(ts)
        accepted.append({**raw, "timestamp": ts, value_key: numeric_value})

    accepted.sort(key=lambda item: item["timestamp"])
    intervals = [
        accepted[index]["timestamp"] - accepted[index - 1]["timestamp"]
        for index in range(1, len(accepted))
    ]
    median_interval = float(median(intervals)) if intervals else None
    interval_cv = None
    if len(intervals) >= 2 and median_interval and median_interval > 0:
        mean_interval = sum(intervals) / len(intervals)
        variance = sum((item - mean_interval) ** 2 for item in intervals) / len(intervals)
        interval_cv = math.sqrt(variance) / mean_interval

    return accepted, {
        "raw_observations": len(rows),
        "valid_observations": len(accepted),
        "invalid_timestamp_count": invalid_timestamp,
        "invalid_value_count": invalid_value,
        "duplicate_timestamp_count": duplicate_timestamp,
        "out_of_order_count": out_of_order,
        "median_observation_interval_ms": median_interval,
        "observation_interval_cv": interval_cv,
        "interval_regularity": (
            "insufficient_samples" if len(intervals) < 2 else
            "regular" if interval_cv is not None and interval_cv <= 0.25 else
            "irregular_or_gapped"
        ),
        "note": "Intervals describe returned observations, not a guaranteed exchange settlement schedule.",
    }
