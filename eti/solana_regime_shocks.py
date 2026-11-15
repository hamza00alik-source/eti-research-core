"""Descriptive market-wide regime co-movement diagnostics.

This module detects timestamp buckets with clustered regime transitions.
It does not infer causality, independent observations, executable signals or P&L.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


def _aware_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("baseline_at must be a non-empty timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("baseline_at must be a valid timezone-aware ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("baseline_at must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def evaluate_solana_market_wide_regime_shocks(
    classified_observations: list[dict[str, Any]],
    *,
    bucket_seconds: int = 300,
    maximum_gap_seconds: int = 900,
    minimum_mints: int = 3,
    minimum_transitions: int = 3,
) -> dict[str, Any]:
    """Find time buckets containing unusually broad observed regime changes.

    Transitions are only formed between adjacent, unambiguous observations for
    the same mint, pool and horizon. A conflicting duplicate timestamp breaks
    continuity; long gaps also prevent a transition from being inferred.
    """
    for name, value in (
        ("bucket_seconds", bucket_seconds),
        ("maximum_gap_seconds", maximum_gap_seconds),
        ("minimum_mints", minimum_mints),
        ("minimum_transitions", minimum_transitions),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(classified_observations, list):
        raise ValueError("classified_observations must be a list")

    grouped: dict[tuple[str, str, str], list[tuple[datetime, str]]] = defaultdict(list)
    for row in classified_observations:
        if not isinstance(row, dict):
            continue
        mint, pool, horizon, regime = (
            row.get("mint"), row.get("pool_address"),
            row.get("horizon_seconds"), row.get("regime"),
        )
        if not all(isinstance(x, str) and x.strip() for x in (mint, pool, regime)):
            continue
        if horizon is None or str(horizon).strip() == "":
            continue
        try:
            stamp = _aware_timestamp(row.get("baseline_at"))
        except ValueError:
            continue
        grouped[(mint.strip(), pool.strip(), str(horizon).strip())].append((stamp, regime.strip()))

    bucket_events: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"transition_count": 0, "mints": set(), "transition_types": defaultdict(int)}
    )
    observed_bucket_mints: dict[int, set[str]] = defaultdict(set)
    excluded_conflict_timestamp_count = 0
    eligible_group_count = 0

    for (mint, _pool, _horizon), observations in grouped.items():
        by_time: dict[datetime, set[str]] = defaultdict(set)
        for stamp, regime in observations:
            by_time[stamp].add(regime)
        ordered = sorted(by_time.items())
        eligible_group_count += 1
        for stamp, regimes in ordered:
            bucket = int(stamp.timestamp()) // bucket_seconds * bucket_seconds
            observed_bucket_mints[bucket].add(mint)
            if len(regimes) > 1:
                excluded_conflict_timestamp_count += 1
        for index in range(1, len(ordered)):
            previous_stamp, previous_regimes = ordered[index - 1]
            current_stamp, current_regimes = ordered[index]
            if len(previous_regimes) != 1 or len(current_regimes) != 1:
                continue
            elapsed = (current_stamp - previous_stamp).total_seconds()
            if elapsed <= 0 or elapsed > maximum_gap_seconds:
                continue
            previous_regime = next(iter(previous_regimes))
            current_regime = next(iter(current_regimes))
            if previous_regime == current_regime:
                continue
            bucket = int(current_stamp.timestamp()) // bucket_seconds * bucket_seconds
            event = bucket_events[bucket]
            event["transition_count"] += 1
            event["mints"].add(mint)
            event["transition_types"][f"{previous_regime}->{current_regime}"] += 1

    buckets = []
    for bucket in sorted(set(observed_bucket_mints) | set(bucket_events)):
        event = bucket_events[bucket]
        observed_mints = observed_bucket_mints[bucket]
        changed_mints = event["mints"]
        transition_count = event["transition_count"]
        buckets.append({
            "bucket_start": datetime.fromtimestamp(bucket, tz=timezone.utc).isoformat(),
            "observed_mint_count": len(observed_mints),
            "transition_mint_count": len(changed_mints),
            "transition_count": transition_count,
            "transition_mint_share": len(changed_mints) / len(observed_mints) if observed_mints else None,
            "transition_types": dict(sorted(event["transition_types"].items())),
            "meets_shock_threshold": (
                len(changed_mints) >= minimum_mints and transition_count >= minimum_transitions
            ),
        })
    shock_buckets = [item for item in buckets if item["meets_shock_threshold"]]
    return {
        "status": "DESCRIPTIVE_MARKET_WIDE_REGIME_COOCCURRENCE",
        "bucket_seconds": bucket_seconds,
        "maximum_gap_seconds": maximum_gap_seconds,
        "minimum_mints": minimum_mints,
        "minimum_transitions": minimum_transitions,
        "eligible_mint_pool_horizon_group_count": eligible_group_count,
        "bucket_count": len(buckets),
        "shock_bucket_count": len(shock_buckets),
        "excluded_conflicting_timestamp_count": excluded_conflict_timestamp_count,
        "buckets": buckets,
        "shock_buckets": shock_buckets,
        "model_fitting_performed": False,
        "training_authorized": False,
        "causality_established": False,
        "independent_mint_replications_established": False,
        "realized_pnl_available": False,
        "note": (
            "Co-occurring observed transitions are descriptive only. Mint recurrence, "
            "shared pools, overlapping labels, common market shocks and timestamp "
            "aggregation can create dependence. Buckets do not establish causality, "
            "independent replications, predictive value, executable returns or P&L."
        ),
    }
