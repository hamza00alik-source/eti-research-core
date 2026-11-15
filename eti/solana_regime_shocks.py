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

    bucket_events: dict[tuple[int, str], dict[str, Any]] = defaultdict(
        lambda: {"transition_count": 0, "mints": set(), "transition_types": defaultdict(int)}
    )
    observed_bucket_mints: dict[tuple[int, str], set[str]] = defaultdict(set)
    excluded_conflict_timestamp_count = 0
    eligible_group_count = 0

    for (mint, _pool, horizon), observations in grouped.items():
        by_time: dict[datetime, set[str]] = defaultdict(set)
        for stamp, regime in observations:
            by_time[stamp].add(regime)
        ordered = sorted(by_time.items())
        eligible_group_count += 1
        for stamp, regimes in ordered:
            bucket = (int(stamp.timestamp()) // bucket_seconds * bucket_seconds, horizon)
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
            bucket = (int(current_stamp.timestamp()) // bucket_seconds * bucket_seconds, horizon)
            event = bucket_events[bucket]
            event["transition_count"] += 1
            event["mints"].add(mint)
            event["transition_types"][f"{previous_regime}->{current_regime}"] += 1

    buckets = []
    for bucket, horizon in sorted(set(observed_bucket_mints) | set(bucket_events)):
        event = bucket_events[(bucket, horizon)]
        observed_mints = observed_bucket_mints[(bucket, horizon)]
        changed_mints = event["mints"]
        transition_count = event["transition_count"]
        buckets.append({
            "horizon_seconds": horizon,
            "bucket_start": datetime.fromtimestamp(bucket, tz=timezone.utc).isoformat(),
            "observed_mint_count": len(observed_mints),
            "transition_mint_count": len(changed_mints),
            "transition_mints": sorted(changed_mints),
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



def evaluate_solana_shock_recurrence(
    shock_report: dict[str, Any],
    *,
    minimum_event_separation_seconds: int = 3600,
    minimum_distinct_event_periods: int = 2,
) -> dict[str, Any]:
    """Describe recurrence and mint concentration among prequalified shock buckets.

    Adjacent qualifying buckets within the separation window are one event period.
    Mint counts are descriptive recurrence, not independent replications.
    """
    for name, value in (
        ("minimum_event_separation_seconds", minimum_event_separation_seconds),
        ("minimum_distinct_event_periods", minimum_distinct_event_periods),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(shock_report, dict) or not isinstance(shock_report.get("shock_buckets"), list):
        raise ValueError("shock_report must contain shock_buckets")

    by_horizon: dict[str, list[tuple[datetime, set[str]]]] = defaultdict(list)
    for row in shock_report["shock_buckets"]:
        if not isinstance(row, dict):
            continue
        horizon = row.get("horizon_seconds")
        mints = row.get("transition_mints")
        if not isinstance(horizon, str) or not horizon.strip() or not isinstance(mints, list):
            continue
        try:
            stamp = _aware_timestamp(row.get("bucket_start"))
        except ValueError:
            continue
        clean_mints = {mint.strip() for mint in mints if isinstance(mint, str) and mint.strip()}
        by_horizon[horizon].append((stamp, clean_mints))

    summaries = {}
    for horizon, events in sorted(by_horizon.items()):
        events.sort(key=lambda item: item[0])
        periods: list[dict[str, Any]] = []
        for stamp, mints in events:
            if not periods or (stamp - periods[-1]["last_timestamp"]).total_seconds() > minimum_event_separation_seconds:
                periods.append({"first_timestamp": stamp, "last_timestamp": stamp, "mints": set(), "bucket_count": 0})
            period = periods[-1]
            period["last_timestamp"] = stamp
            period["mints"].update(mints)
            period["bucket_count"] += 1

        mint_period_counts: dict[str, int] = defaultdict(int)
        for period in periods:
            for mint in period["mints"]:
                mint_period_counts[mint] += 1
        repeated = {mint: count for mint, count in mint_period_counts.items() if count > 1}
        total_mint_periods = sum(mint_period_counts.values())
        top_mint_periods = max(mint_period_counts.values(), default=0)
        summaries[horizon] = {
            "qualifying_bucket_count": len(events),
            "distinct_event_period_count": len(periods),
            "event_periods": [{
                "first_timestamp": period["first_timestamp"].isoformat(),
                "last_timestamp": period["last_timestamp"].isoformat(),
                "qualifying_bucket_count": period["bucket_count"],
                "distinct_mint_count": len(period["mints"]),
            } for period in periods],
            "distinct_mint_count": len(mint_period_counts),
            "mints_reappearing_in_multiple_event_periods": dict(sorted(repeated.items())),
            "mint_period_observation_count": total_mint_periods,
            "largest_mint_period_share": top_mint_periods / total_mint_periods if total_mint_periods else None,
            "meets_recurrence_threshold": len(periods) >= minimum_distinct_event_periods,
        }

    return {
        "status": "DESCRIPTIVE_SHOCK_RECURRENCE_AND_MINT_CONCENTRATION",
        "minimum_event_separation_seconds": minimum_event_separation_seconds,
        "minimum_distinct_event_periods": minimum_distinct_event_periods,
        "by_horizon_seconds": summaries,
        "causality_established": False,
        "independent_mint_replications_established": False,
        "predictive_value_established": False,
        "training_authorized": False,
        "realized_pnl_available": False,
        "note": "Event periods are a configurable grouping of observed qualifying buckets, not statistically independent shocks. Repeated mint appearances indicate concentration, not independent replication or predictive value.",
    }



def evaluate_solana_shock_period_stability(
    shock_report: dict[str, Any],
    *,
    period_seconds: int = 604800,
    minimum_buckets_per_period: int = 5,
    minimum_observed_mints_per_period: int = 3,
) -> dict[str, Any]:
    """Compare descriptive shock rates and mint concentration across fixed UTC periods."""
    for name, value in (
        ("period_seconds", period_seconds),
        ("minimum_buckets_per_period", minimum_buckets_per_period),
        ("minimum_observed_mints_per_period", minimum_observed_mints_per_period),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(shock_report, dict) or not isinstance(shock_report.get("buckets"), list):
        raise ValueError("shock_report must contain buckets")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in shock_report["buckets"]:
        if not isinstance(row, dict):
            continue
        horizon = row.get("horizon_seconds")
        if not isinstance(horizon, str) or not horizon.strip():
            continue
        try:
            stamp = _aware_timestamp(row.get("bucket_start"))
        except ValueError:
            continue
        period_start = int(stamp.timestamp()) // period_seconds * period_seconds
        grouped[(horizon, period_start)].append(row)

    by_horizon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (horizon, period_start), rows in sorted(grouped.items()):
        observed_mints: set[str] = set()
        transition_mints: set[str] = set()
        observed_bucket_count = len(rows)
        shock_bucket_count = 0
        transition_count = 0
        for row in rows:
            observed = row.get("observed_mint_count")
            if isinstance(observed, int) and not isinstance(observed, bool) and observed >= 0:
                # Only aggregate reported counts; identities are unavailable for all observed mints.
                pass
            if row.get("meets_shock_threshold") is True:
                shock_bucket_count += 1
                mints = row.get("transition_mints", [])
                if isinstance(mints, list):
                    transition_mints.update(
                        mint.strip() for mint in mints
                        if isinstance(mint, str) and mint.strip()
                    )
            count = row.get("transition_count")
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                transition_count += count
        # Shock buckets alone do not expose all observed mint identities. Use the
        # per-bucket observed counts for a conservative minimum coverage proxy.
        observed_mint_count_sum = sum(
            row.get("observed_mint_count", 0)
            for row in rows
            if isinstance(row.get("observed_mint_count"), int)
            and not isinstance(row.get("observed_mint_count"), bool)
            and row.get("observed_mint_count", 0) >= 0
        )
        distinct_observed_mints = max(
            observed_mints.__len__(), len(transition_mints)
        )
        eligible = (
            observed_bucket_count >= minimum_buckets_per_period
            and distinct_observed_mints >= minimum_observed_mints_per_period
        )
        by_horizon[horizon].append({
            "period_start": datetime.fromtimestamp(period_start, tz=timezone.utc).isoformat(),
            "period_end_exclusive": datetime.fromtimestamp(period_start + period_seconds, tz=timezone.utc).isoformat(),
            "observed_bucket_count": observed_bucket_count,
            "shock_bucket_count": shock_bucket_count,
            "shock_bucket_rate": shock_bucket_count / observed_bucket_count if observed_bucket_count else None,
            "transition_count": transition_count,
            "distinct_transition_mint_count": len(transition_mints),
            "reported_observed_mint_bucket_count_sum": observed_mint_count_sum,
            "minimum_coverage_proxy_mint_count": distinct_observed_mints,
            "eligible_for_descriptive_comparison": eligible,
            "insufficient_coverage": not eligible,
        })

    return {
        "status": "DESCRIPTIVE_CHRONOLOGICAL_SHOCK_PERIOD_STABILITY",
        "period_seconds": period_seconds,
        "minimum_buckets_per_period": minimum_buckets_per_period,
        "minimum_observed_mints_per_period": minimum_observed_mints_per_period,
        "by_horizon_seconds": dict(by_horizon),
        "predictive_value_established": False,
        "causality_established": False,
        "independent_periods_established": False,
        "training_authorized": False,
        "realized_pnl_available": False,
        "note": "Fixed UTC periods are descriptive partitions, not independent samples. The shock input contains observed mint counts but not identities for all observed mints, so coverage eligibility uses only the distinct transition-mint count as a lower bound; it cannot establish full mint-population coverage or absence of concentration.",
    }
