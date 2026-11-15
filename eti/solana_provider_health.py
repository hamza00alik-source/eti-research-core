"""Bounded, evidence-only diagnostics for observed Solana provider requests.

This module reports the supplied observations; it does not probe providers,
estimate uptime, or infer health from missing records.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def evaluate_solana_provider_health(
    observations: Iterable[dict[str, Any]],
    *,
    now: str,
    stale_after_seconds: int = 300,
    degraded_latency_ms: float = 3000.0,
) -> dict[str, Any]:
    """Summarize explicit request outcomes by provider without inventing probes."""
    current = _timestamp(now)
    if current is None:
        raise ValueError("now must be a timezone-aware ISO timestamp")
    if isinstance(stale_after_seconds, bool) or not isinstance(stale_after_seconds, int) or stale_after_seconds < 1:
        raise ValueError("stale_after_seconds must be a positive integer")
    if isinstance(degraded_latency_ms, bool) or not isinstance(degraded_latency_ms, (int, float)) or not math.isfinite(degraded_latency_ms) or degraded_latency_ms <= 0:
        raise ValueError("degraded_latency_ms must be finite and positive")

    grouped: dict[str, list[dict[str, Any]]] = {}
    invalid_rows = 0
    for row in observations:
        if not isinstance(row, dict):
            invalid_rows += 1
            continue
        provider = row.get("provider")
        observed_at = _timestamp(row.get("observed_at"))
        status = row.get("status")
        latency = row.get("latency_ms")
        if not isinstance(provider, str) or not provider.strip() or observed_at is None or status not in {"success", "error"}:
            invalid_rows += 1
            continue
        if latency is not None and (
            isinstance(latency, bool) or not isinstance(latency, (int, float))
            or not math.isfinite(latency) or latency < 0
        ):
            invalid_rows += 1
            continue
        grouped.setdefault(provider.strip(), []).append({
            "observed_at": observed_at, "status": status,
            "latency_ms": float(latency) if latency is not None else None,
            "error_code": row.get("error_code") if isinstance(row.get("error_code"), str) else None,
        })

    reports = []
    for provider, rows in sorted(grouped.items()):
        latest = max(rows, key=lambda item: item["observed_at"])
        age = (current - latest["observed_at"]).total_seconds()
        if age < 0:
            status = "UNKNOWN"
            reason = "latest_observation_is_in_the_future"
        elif age > stale_after_seconds:
            status = "UNKNOWN"
            reason = "latest_observation_is_stale"
        elif latest["status"] == "error":
            status = "DEGRADED"
            reason = "latest_observed_request_failed"
        elif latest["latency_ms"] is None:
            status = "UNKNOWN"
            reason = "latest_success_has_no_latency_measurement"
        elif latest["latency_ms"] > degraded_latency_ms:
            status = "DEGRADED"
            reason = "latest_observed_latency_exceeds_threshold"
        else:
            status = "HEALTHY"
            reason = "latest_observed_request_succeeded_within_threshold"

        reports.append({
            "provider": provider,
            "status": status,
            "reason": reason,
            "latest_observed_at": latest["observed_at"].isoformat().replace("+00:00", "Z"),
            "latest_observation_age_seconds": round(age, 3),
            "latest_request_status": latest["status"],
            "latest_latency_ms": latest["latency_ms"],
            "latest_error_code": latest["error_code"],
            "supplied_observation_count": len(rows),
            "supplied_success_count": sum(item["status"] == "success" for item in rows),
            "supplied_error_count": sum(item["status"] == "error" for item in rows),
        })

    return {
        "status": "OBSERVATION_SUMMARY_ONLY",
        "providers": reports,
        "provider_count": len(reports),
        "invalid_observation_count": invalid_rows,
        "stale_after_seconds": stale_after_seconds,
        "degraded_latency_ms": float(degraded_latency_ms),
        "uptime_estimated": False,
        "provider_health_proven": False,
        "missing_observation_means_failure": False,
        "trading_authorized": False,
    }
