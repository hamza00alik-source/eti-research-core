"""Validation diagnostics for exported Solana research observations."""
from __future__ import annotations

from datetime import datetime
import math
from typing import Any


NUMERIC_FIELDS = (
    "price_usd", "liquidity_usd", "volume_h24_usd",
    "txns_h1_buys", "txns_h1_sells",
)


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _is_finite_number(value: int | float) -> bool:\n    try:\n        return math.isfinite(float(value))\n    except (OverflowError, TypeError, ValueError):\n        return False\n\n\ndef validate_snapshot_rows(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Report data-quality issues without coercing, dropping, or repairing source values."""
    issue_counts = {
        "malformed_payload": 0, "missing_evidence": 0, "missing_feature_values": 0,
        "non_finite_feature_values": 0, "negative_feature_values": 0,
        "zero_price_or_liquidity": 0, "invalid_timestamp": 0,
        "duplicate_entity_timestamp": 0, "non_increasing_entity_time": 0,
    }
    seen: set[tuple[str, str, str]] = set()
    previous: dict[tuple[str, str], datetime] = {}
    invalid_rows: list[dict[str, Any]] = []
    for row in snapshots:
        problems: list[str] = []
        payload = row.get("payload")
        if not isinstance(payload, dict):
            issue_counts["malformed_payload"] += 1
            problems.append("malformed_payload")
            payload = {}
        evidence = payload.get("evidence")
        if not isinstance(evidence, dict):
            issue_counts["missing_evidence"] += 1
            problems.append("missing_evidence")
            evidence = {}
        for field in NUMERIC_FIELDS:
            value = evidence.get(field)
            if value is None:
                issue_counts["missing_feature_values"] += 1
                problems.append("missing:" + field)
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not _is_finite_number(value):
                issue_counts["non_finite_feature_values"] += 1
                problems.append("invalid_numeric:" + field)
                continue
            if value < 0:
                issue_counts["negative_feature_values"] += 1
                problems.append("negative:" + field)
            if value == 0 and field in {"price_usd", "liquidity_usd"}:
                issue_counts["zero_price_or_liquidity"] += 1
                problems.append("zero:" + field)
        stamp = _timestamp(row.get("fetched_at"))
        if stamp is None:
            issue_counts["invalid_timestamp"] += 1
            problems.append("invalid_timestamp")
        else:
            entity = (str(row.get("mint", "")), str(row.get("pool_address", "")))
            identity = (*entity, stamp.isoformat())
            if identity in seen:
                issue_counts["duplicate_entity_timestamp"] += 1
                problems.append("duplicate_entity_timestamp")
            seen.add(identity)
            prior = previous.get(entity)
            if prior is not None and stamp <= prior:
                issue_counts["non_increasing_entity_time"] += 1
                problems.append("non_increasing_entity_time")
            if prior is None or stamp > prior:
                previous[entity] = stamp
        if problems:
            invalid_rows.append({"snapshot_id": row.get("id"), "issues": problems})
    return {
        "status": "DIAGNOSTICS_ONLY",
        "rows_checked": len(snapshots),
        "issue_counts": issue_counts,
        "rows_with_issues": len(invalid_rows),
        "row_issues": invalid_rows[:500],
        "row_issue_list_truncated": len(invalid_rows) > 500,
        "note": "No values are repaired or imputed; zero liquidity/price and negative values are flagged for review.",
    }
