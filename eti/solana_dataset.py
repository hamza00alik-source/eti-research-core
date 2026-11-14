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


def _is_finite_number(value: int | float) -> bool:
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def validate_snapshot_rows(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Report data-quality issues without coercing, dropping, or repairing source values."""
    issue_counts = {
        "malformed_payload": 0, "unsupported_schema_version": 0,
        "missing_evidence": 0, "missing_feature_values": 0,
        "non_finite_feature_values": 0, "negative_feature_values": 0,
        "zero_price_or_liquidity": 0, "invalid_timestamp": 0,
        "duplicate_entity_timestamp": 0, "non_increasing_entity_time": 0,
    }
    seen: set[tuple[str, str, str]] = set()
    previous: dict[tuple[str, str], datetime] = {}
    invalid_rows: list[dict[str, Any]] = []
    for row in snapshots:
        problems: list[str] = []
        schema_version = row.get("schema_version")
        if isinstance(schema_version, bool) or schema_version != 1:
            issue_counts["unsupported_schema_version"] += 1
            problems.append("unsupported_schema_version")
        payload = row.get("payload")
        if not isinstance(payload, dict):
            issue_counts["malformed_payload"] += 1
            problems.append("malformed_payload")
            payload = {}
        if row.get("capability") == "solana.market.scan":
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


def prepare_solana_training_rows(dataset_export: dict[str, Any]) -> dict[str, Any]:
    """Build a conservative baseline-only research matrix; never authorize training.

    A label is retained only when both referenced snapshots are present in this
    export, the export reports no label-reference issues, both snapshots are
    valid market-scan rows, all baseline features are finite and positive where
    required, and the sampled-return outcome is finite. Target snapshot values
    are never copied into the feature vector.
    """
    if not isinstance(dataset_export, dict) or dataset_export.get("schema") != "eti.solana.research_dataset.v1":
        raise ValueError("unsupported Solana research dataset export")
    snapshots = dataset_export.get("snapshots")
    labels = dataset_export.get("labels")
    if not isinstance(snapshots, list) or not isinstance(labels, list):
        raise ValueError("dataset export must contain snapshot and label lists")

    snapshot_by_id = {
        row.get("id"): row for row in snapshots
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    diagnostics = validate_snapshot_rows(snapshots)
    invalid_snapshot_ids = {
        row.get("id") for row in snapshots
        if isinstance(row, dict) and validate_snapshot_rows([row])["rows_with_issues"]
    }
    # The aggregate diagnostic intentionally caps its displayed issue list;
    # per-row checks above ensure that cap never weakens preparation filtering.
    invalid_snapshot_ids.update(
        issue.get("snapshot_id") for issue in diagnostics["row_issues"]
    )
    eligible: list[dict[str, Any]] = []
    excluded: dict[str, int] = {}

    def reject(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for label in labels:
        if not isinstance(label, dict):
            reject("malformed_label")
            continue
        if label.get("label_parse_error") or not isinstance(label.get("label"), dict):
            reject("malformed_label")
            continue
        if label.get("reference_issues") != []:
            reject("label_reference_issues")
            continue
        baseline_id = label.get("baseline_snapshot_id")
        target_id = label.get("target_snapshot_id")
        if not label.get("baseline_snapshot_in_export") or not label.get("target_snapshot_in_export"):
            reject("referenced_snapshot_not_in_export")
            continue
        baseline = snapshot_by_id.get(baseline_id)
        target = snapshot_by_id.get(target_id)
        if baseline is None or target is None:
            reject("referenced_snapshot_missing")
            continue
        if baseline_id == target_id:
            reject("invalid_snapshot_pair")
            continue
        if baseline_id in invalid_snapshot_ids or target_id in invalid_snapshot_ids:
            reject("snapshot_validation_issues")
            continue
        if baseline.get("capability") != "solana.market.scan" or target.get("capability") != "solana.market.scan":
            reject("unsupported_snapshot_capability")
            continue
        if (
            isinstance(baseline.get("schema_version"), bool)
            or isinstance(target.get("schema_version"), bool)
            or baseline.get("schema_version") != 1
            or target.get("schema_version") != 1
        ):
            reject("unsupported_snapshot_schema")
            continue
        if (baseline.get("mint"), baseline.get("pool_address")) != (label.get("mint"), label.get("pool_address")):
            reject("baseline_identity_mismatch")
            continue
        if (target.get("mint"), target.get("pool_address")) != (label.get("mint"), label.get("pool_address")):
            reject("target_identity_mismatch")
            continue
        horizon = label.get("horizon_seconds")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            reject("invalid_horizon")
            continue
        outcome = label["label"].get("sampled_price_return_pct")
        if isinstance(outcome, bool) or not isinstance(outcome, (int, float)) or not _is_finite_number(outcome):
            reject("invalid_outcome")
            continue
        evidence = baseline.get("payload", {}).get("evidence")
        if not isinstance(evidence, dict):
            reject("missing_baseline_evidence")
            continue
        features: dict[str, int | float] = {}
        feature_error = False
        for field in NUMERIC_FIELDS:
            value = evidence.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not _is_finite_number(value):
                feature_error = True
                break
            if value < 0 or (value == 0 and field in {"price_usd", "liquidity_usd"}):
                feature_error = True
                break
            features[field] = value
        if feature_error:
            reject("invalid_baseline_features")
            continue
        eligible.append({
            "label_id": label.get("id"),
            "mint": label.get("mint"),
            "pool_address": label.get("pool_address"),
            "baseline_snapshot_id": baseline_id,
            "target_snapshot_id": target_id,
            "horizon_seconds": horizon,
            "features": features,
            "target_sampled_price_return_pct": outcome,
        })

    return {
        "status": "PREPARATION_ONLY",
        "training_authorized": False,
        "prediction_features_source": "baseline_snapshot_only",
        "target_outcome": "sampled_price_return_pct",
        "rows": eligible,
        "eligible_row_count": len(eligible),
        "excluded_label_count": sum(excluded.values()),
        "exclusion_counts": excluded,
        "limitations": [
            "This helper prepares research rows only; it does not fit, train, approve, or deploy a model.",
            "Target snapshot values are never included in prediction features.",
            "Both baseline and target snapshots must be present and pass export validation; bounded exports may therefore exclude otherwise valid labels.",
            "Sampled price returns are not executable returns or net P&L.",
        ],
    }
