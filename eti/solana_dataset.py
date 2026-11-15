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
            "baseline_at": baseline.get("fetched_at"),
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



def evaluate_solana_discovery_cohort_coverage(
    events: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure discovery-to-research coverage without treating missing data as failure.

    Event/task/snapshot linkage is point-in-time and owner scoping must be applied
    by the caller before invoking this pure diagnostic.
    """
    if not all(isinstance(value, list) for value in (events, snapshots, tasks)):
        raise ValueError("events, snapshots and tasks must be lists")

    def parse_time(value: Any) -> datetime | None:
        return _timestamp(value)

    task_by_event: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        payload = task.get("input")
        if not isinstance(payload, dict):
            payload = task.get("input_json")
        if isinstance(payload, str):
            try:
                import json
                payload = json.loads(payload)
            except (TypeError, ValueError):
                payload = None
        if isinstance(payload, dict) and isinstance(payload.get("source_event_id"), str):
            task_by_event.setdefault(payload["source_event_id"], []).append(task)

    snapshots_by_mint: dict[str, list[dict[str, Any]]] = {}
    for snapshot in snapshots:
        if isinstance(snapshot, dict) and isinstance(snapshot.get("mint"), str):
            snapshots_by_mint.setdefault(snapshot["mint"], []).append(snapshot)

    counts: dict[str, dict[str, int]] = {}
    event_rows: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        event_id = event.get("id")
        mint = event.get("mint")
        event_type = event.get("event_type")
        if not isinstance(event_id, str) or not isinstance(mint, str) or event_type not in {"new_token", "migration"}:
            continue
        if event_id in seen_event_ids:
            continue
        seen_event_ids.add(event_id)
        received = parse_time(event.get("received_at"))
        linked_tasks = task_by_event.get(event_id, [])
        task_statuses = sorted({
            str(task.get("status", "UNKNOWN")).lower() for task in linked_tasks
        })
        linked_task_ids = {
            task.get("id") for task in linked_tasks if isinstance(task.get("id"), str)
        }
        later_snapshots = []
        for snapshot in snapshots_by_mint.get(mint, []):
            stamp = parse_time(snapshot.get("fetched_at"))
            if (
                received is not None and stamp is not None and stamp >= received
                and snapshot.get("task_id") in linked_task_ids
            ):
                later_snapshots.append(snapshot)
        scan_snapshots = [
            row for row in later_snapshots
            if row.get("capability") == "solana.market.scan"
        ]
        if scan_snapshots:
            coverage_status = "SCAN_SNAPSHOT_RECORDED"
        elif "failed" in task_statuses:
            coverage_status = "SCAN_TASK_FAILED_NO_SNAPSHOT"
        elif any(status in {"cancelled", "canceled"} for status in task_statuses):
            coverage_status = "SCAN_TASK_CANCELLED_NO_SNAPSHOT"
        elif linked_tasks:
            coverage_status = "SCAN_TASK_PENDING_OR_NO_RESULT"
        else:
            coverage_status = "NO_LINKED_SCAN_TASK"
        bucket = counts.setdefault(event_type, {
            "event_count": 0, "distinct_mint_count": 0,
            "scan_snapshot_recorded_event_count": 0,
            "failed_task_without_snapshot_event_count": 0,
            "cancelled_task_without_snapshot_event_count": 0,
            "pending_or_unresolved_event_count": 0,
            "no_linked_task_event_count": 0,
        })
        bucket["event_count"] += 1
        if scan_snapshots:
            bucket["scan_snapshot_recorded_event_count"] += 1
        elif coverage_status == "SCAN_TASK_FAILED_NO_SNAPSHOT":
            bucket["failed_task_without_snapshot_event_count"] += 1
        elif coverage_status == "SCAN_TASK_CANCELLED_NO_SNAPSHOT":
            bucket["cancelled_task_without_snapshot_event_count"] += 1
        elif coverage_status == "SCAN_TASK_PENDING_OR_NO_RESULT":
            bucket["pending_or_unresolved_event_count"] += 1
        else:
            bucket["no_linked_task_event_count"] += 1
        event_rows.append({
            "event_id": event_id,
            "event_type": event_type,
            "mint": mint,
            "received_at": event.get("received_at"),
            "linked_task_count": len(linked_tasks),
            "task_statuses": task_statuses,
            "later_market_scan_snapshot_count": len(scan_snapshots),
            "later_audit_snapshot_count": sum(
                row.get("capability") == "solana.token.audit" for row in later_snapshots
            ),
            "coverage_status": coverage_status,
        })

    for event_type, bucket in counts.items():
        bucket["distinct_mint_count"] = len({
            row["mint"] for row in event_rows if row["event_type"] == event_type
        })
        bucket["scan_snapshot_event_coverage_rate"] = (
            bucket["scan_snapshot_recorded_event_count"] / bucket["event_count"]
            if bucket["event_count"] else None
        )

    return {
        "status": "DISCOVERY_COHORT_COVERAGE_DIAGNOSTICS",
        "by_event_type": counts,
        "event_rows": event_rows[:1000],
        "event_rows_truncated": len(event_rows) > 1000,
        "unresolved_outcomes_are_not_failures": True,
        "disappearance_or_rug_pull_inferred": False,
        "survivorship_bias_ruled_out": False,
        "training_authorized": False,
        "note": (
            "Only snapshots whose task IDs are explicitly linked to the discovery event are counted. "
            "A missing snapshot does not establish token failure, disappearance, inactivity, "
            "rug pull or delisting. Task failure is reported only from an explicit terminal "
            "task status. This diagnostic cannot rule out survivorship bias or provider gaps."
        ),
    }


def evaluate_solana_token_lifecycle_coverage(
    events: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    refresh_jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Describe only lifecycle states supported by explicit ledger evidence.

    No absence of events, scans, or provider updates is interpreted as token
    inactivity, delisting, rug pull, or terminal failure.
    """
    if not all(isinstance(value, list) for value in (events, snapshots, tasks, refresh_jobs)):
        raise ValueError("events, snapshots, tasks and refresh_jobs must be lists")

    def parse_time(value: Any) -> datetime | None:
        return _timestamp(value)

    events_by_mint: dict[str, list[dict[str, Any]]] = {}
    snapshots_by_mint: dict[str, list[dict[str, Any]]] = {}
    tasks_by_refresh: dict[str, list[dict[str, Any]]] = {}
    mint_set: set[str] = set()

    for event in events:
        if isinstance(event, dict) and isinstance(event.get("mint"), str):
            mint = event["mint"]
            mint_set.add(mint)
            events_by_mint.setdefault(mint, []).append(event)
    for snapshot in snapshots:
        if isinstance(snapshot, dict) and isinstance(snapshot.get("mint"), str):
            mint = snapshot["mint"]
            mint_set.add(mint)
            snapshots_by_mint.setdefault(mint, []).append(snapshot)
    for task in tasks:
        if not isinstance(task, dict):
            continue
        payload = task.get("input")
        if isinstance(payload, dict) and isinstance(payload.get("refresh_job_id"), str):
            tasks_by_refresh.setdefault(payload["refresh_job_id"], []).append(task)
        if isinstance(payload, dict) and isinstance(payload.get("mint"), str):
            mint_set.add(payload["mint"])
    for job in refresh_jobs:
        if isinstance(job, dict) and isinstance(job.get("mint"), str):
            mint_set.add(job["mint"])

    rows: list[dict[str, Any]] = []
    for mint in sorted(mint_set):
        mint_events = events_by_mint.get(mint, [])
        mint_snapshots = snapshots_by_mint.get(mint, [])
        migrations = [
            event for event in mint_events if event.get("event_type") == "migration"
        ]
        pools = sorted({
            row.get("pool_address") for row in mint_snapshots
            if isinstance(row.get("pool_address"), str) and row.get("pool_address")
            and row.get("pool_address") != "__pending__"
        })
        jobs = [
            job for job in refresh_jobs
            if isinstance(job, dict) and job.get("mint") == mint
        ]
        job_rows = []
        for job in jobs:
            job_id = job.get("id")
            linked_tasks = tasks_by_refresh.get(job_id, []) if isinstance(job_id, str) else []
            ordered_tasks = sorted(
                linked_tasks,
                key=lambda row: parse_time(row.get("created_at")) or datetime.min.replace(tzinfo=__import__("datetime").timezone.utc),
            )
            latest_task = ordered_tasks[-1] if ordered_tasks else None
            matching_snapshots = [
                row for row in mint_snapshots
                if row.get("pool_address") == job.get("pool_address")
            ]
            latest_snapshot = max(
                matching_snapshots,
                key=lambda row: parse_time(row.get("fetched_at")) or datetime.min.replace(tzinfo=__import__("datetime").timezone.utc),
                default=None,
            )
            task_status = str(latest_task.get("status", "unknown")).lower() if latest_task else "no_task"
            if latest_snapshot is not None:
                status = "POOL_SCAN_OBSERVED"
            elif task_status == "failed":
                status = "REFRESH_TASK_FAILED_NO_SNAPSHOT"
            elif task_status in {"queued", "running"}:
                status = "REFRESH_TASK_PENDING"
            elif task_status in {"cancelled", "canceled"}:
                status = "REFRESH_TASK_CANCELLED"
            elif task_status == "completed":
                status = "COMPLETED_TASK_NO_SNAPSHOT_UNRESOLVED"
            elif job.get("enabled") is True or job.get("enabled") == 1:
                status = "REFRESH_SCHEDULED_NO_TASK_EVIDENCE"
            else:
                status = "NO_REFRESH_OR_SCAN_EVIDENCE"
            job_rows.append({
                "refresh_job_id": job_id,
                "pool_address": job.get("pool_address"),
                "enabled": bool(job.get("enabled")),
                "interval_seconds": job.get("interval_seconds"),
                "next_due_at": job.get("next_due_at"),
                "last_enqueued_at": job.get("last_enqueued_at"),
                "latest_task_status": task_status,
                "latest_snapshot_at": latest_snapshot.get("fetched_at") if latest_snapshot else None,
                "status": status,
            })

        if migrations:
            lifecycle_status = "MIGRATION_EVENT_OBSERVED"
        elif mint_snapshots:
            lifecycle_status = "MARKET_SCAN_OBSERVED"
        elif any(row["status"] == "REFRESH_TASK_FAILED_NO_SNAPSHOT" for row in job_rows):
            lifecycle_status = "REFRESH_FAILURE_EVIDENCE"
        elif mint_events:
            lifecycle_status = "DISCOVERED_NO_MARKET_SCAN_EVIDENCE"
        elif jobs:
            lifecycle_status = "REFRESH_SCHEDULED_NO_MARKET_SCAN_EVIDENCE"
        else:
            lifecycle_status = "UNRESOLVED"

        rows.append({
            "mint": mint,
            "discovery_event_count": len(mint_events),
            "migration_event_count": len(migrations),
            "distinct_observed_pool_count": len(pools),
            "observed_pool_addresses": pools,
            "market_scan_snapshot_count": sum(
                row.get("capability") == "solana.market.scan" for row in mint_snapshots
            ),
            "audit_snapshot_count": sum(
                row.get("capability") == "solana.token.audit" for row in mint_snapshots
            ),
            "latest_snapshot_at": max(
                (row.get("fetched_at") for row in mint_snapshots if parse_time(row.get("fetched_at")) is not None),
                default=None,
            ),
            "refresh_jobs": job_rows,
            "lifecycle_evidence_status": lifecycle_status,
        })

    status_counts: dict[str, int] = {}
    for row in rows:
        status = row["lifecycle_evidence_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "status": "TOKEN_LIFECYCLE_EVIDENCE_COVERAGE",
        "mint_count": len(rows),
        "status_counts": status_counts,
        "tokens": rows[:1000],
        "tokens_truncated": len(rows) > 1000,
        "provider_gap_confirmed": False,
        "token_inactivity_inferred": False,
        "delisting_inferred": False,
        "rug_pull_inferred": False,
        "survivorship_bias_ruled_out": False,
        "training_authorized": False,
        "note": (
            "A missing or stale snapshot is not proof of provider failure or token inactivity. "
            "Migration is reported only from an explicit migration event. Refresh-task outcomes "
            "are reported separately from token lifecycle. Multiple observed pools do not by "
            "themselves prove a migration relationship."
        ),
    }
