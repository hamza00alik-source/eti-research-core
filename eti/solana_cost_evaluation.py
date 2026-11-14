"""Owner/workspace-scoped bridge from the durable cost ledger to diagnostics.

This service selects the newest ledger record for each exported label. If that
newest record fails its stored-byte integrity check, it is rejected; older
records are never silently used as a fallback.
"""
from __future__ import annotations

from typing import Any

from eti.solana_evaluation import evaluate_prepared_solana_walk_forward


def evaluate_stored_solana_walk_forward(
    store: Any,
    owner_id: str,
    workspace_id: str,
    exported_labels: list[dict[str, Any]],
    prepared_dataset: dict[str, Any],
    **evaluation_options: Any,
) -> dict[str, Any]:
    """Evaluate validation rows using only the newest intact scoped ledger item.

    The store enforces owner/workspace scope. The evaluator still checks the
    evidence envelope's label, mint and pool identity. This function does not
    authenticate evidence sources or infer freshness.
    """
    if not isinstance(exported_labels, list):
        raise ValueError("exported_labels must be a list")

    evidence_by_label_id: dict[str, dict[str, Any]] = {}
    selection_by_label_id: dict[str, dict[str, Any]] = {}
    for exported in exported_labels:
        if not isinstance(exported, dict):
            raise ValueError("every exported label must be an object")
        label_id = exported.get("id")
        if not isinstance(label_id, str) or not label_id.strip():
            raise ValueError("stored-ledger evaluation requires string label IDs")
        if label_id in selection_by_label_id:
            raise ValueError("export contains duplicate label IDs")

        records = store.list_solana_execution_cost_evidence(
            owner_id, workspace_id, label_id=label_id, limit=1000
        )
        if not records:
            selection_by_label_id[label_id] = {
                "status": "NO_LEDGER_EVIDENCE",
                "record_id": None,
            }
            continue

        # Storage orders newest first by created_at DESC, then id DESC.
        newest = records[0]
        if newest.get("integrity_valid") is not True:
            selection_by_label_id[label_id] = {
                "status": "NEWEST_RECORD_INTEGRITY_INVALID",
                "record_id": newest.get("id"),
                "evidence_sha256": newest.get("evidence_sha256"),
            }
            continue

        envelope = newest.get("evidence")
        if not isinstance(envelope, dict):
            selection_by_label_id[label_id] = {
                "status": "NEWEST_RECORD_INVALID_ENVELOPE",
                "record_id": newest.get("id"),
                "evidence_sha256": newest.get("evidence_sha256"),
            }
            continue

        evidence_by_label_id[label_id] = envelope
        selection_by_label_id[label_id] = {
            "status": "NEWEST_RECORD_INTEGRITY_VALID",
            "record_id": newest.get("id"),
            "evidence_sha256": newest.get("evidence_sha256"),
        }

    report = evaluate_prepared_solana_walk_forward(
        exported_labels,
        prepared_dataset,
        execution_cost_evidence_by_label_id=evidence_by_label_id,
        **evaluation_options,
    )

    cost_report = report.get("execution_cost_evidence", {})
    rows = cost_report.get("rows", [])
    for row in rows:
        selection = selection_by_label_id.get(row.get("label_id"), {
            "status": "NO_LEDGER_EVIDENCE",
            "record_id": None,
        })
        row["ledger_evidence_status"] = selection["status"]
        row["ledger_record_id"] = selection.get("record_id")
        if selection.get("evidence_sha256") is not None:
            row["ledger_evidence_sha256"] = selection["evidence_sha256"]

        if selection["status"] in {
            "NEWEST_RECORD_INTEGRITY_INVALID",
            "NEWEST_RECORD_INVALID_ENVELOPE",
        }:
            row["status"] = "COST_EVIDENCE_INCOMPLETE"
            row["cost_evidence_complete"] = False
            row["cost_adjusted_sampled_mark_scenario_return_pct"] = None
            row["measured_round_trip_cost_bps"] = None
            row["invalid_components"].append("ledger_integrity_invalid")

    selected_statuses = [row["ledger_evidence_status"] for row in rows]
    cost_report["ledger_records_checked"] = len(rows)
    cost_report["ledger_integrity_valid_count"] = sum(
        status == "NEWEST_RECORD_INTEGRITY_VALID" for status in selected_statuses
    )
    cost_report["ledger_integrity_invalid_count"] = sum(
        status in {"NEWEST_RECORD_INTEGRITY_INVALID", "NEWEST_RECORD_INVALID_ENVELOPE"}
        for status in selected_statuses
    )
    cost_report["ledger_evidence_missing_count"] = sum(
        status == "NO_LEDGER_EVIDENCE" for status in selected_statuses
    )
    cost_report["ledger_selection_policy"] = (
        "Newest record by created_at DESC,id DESC; if newest fails integrity, "
        "reject it and do not fall back to an older record."
    )
    cost_report["note"] += (
        " Evidence was loaded from the authenticated owner/workspace ledger; "
        "ledger hashes verify stored-byte consistency only, not source authenticity."
    )
    # Recompute counts after ledger integrity failures were applied.
    complete_rows = [
        row for row in rows
        if row.get("cost_evidence_complete") is True
        and isinstance(row.get("cost_adjusted_sampled_mark_scenario_return_pct"), (int, float))
    ]
    cost_report["complete_evidence_count"] = len(complete_rows)
    cost_report["incomplete_evidence_count"] = len(rows) - len(complete_rows)
    scenarios = [float(row["cost_adjusted_sampled_mark_scenario_return_pct"]) for row in complete_rows]
    cost_report["cost_adjusted_sampled_mark_scenario_mean_pct"] = (
        sum(scenarios) / len(scenarios) if scenarios else None
    )
    ordered = sorted(scenarios)
    if ordered:
        middle = len(ordered) // 2
        cost_report["cost_adjusted_sampled_mark_scenario_median_pct"] = (
            ordered[middle] if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
    else:
        cost_report["cost_adjusted_sampled_mark_scenario_median_pct"] = None
    return report
