from datetime import datetime, timezone

import pytest

from eti.solana_dataset import validate_snapshot_rows
from eti.storage import MemoryStore


def test_dataset_validator_flags_missing_negative_zero_and_duplicate_values():
    rows = [
        {
            "id": "one", "capability": "solana.market.scan", "mint": "A" * 40, "pool_address": "PoolA",
            "fetched_at": "2026-01-01T00:00:00Z",
            "payload": {"evidence": {
                "price_usd": 0, "liquidity_usd": -1, "volume_h24_usd": 2,
                "txns_h1_buys": 1, "txns_h1_sells": 0,
            }},
        },
        {
            "id": "two", "capability": "solana.market.scan", "mint": "A" * 40, "pool_address": "PoolA",
            "fetched_at": "2026-01-01T00:00:00Z",
            "payload": {"evidence": {}},
        },
    ]
    report = validate_snapshot_rows(rows)
    assert report["rows_checked"] == 2
    assert report["issue_counts"]["zero_price_or_liquidity"] == 1
    assert report["issue_counts"]["negative_feature_values"] == 1
    assert report["issue_counts"]["duplicate_entity_timestamp"] == 1
    assert report["rows_with_issues"] == 2


def test_dataset_export_is_owner_scoped_and_stably_ordered(tmp_path):
    store = MemoryStore(tmp_path / "dataset-export.sqlite3")
    workspace = store.create_workspace("owner-1", "export")
    with store._connection() as conn:
        for index, stamp in enumerate(("2026-01-02T00:00:00Z", "2026-01-01T00:00:00Z")):
            payload = '{"evidence":{"price_usd":1,"liquidity_usd":2,"volume_h24_usd":3,"txns_h1_buys":4,"txns_h1_sells":5}}'
            import hashlib
            digest = hashlib.sha256(payload.encode()).hexdigest()
            conn.execute(
                "INSERT INTO solana_research_snapshots "
                "(id,owner_id,workspace_id,task_id,capability,mint,pool_address,schema_version,"
                "payload_json,payload_sha256,fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (f"s{index}", "owner-1", workspace["id"], f"t{index}", "solana.market.scan",
                 "A" * 40, "PoolA", 1, payload, digest, stamp),
            )
    exported = store.export_solana_research_dataset("owner-1", workspace["id"])
    assert [item["id"] for item in exported["snapshots"]] == ["s1", "s0"]
    assert exported["schema"] == "eti.solana.research_dataset.v1"
    assert exported["validation"]["rows_checked"] == 2
    import pytest
    with pytest.raises(LookupError, match="workspace not found"):
        store.export_solana_research_dataset("owner-2", workspace["id"])


def test_dataset_export_respects_limit_and_mint_filter(tmp_path):
    store = MemoryStore(tmp_path / "dataset-export-limit.sqlite3")
    workspace = store.create_workspace("owner-1", "export")
    with store._connection() as conn:
        for index, mint in enumerate(("A" * 40, "B" * 40)):
            payload = '{"evidence":{}}'
            import hashlib
            conn.execute(
                "INSERT INTO solana_research_snapshots "
                "(id,owner_id,workspace_id,task_id,capability,mint,pool_address,schema_version,"
                "payload_json,payload_sha256,fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (f"s{index}", "owner-1", workspace["id"], f"t{index}", "solana.market.scan",
                 mint, "PoolA", 1, payload, hashlib.sha256(payload.encode()).hexdigest(),
                 f"2026-01-0{index+1}T00:00:00Z"),
            )
    result = store.export_solana_research_dataset("owner-1", workspace["id"], limit=1, mint="B" * 40)
    assert result["snapshot_count"] == 1
    assert result["snapshots"][0]["mint"] == "B" * 40



def test_dataset_validator_does_not_apply_market_features_to_audit_snapshots():
    report = validate_snapshot_rows([{
        "id": "audit", "capability": "solana.token.audit", "mint": "A" * 40,
        "pool_address": "PoolA", "fetched_at": "2026-01-01T00:00:00Z",
        "payload": {"passed": False, "checks": []},
    }])
    assert report["issue_counts"]["missing_evidence"] == 0
    assert report["issue_counts"]["missing_feature_values"] == 0


def test_bounded_export_marks_label_target_outside_snapshot_window(tmp_path):
    import hashlib

    store = MemoryStore(tmp_path / "bounded-label-export.sqlite3")
    workspace = store.create_workspace("owner-1", "bounded")
    payload = '{"evidence":{"price_usd":1,"liquidity_usd":2,"volume_h24_usd":3,"txns_h1_buys":4,"txns_h1_sells":5}}'
    digest = hashlib.sha256(payload.encode()).hexdigest()
    with store._connection() as conn:
        for snapshot_id, task_id, stamp in (
            ("baseline", "task-1", "2026-01-01T00:00:00Z"),
            ("target", "task-2", "2026-01-02T00:00:00Z"),
        ):
            conn.execute(
                "INSERT INTO solana_research_snapshots "
                "(id,owner_id,workspace_id,task_id,capability,mint,pool_address,schema_version,"
                "payload_json,payload_sha256,fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (snapshot_id, "owner-1", workspace["id"], task_id, "solana.market.scan",
                 "A" * 40, "PoolA", 1, payload, digest, stamp),
            )
        conn.execute(
            "INSERT INTO solana_research_labels "
            "(id,owner_id,workspace_id,mint,pool_address,baseline_snapshot_id,target_snapshot_id,"
            "horizon_seconds,label_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("label-1", "owner-1", workspace["id"], "A" * 40, "PoolA",
             "baseline", "target", 300, '{"sampled_price_return_pct":10}',
             "2026-01-02T00:00:00Z"),
        )

    exported = store.export_solana_research_dataset("owner-1", workspace["id"], limit=1)
    assert exported["truncated_at_limit"] is True
    assert [row["id"] for row in exported["snapshots"]] == ["baseline"]
    assert exported["labels"][0]["baseline_snapshot_in_export"] is True
    assert exported["labels"][0]["target_snapshot_in_export"] is False

def test_dataset_export_reports_malformed_stored_json_without_crashing(tmp_path):
    store = MemoryStore(tmp_path / "malformed-export.sqlite3")
    workspace = store.create_workspace("owner-1", "malformed")
    with store._connection() as conn:
        conn.execute(
            "INSERT INTO solana_research_snapshots "
            "(id,owner_id,workspace_id,task_id,capability,mint,pool_address,schema_version,"
            "payload_json,payload_sha256,fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            ("bad-snapshot", "owner-1", workspace["id"], "task-bad", "solana.market.scan",
             "A" * 40, "PoolA", 1, "{not-json", "invalid-digest",
             "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO solana_research_labels "
            "(id,owner_id,workspace_id,mint,pool_address,baseline_snapshot_id,target_snapshot_id,"
            "horizon_seconds,label_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("bad-label", "owner-1", workspace["id"], "A" * 40, "PoolA",
             "bad-snapshot", "missing-target", 300, "{not-json",
             "2026-01-02T00:00:00Z"),
        )

    exported = store.export_solana_research_dataset("owner-1", workspace["id"])
    assert exported["snapshots"][0]["payload"] is None
    assert exported["validation"]["issue_counts"]["malformed_payload"] == 1
    assert exported["validation"]["malformed_label_count"] == 1
    assert exported["labels"][0]["label"] is None
    assert exported["labels"][0]["label_parse_error"] is True


def test_dataset_validator_flags_unknown_and_malformed_schema_versions():
    rows = [
        {
            "id": "future-version", "schema_version": 2,
            "capability": "solana.token.audit", "mint": "A" * 40,
            "pool_address": "PoolA", "fetched_at": "2026-01-01T00:00:00Z",
            "payload": {"passed": True},
        },
        {
            "id": "missing-version", "schema_version": True,
            "capability": "solana.token.audit", "mint": "B" * 40,
            "pool_address": "PoolB", "fetched_at": "2026-01-01T00:00:00Z",
            "payload": {"passed": True},
        },
    ]
    report = validate_snapshot_rows(rows)
    assert report["issue_counts"]["unsupported_schema_version"] == 2
    assert report["rows_with_issues"] == 2
    assert all("unsupported_schema_version" in row["issues"] for row in report["row_issues"])


def test_dataset_export_flags_label_reference_identity_and_time_corruption(tmp_path):
    import hashlib

    store = MemoryStore(tmp_path / "label-reference-integrity.sqlite3")
    workspace = store.create_workspace("owner-1", "integrity")
    payload = '{"evidence":{"price_usd":1,"liquidity_usd":2,"volume_h24_usd":3,"txns_h1_buys":4,"txns_h1_sells":5}}'
    digest = hashlib.sha256(payload.encode()).hexdigest()
    with store._connection() as conn:
        snapshots = [
            ("base", "2026-01-01T00:00:00Z", "A" * 40, "PoolA"),
            ("target", "2026-01-01T00:04:00Z", "B" * 40, "PoolB"),
        ]
        for snapshot_id, stamp, mint, pool in snapshots:
            conn.execute(
                "INSERT INTO solana_research_snapshots "
                "(id,owner_id,workspace_id,task_id,capability,mint,pool_address,schema_version,"
                "payload_json,payload_sha256,fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (snapshot_id, "owner-1", workspace["id"], "task-" + snapshot_id,
                 "solana.market.scan", mint, pool, 1, payload, digest, stamp),
            )
        conn.execute(
            "INSERT INTO solana_research_labels "
            "(id,owner_id,workspace_id,mint,pool_address,baseline_snapshot_id,target_snapshot_id,"
            "horizon_seconds,label_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("label-corrupt", "owner-1", workspace["id"], "A" * 40, "PoolA",
             "base", "target", 300, '{"sampled_price_return_pct":10}',
             "2026-01-01T00:04:00Z"),
        )

    exported = store.export_solana_research_dataset("owner-1", workspace["id"])
    label = exported["labels"][0]
    assert "target_identity_mismatch" in label["reference_issues"]
    assert "target_before_label_horizon" in label["reference_issues"]
    assert exported["validation"]["label_reference_issue_count"] == 1


def test_label_generation_skips_malformed_and_unsupported_snapshot_rows(tmp_path):
    import hashlib

    store = MemoryStore(tmp_path / "label-generation-invalid-snapshots.sqlite3")
    workspace = store.create_workspace("owner-1", "label-generation")
    payload = '{"evidence":{"price_usd":1,"liquidity_usd":2,"volume_h24_usd":3,"txns_h1_buys":4,"txns_h1_sells":5}}'
    digest = hashlib.sha256(payload.encode()).hexdigest()
    with store._connection() as conn:
        rows = [
            ("bad-json", 1, "{not-json", "2026-01-01T00:00:00Z"),
            ("future-schema", 2, payload, "2026-01-01T00:06:00Z"),
        ]
        for snapshot_id, schema_version, stored_payload, stamp in rows:
            conn.execute(
                "INSERT INTO solana_research_snapshots "
                "(id,owner_id,workspace_id,task_id,capability,mint,pool_address,schema_version,"
                "payload_json,payload_sha256,fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (snapshot_id, "owner-1", workspace["id"], "task-" + snapshot_id,
                 "solana.market.scan", "A" * 40, "PoolA", schema_version,
                 stored_payload, digest, stamp),
            )

    result = store.generate_solana_research_labels(
        "owner-1", workspace["id"],
        now=datetime(2026, 1, 1, 0, 20, tzinfo=timezone.utc),
    )
    assert result == {
        "generated": 0, "available": 0, "skipped_invalid_snapshots": 2,
        "unchanged": 0, "conflicts": 0,
    }


def test_dataset_export_does_not_resolve_label_target_from_another_owner(tmp_path):
    import hashlib

    store = MemoryStore(tmp_path / "cross-owner-label-reference.sqlite3")
    owner_workspace = store.create_workspace("owner-1", "owner")
    other_workspace = store.create_workspace("owner-2", "other")
    payload = '{"evidence":{"price_usd":1,"liquidity_usd":2,"volume_h24_usd":3,"txns_h1_buys":4,"txns_h1_sells":5}}'
    digest = hashlib.sha256(payload.encode()).hexdigest()
    with store._connection() as conn:
        conn.execute(
            "INSERT INTO solana_research_snapshots "
            "(id,owner_id,workspace_id,task_id,capability,mint,pool_address,schema_version,"
            "payload_json,payload_sha256,fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            ("owner-baseline", "owner-1", owner_workspace["id"], "task-base",
             "solana.market.scan", "A" * 40, "PoolA", 1, payload, digest,
             "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO solana_research_snapshots "
            "(id,owner_id,workspace_id,task_id,capability,mint,pool_address,schema_version,"
            "payload_json,payload_sha256,fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            ("foreign-target", "owner-2", other_workspace["id"], "task-target",
             "solana.market.scan", "A" * 40, "PoolA", 1, payload, digest,
             "2026-01-01T00:10:00Z"),
        )
        conn.execute(
            "INSERT INTO solana_research_labels "
            "(id,owner_id,workspace_id,mint,pool_address,baseline_snapshot_id,target_snapshot_id,"
            "horizon_seconds,label_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("owner-label", "owner-1", owner_workspace["id"], "A" * 40, "PoolA",
             "owner-baseline", "foreign-target", 300,
             '{"sampled_price_return_pct":0}', "2026-01-01T00:10:00Z"),
        )

    exported = store.export_solana_research_dataset("owner-1", owner_workspace["id"])
    label = exported["labels"][0]
    assert "target_snapshot_missing_or_outside_workspace" in label["reference_issues"]
    assert label["target_snapshot_in_export"] is False
    assert exported["validation"]["label_reference_issue_count"] == 1
    assert all(row["id"] != "foreign-target" for row in exported["snapshots"])


def test_training_preparation_uses_only_valid_baseline_features():
    from eti.solana_dataset import prepare_solana_training_rows

    baseline = {
        "id": "base", "capability": "solana.market.scan", "mint": "A" * 40,
        "pool_address": "PoolA", "schema_version": 1,
        "fetched_at": "2026-01-01T00:00:00Z",
        "payload": {"evidence": {
            "price_usd": 1.0, "liquidity_usd": 100.0, "volume_h24_usd": 20.0,
            "txns_h1_buys": 3, "txns_h1_sells": 2,
        }},
    }
    target = {
        "id": "target", "capability": "solana.market.scan", "mint": "A" * 40,
        "pool_address": "PoolA", "schema_version": 1,
        "fetched_at": "2026-01-01T00:06:00Z",
        "payload": {"evidence": {
            "price_usd": 2.0, "liquidity_usd": 90.0, "volume_h24_usd": 30.0,
            "txns_h1_buys": 5, "txns_h1_sells": 1,
        }},
    }
    exported = {
        "schema": "eti.solana.research_dataset.v1",
        "snapshots": [baseline, target],
        "labels": [{
            "id": "label-1", "mint": "A" * 40, "pool_address": "PoolA",
            "baseline_snapshot_id": "base", "target_snapshot_id": "target",
            "baseline_snapshot_in_export": True, "target_snapshot_in_export": True,
            "horizon_seconds": 300, "label_parse_error": False,
            "reference_issues": [],
            "label": {"sampled_price_return_pct": 100.0},
        }],
    }

    prepared = prepare_solana_training_rows(exported)
    assert prepared["status"] == "PREPARATION_ONLY"
    assert prepared["training_authorized"] is False
    assert prepared["eligible_row_count"] == 1
    row = prepared["rows"][0]
    assert row["features"]["price_usd"] == 1.0
    assert row["target_sampled_price_return_pct"] == 100.0
    assert row["baseline_at"] == baseline["fetched_at"]
    assert row["target_at"] == target["fetched_at"]
    assert row["actual_elapsed_seconds"] == 360.0
    assert "target_price_usd" not in row["features"]
    assert row["features"] == baseline["payload"]["evidence"]


def test_training_preparation_excludes_bad_labels_and_truncated_targets():
    from eti.solana_dataset import prepare_solana_training_rows

    baseline = {
        "id": "base", "capability": "solana.market.scan", "mint": "A" * 40,
        "pool_address": "PoolA", "schema_version": 1,
        "fetched_at": "2026-01-01T00:00:00Z",
        "payload": {"evidence": {
            "price_usd": 1.0, "liquidity_usd": 100.0, "volume_h24_usd": 20.0,
            "txns_h1_buys": 3, "txns_h1_sells": 2,
        }},
    }
    common = {
        "mint": "A" * 40, "pool_address": "PoolA",
        "baseline_snapshot_id": "base", "horizon_seconds": 300,
        "baseline_snapshot_in_export": True, "label_parse_error": False,
        "label": {"sampled_price_return_pct": 5.0},
    }
    exported = {
        "schema": "eti.solana.research_dataset.v1",
        "snapshots": [baseline],
        "labels": [
            {**common, "id": "missing-target", "target_snapshot_id": "not-exported",
             "target_snapshot_in_export": False, "reference_issues": []},
            {**common, "id": "corrupt-reference", "target_snapshot_id": "not-exported",
             "target_snapshot_in_export": True,
             "reference_issues": ["target_snapshot_missing_or_outside_workspace"]},
        ],
    }
    prepared = prepare_solana_training_rows(exported)
    assert prepared["eligible_row_count"] == 0
    assert prepared["excluded_label_count"] == 2
    assert prepared["exclusion_counts"] == {
        "referenced_snapshot_not_in_export": 1,
        "label_reference_issues": 1,
    }


def test_discovery_cohort_coverage_requires_explicit_event_task_snapshot_linkage():
    from eti.solana_dataset import evaluate_solana_discovery_cohort_coverage

    events = [
        {"id": "event-1", "event_type": "new_token", "mint": "mint-a",
         "received_at": "2026-01-01T00:00:00Z"},
        {"id": "event-2", "event_type": "migration", "mint": "mint-b",
         "received_at": "2026-01-01T00:00:00Z"},
        {"id": "event-3", "event_type": "new_token", "mint": "mint-c",
         "received_at": "2026-01-01T00:00:00Z"},
    ]
    tasks = [
        {"id": "task-1", "status": "completed",
         "input": {"source_event_id": "event-1"}},
        {"id": "task-2", "status": "failed",
         "input": {"source_event_id": "event-2"}},
    ]
    snapshots = [
        {"id": "snapshot-1", "task_id": "task-1", "mint": "mint-a",
         "capability": "solana.market.scan", "fetched_at": "2026-01-01T00:01:00Z"},
        # Same mint and later timestamp, but unrelated task: must not count.
        {"id": "snapshot-unlinked", "task_id": "other-task", "mint": "mint-c",
         "capability": "solana.market.scan", "fetched_at": "2026-01-01T00:02:00Z"},
    ]
    report = evaluate_solana_discovery_cohort_coverage(events, snapshots, tasks)
    new_token = report["by_event_type"]["new_token"]
    migration = report["by_event_type"]["migration"]
    assert new_token["event_count"] == 2
    assert new_token["scan_snapshot_recorded_event_count"] == 1
    assert new_token["scan_snapshot_event_coverage_rate"] == 0.5
    assert migration["failed_task_without_snapshot_event_count"] == 1
    statuses = {row["event_id"]: row["coverage_status"] for row in report["event_rows"]}
    assert statuses == {
        "event-1": "SCAN_SNAPSHOT_RECORDED",
        "event-2": "SCAN_TASK_FAILED_NO_SNAPSHOT",
        "event-3": "NO_LINKED_SCAN_TASK",
    }
    assert report["survivorship_bias_ruled_out"] is False
    assert report["training_authorized"] is False


def test_discovery_cohort_coverage_marks_pending_and_rejects_bad_inputs():
    from eti.solana_dataset import evaluate_solana_discovery_cohort_coverage

    report = evaluate_solana_discovery_cohort_coverage(
        [{"id": "e", "event_type": "new_token", "mint": "m",
          "received_at": "2026-01-01T00:00:00Z"}],
        [],
        [{"id": "t", "status": "queued", "input_json": '{"source_event_id":"e"}'}],
    )
    assert report["event_rows"][0]["coverage_status"] == "SCAN_TASK_PENDING_OR_NO_RESULT"
    with pytest.raises(ValueError):
        evaluate_solana_discovery_cohort_coverage([], {}, [])


def test_token_lifecycle_coverage_separates_migration_scan_and_refresh_evidence():
    from eti.solana_dataset import evaluate_solana_token_lifecycle_coverage

    events = [
        {"id": "e1", "mint": "mint-a", "event_type": "new_token"},
        {"id": "e2", "mint": "mint-a", "event_type": "migration"},
        {"id": "e3", "mint": "mint-b", "event_type": "new_token"},
        {"id": "e4", "mint": "mint-c", "event_type": "new_token"},
    ]
    snapshots = [
        {"id": "s1", "mint": "mint-a", "pool_address": "pool-old",
         "capability": "solana.market.scan", "fetched_at": "2026-01-01T00:00:00Z"},
        {"id": "s2", "mint": "mint-a", "pool_address": "pool-new",
         "capability": "solana.market.scan", "fetched_at": "2026-01-01T00:10:00Z"},
        {"id": "s3", "mint": "mint-b", "pool_address": "pool-b",
         "capability": "solana.token.audit", "fetched_at": "2026-01-01T00:05:00Z"},
    ]
    tasks = [
        {"id": "task-failed", "status": "failed",
         "input": {"refresh_job_id": "job-b", "mint": "mint-b"}},
        {"id": "task-complete", "status": "completed",
         "input": {"refresh_job_id": "job-c", "mint": "mint-c"}},
    ]
    jobs = [
        {"id": "job-b", "mint": "mint-b", "pool_address": "pool-b",
         "enabled": True, "interval_seconds": 300},
        {"id": "job-c", "mint": "mint-c", "pool_address": "pool-c",
         "enabled": True, "interval_seconds": 300},
        {"id": "job-d", "mint": "mint-d", "pool_address": "pool-d",
         "enabled": True, "interval_seconds": 300},
    ]
    report = evaluate_solana_token_lifecycle_coverage(events, snapshots, tasks, jobs)
    by_mint = {row["mint"]: row for row in report["tokens"]}
    assert by_mint["mint-a"]["lifecycle_evidence_status"] == "MIGRATION_EVENT_OBSERVED"
    assert by_mint["mint-a"]["distinct_observed_pool_count"] == 2
    assert by_mint["mint-b"]["lifecycle_evidence_status"] == "REFRESH_FAILURE_EVIDENCE"
    assert by_mint["mint-b"]["refresh_jobs"][0]["status"] == "REFRESH_TASK_FAILED_NO_SNAPSHOT"
    assert by_mint["mint-c"]["refresh_jobs"][0]["status"] == "COMPLETED_TASK_NO_SNAPSHOT_UNRESOLVED"
    assert by_mint["mint-d"]["refresh_jobs"][0]["status"] == "REFRESH_SCHEDULED_NO_TASK_EVIDENCE"
    assert report["provider_gap_confirmed"] is False
    assert report["token_inactivity_inferred"] is False
    assert report["rug_pull_inferred"] is False
    assert report["training_authorized"] is False


def test_token_lifecycle_coverage_rejects_non_list_inputs_and_does_not_infer_inactivity():
    from eti.solana_dataset import evaluate_solana_token_lifecycle_coverage

    report = evaluate_solana_token_lifecycle_coverage(
        [{"mint": "mint-x", "event_type": "new_token"}], [], [], []
    )
    assert report["tokens"][0]["lifecycle_evidence_status"] == "DISCOVERED_NO_MARKET_SCAN_EVIDENCE"
    with pytest.raises(ValueError):
        evaluate_solana_token_lifecycle_coverage([], [], {}, [])


def test_training_preparation_binds_label_times_and_outcome_to_snapshot_evidence():
    from copy import deepcopy
    from eti.solana_dataset import prepare_solana_training_rows

    baseline = {
        "id": "base", "capability": "solana.market.scan", "mint": "A" * 40,
        "pool_address": "PoolA", "schema_version": 1,
        "fetched_at": "2026-01-01T00:00:00Z",
        "payload": {"evidence": {
            "price_usd": 1.0, "liquidity_usd": 100.0, "volume_h24_usd": 20.0,
            "txns_h1_buys": 3, "txns_h1_sells": 2,
        }},
    }
    target = {
        "id": "target", "capability": "solana.market.scan", "mint": "A" * 40,
        "pool_address": "PoolA", "schema_version": 1,
        "fetched_at": "2026-01-01T00:06:00Z",
        "payload": {"evidence": {
            "price_usd": 2.0, "liquidity_usd": 90.0, "volume_h24_usd": 30.0,
            "txns_h1_buys": 5, "txns_h1_sells": 1,
        }},
    }
    label = {
        "id": "label-1", "mint": "A" * 40, "pool_address": "PoolA",
        "baseline_snapshot_id": "base", "target_snapshot_id": "target",
        "baseline_snapshot_in_export": True, "target_snapshot_in_export": True,
        "horizon_seconds": 300, "label_parse_error": False,
        "reference_issues": [],
        "label": {
            "baseline_at": baseline["fetched_at"],
            "target_at": target["fetched_at"],
            "actual_elapsed_seconds": 360.0,
            "sampled_price_return_pct": 100.0,
        },
    }
    valid_export = {
        "schema": "eti.solana.research_dataset.v1",
        "snapshots": [baseline, target], "labels": [label],
    }
    assert prepare_solana_training_rows(valid_export)["eligible_row_count"] == 1

    # A duplicated durable label key must not create duplicate training rows.
    # This also covers conflicting target references for the same baseline/horizon.
    duplicate_export = deepcopy(valid_export)
    conflicting_label = deepcopy(label)
    conflicting_label["id"] = "label-2"
    conflicting_label["target_snapshot_id"] = "other-target"
    duplicate_export["labels"].append(conflicting_label)
    duplicate_result = prepare_solana_training_rows(duplicate_export)
    assert duplicate_result["eligible_row_count"] == 0
    assert duplicate_result["excluded_label_count"] == 2
    assert duplicate_result["exclusion_counts"] == {"duplicate_label_key": 2}

    mutations = [
        ("baseline_at", "2026-01-01T00:01:00Z", "label_snapshot_timestamp_mismatch"),
        ("target_at", "2026-01-01T00:07:00Z", "label_snapshot_timestamp_mismatch"),
        ("actual_elapsed_seconds", 999.0, "label_elapsed_time_mismatch"),
        ("horizon_seconds", [], "invalid_horizon"),
        ("sampled_price_return_pct", 99.0, "label_outcome_snapshot_mismatch"),
    ]
    for field, value, reason in mutations:
        corrupted = deepcopy(valid_export)
        corrupted["labels"][0]["label"][field] = value
        result = prepare_solana_training_rows(corrupted)
        assert result["eligible_row_count"] == 0
        assert result["exclusion_counts"] == {reason: 1}

    corrupted_target = deepcopy(valid_export)
    corrupted_target["snapshots"][1]["payload"]["evidence"]["price_usd"] = 3.0
    result = prepare_solana_training_rows(corrupted_target)
    assert result["eligible_row_count"] == 0
    assert result["exclusion_counts"] == {"label_outcome_snapshot_mismatch": 1}



def test_dataset_validator_flags_every_occurrence_of_duplicate_snapshot_id():
    rows = [
        {"id": "same-id", "capability": "solana.market.scan", "mint": "MintA",
         "pool_address": "PoolA", "schema_version": 1, "fetched_at": "2026-01-01T00:00:00Z",
         "payload": {"evidence": {}}},
        {"id": "same-id", "capability": "solana.market.scan", "mint": "MintB",
         "pool_address": "PoolB", "schema_version": 1, "fetched_at": "2026-01-01T00:01:00Z",
         "payload": {"evidence": {}}},
    ]
    report = validate_snapshot_rows(rows)
    assert report["issue_counts"]["duplicate_snapshot_id"] == 2
    assert report["rows_with_issues"] == 2
    assert all("duplicate_snapshot_id" in row["issues"] for row in report["row_issues"])



def test_dataset_report_uses_conventional_median_and_reports_sampling_gaps(tmp_path):
    store = MemoryStore(tmp_path / "sampling-report.sqlite3")
    workspace = store.create_workspace("owner", "sampling")
    stamps = [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:05:00Z",
        "2026-01-01T00:15:00Z",
        "2026-01-01T00:35:00Z",
        "2026-01-01T01:05:00Z",
        "2026-01-01T01:05:00Z",
    ]
    payload = '{"evidence":{"price_usd":1,"liquidity_usd":2,"volume_h24_usd":3,"txns_h1_buys":4,"txns_h1_sells":5}}'
    import hashlib
    digest = hashlib.sha256(payload.encode()).hexdigest()
    with store._connection() as conn:
        for index, stamp in enumerate(stamps):
            conn.execute(
                "INSERT INTO solana_research_snapshots "
                "(id,owner_id,workspace_id,task_id,capability,mint,pool_address,schema_version,"
                "payload_json,payload_sha256,fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (f"sample-{index}", "owner", workspace["id"], f"task-{index}",
                 "solana.market.scan", "MintA", "PoolA", 1, payload, digest, stamp),
            )

    report = store.get_solana_research_dataset_report("owner", workspace["id"])
    sampling = report["snapshots"]
    assert sampling["sample_interval_count"] == 4
    assert sampling["median_sample_interval_seconds"] == 900.0
    assert sampling["p90_sample_interval_seconds"] == 1800.0
    assert sampling["maximum_sample_interval_seconds"] == 1800.0
    assert sampling["duplicate_timestamp_intervals"] == 1
    assert sampling["sample_gaps_over_10_minutes"] == 2
    assert sampling["sample_gap_fraction_over_10_minutes"] == 0.5
    assert report["training_gate"]["ready"] is False



def test_dataset_report_includes_full_workspace_forward_label_coverage(tmp_path):
    import hashlib
    from datetime import timedelta

    store = MemoryStore(tmp_path / "forward-coverage-report.sqlite3")
    workspace = store.create_workspace("owner", "coverage")
    now = datetime.now(timezone.utc).replace(microsecond=0)
    payload = '{"evidence":{"price_usd":1,"liquidity_usd":1000,"volume_h24_usd":500}}'
    digest = hashlib.sha256(payload.encode()).hexdigest()
    stamps = [
        ("base-1", now - timedelta(minutes=20)),
        ("target-1", now - timedelta(minutes=14)),
        ("base-2", now - timedelta(minutes=8)),
        ("base-3", now - timedelta(minutes=2)),
        ("future", now + timedelta(minutes=2)),
    ]
    with store._connection() as conn:
        for snapshot_id, stamp in stamps:
            conn.execute(
                "INSERT INTO solana_research_snapshots "
                "(id,owner_id,workspace_id,task_id,capability,mint,pool_address,schema_version,"
                "payload_json,payload_sha256,fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (snapshot_id, "owner", workspace["id"], "task-" + snapshot_id,
                 "solana.market.scan", "A" * 40, "Pool111", 1, payload, digest,
                 stamp.isoformat().replace("+00:00", "Z")),
            )
        conn.execute(
            "INSERT INTO solana_research_labels "
            "(id,owner_id,workspace_id,mint,pool_address,baseline_snapshot_id,target_snapshot_id,"
            "horizon_seconds,label_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("label-1", "owner", workspace["id"], "B" * 40, "PoolOther",
             "base-1", "target-1", 300,
             '{"mint":"' + "A" * 40 + '","pool_address":"Pool111","sampled_price_return_pct":0}',
             now.isoformat().replace("+00:00", "Z")),
        )
        conn.execute(
            "INSERT INTO solana_research_labels "
            "(id,owner_id,workspace_id,mint,pool_address,baseline_snapshot_id,target_snapshot_id,"
            "horizon_seconds,label_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("label-corrupt", "owner", workspace["id"], "A" * 40, "Pool111",
             "base-2", "base-3", 300, "{malformed-json",
             now.isoformat().replace("+00:00", "Z")),
        )
        conn.execute(
            "INSERT INTO solana_research_labels "
            "(id,owner_id,workspace_id,mint,pool_address,baseline_snapshot_id,target_snapshot_id,"
            "horizon_seconds,label_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("label-non-object", "owner", workspace["id"], "A" * 40, "Pool111",
             "base-3", "future", 300, '["not", "an", "object"]',
             now.isoformat().replace("+00:00", "Z")),
        )
    bounded = store.export_solana_research_dataset("owner", workspace["id"], limit=1)
    assert bounded["truncated_at_limit"] is True
    assert bounded["snapshot_count"] == 1
    report = store.get_solana_research_dataset_report("owner", workspace["id"])
    coverage = report["forward_label_coverage"]
    horizon = coverage["by_horizon_seconds"]["300"]
    assert horizon["eligible_baseline_count"] == 5
    assert horizon["mature_baseline_count"] == 3
    assert horizon["not_yet_mature_count"] == 2
    assert horizon["mature_with_future_target_count"] == 3
    assert horizon["mature_without_future_target_count"] == 0
    # The normalized label JSON claims the snapshot identity, but the
    # authoritative persisted label columns disagree. Coverage must reject it.
    assert horizon["label_record_count"] == 0
    # The non-object JSON row is retained safely, but its baseline is not yet
    # mature; immature baselines are deliberately excluded from stale-label
    # conflict counts.
    assert horizon["stale_label_target_conflict_count"] == 2
    assert coverage["training_authorized"] is False



def test_dataset_report_exposes_full_workspace_snapshot_validation(tmp_path):
    import hashlib

    store = MemoryStore(tmp_path / "snapshot-validation-report.sqlite3")
    workspace = store.create_workspace("owner", "quality")
    payload = (
        '{"evidence":{"price_usd":0,"liquidity_usd":-1,'
        '"volume_h24_usd":2,"txns_h1_buys":-3,"txns_h1_sells":4}}'
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    with store._connection() as conn:
        conn.execute(
            "INSERT INTO solana_research_snapshots "
            "(id,owner_id,workspace_id,task_id,capability,mint,pool_address,schema_version,"
            "payload_json,payload_sha256,fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            ("invalid-quality", "owner", workspace["id"], "task-quality",
             "solana.market.scan", "A" * 40, "PoolQuality", 1, payload, digest,
             "2026-01-01T00:00:00Z"),
        )

    report = store.get_solana_research_dataset_report("owner", workspace["id"])
    validation = report["snapshot_validation"]
    assert validation["rows_checked"] == 1
    assert validation["rows_with_issues"] == 1
    assert validation["issue_counts"]["zero_price_or_liquidity"] == 1
    assert validation["issue_counts"]["negative_feature_values"] == 2
    # Finite-value presence is not the same as usable feature coverage.
    assert report["feature_coverage"]["price_usd"]["count"] == 1
    assert report["feature_coverage"]["price_usd"]["valid_count"] == 0
    assert report["feature_coverage"]["liquidity_usd"]["count"] == 1
    assert report["feature_coverage"]["liquidity_usd"]["valid_count"] == 0
    assert report["feature_coverage"]["volume_h24_usd"]["valid_count"] == 1
    assert report["training_gate"]["ready"] is False



def test_label_generation_is_idempotent_and_surfaces_late_target_conflicts(tmp_path):
    import hashlib

    store = MemoryStore(tmp_path / "label-idempotency.sqlite3")
    workspace = store.create_workspace("owner", "idempotency")
    payloads = {
        "one": '{"evidence":{"price_usd":1,"liquidity_usd":100,"volume_h24_usd":10}}',
        "two": '{"evidence":{"price_usd":2,"liquidity_usd":100,"volume_h24_usd":10}}',
        "three": '{"evidence":{"price_usd":3,"liquidity_usd":100,"volume_h24_usd":10}}',
    }
    def add_snapshot(snapshot_id, stamp):
        payload = payloads[snapshot_id]
        with store._connection() as conn:
            conn.execute(
                "INSERT INTO solana_research_snapshots "
                "(id,owner_id,workspace_id,task_id,capability,mint,pool_address,schema_version,"
                "payload_json,payload_sha256,fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (snapshot_id, "owner", workspace["id"], "task-" + snapshot_id,
                 "solana.market.scan", "A" * 40, "PoolA", 1, payload,
                 hashlib.sha256(payload.encode()).hexdigest(), stamp),
            )

    add_snapshot("one", "2026-01-01T00:00:00Z")
    add_snapshot("three", "2026-01-01T00:10:00Z")
    now = datetime(2026, 1, 1, 0, 20, tzinfo=timezone.utc)
    first = store.generate_solana_research_labels("owner", workspace["id"], now=now)
    second = store.generate_solana_research_labels("owner", workspace["id"], now=now)
    assert first["generated"] == 1
    assert second["generated"] == 0
    assert second["unchanged"] == 1
    assert second["conflicts"] == 0

    # A late-arriving observation earlier than the selected target changes the
    # computed label. The stored row is not silently overwritten.
    add_snapshot("two", "2026-01-01T00:06:00Z")
    third = store.generate_solana_research_labels("owner", workspace["id"], now=now)
    assert third["generated"] == 0
    assert third["conflicts"] == 1
    exported = store.export_solana_research_dataset("owner", workspace["id"])
    assert len(exported["labels"]) == 1
    assert exported["labels"][0]["target_snapshot_id"] == "three"



def test_training_preparation_checks_all_invalid_snapshots_without_quadratic_revalidation():
    from eti.solana_dataset import prepare_solana_training_rows

    baseline = {
        "id": "base", "capability": "solana.market.scan", "mint": "A" * 40,
        "pool_address": "PoolA", "schema_version": 1,
        "fetched_at": "2026-01-01T00:00:00Z",
        "payload": {"evidence": {
            "price_usd": 1.0, "liquidity_usd": 100.0, "volume_h24_usd": 20.0,
            "txns_h1_buys": 3, "txns_h1_sells": 2,
        }},
    }
    invalid_rows = [
        {
            "id": f"invalid-{index}", "capability": "solana.market.scan",
            "mint": f"mint-{index}", "pool_address": f"pool-{index}",
            "schema_version": 1, "fetched_at": f"2026-01-01T00:{index // 60:02d}:{index % 60:02d}Z",
            "payload": {"evidence": {}},
        }
        for index in range(500)
    ]
    bad_target = {
        "id": "bad-target", "capability": "solana.market.scan", "mint": "A" * 40,
        "pool_address": "PoolA", "schema_version": 1,
        "fetched_at": "2026-01-01T00:06:00Z",
        "payload": {"evidence": {}},
    }
    exported = {
        "schema": "eti.solana.research_dataset.v1",
        "snapshots": [baseline, *invalid_rows, bad_target],
        "labels": [{
            "id": "label-1", "mint": "A" * 40, "pool_address": "PoolA",
            "baseline_snapshot_id": "base", "target_snapshot_id": "bad-target",
            "baseline_snapshot_in_export": True, "target_snapshot_in_export": True,
            "horizon_seconds": 300, "label_parse_error": False,
            "reference_issues": [],
            "label": {"sampled_price_return_pct": 0.0},
        }],
    }

    prepared = prepare_solana_training_rows(exported)
    assert prepared["eligible_row_count"] == 0
    assert prepared["exclusion_counts"] == {"snapshot_validation_issues": 1}


def test_snapshot_validation_row_issue_limit_only_caps_detail_not_counts():
    rows = [
        {
            "id": f"invalid-{index}", "capability": "solana.market.scan",
            "mint": f"mint-{index}", "pool_address": f"pool-{index}",
            "schema_version": 1, "fetched_at": f"2026-01-01T00:{index // 60:02d}:{index % 60:02d}Z",
            "payload": {"evidence": {}},
        }
        for index in range(501)
    ]
    report = validate_snapshot_rows(rows)
    assert report["rows_with_issues"] == 501
    assert len(report["row_issues"]) == 500
    assert report["row_issue_list_truncated"] is True
    complete = validate_snapshot_rows(rows, row_issue_limit=len(rows))
    assert len(complete["row_issues"]) == 501
    assert complete["row_issue_list_truncated"] is False



def test_snapshot_validator_reports_non_object_rows_without_crashing():
    rows = [
        None,
        ["not", "a", "snapshot"],
        "malformed",
        {
            "id": "valid-audit", "capability": "solana.token.audit",
            "mint": "MintA", "pool_address": "PoolA", "schema_version": 1,
            "fetched_at": "2026-01-01T00:00:00Z", "payload": {"passed": True},
        },
    ]
    report = validate_snapshot_rows(rows)
    assert report["rows_checked"] == 4
    assert report["issue_counts"]["malformed_snapshot_row"] == 3
    assert report["rows_with_issues"] == 3
    assert len(report["row_issues"]) == 3
    assert all(row["issues"] == ["malformed_snapshot_row"] for row in report["row_issues"])


def test_training_preparation_excludes_non_object_snapshot_rows():
    from eti.solana_dataset import prepare_solana_training_rows

    exported = {
        "schema": "eti.solana.research_dataset.v1",
        "snapshots": [None, ["malformed"]],
        "labels": [],
    }
    result = prepare_solana_training_rows(exported)
    assert result["eligible_row_count"] == 0
    assert result["excluded_label_count"] == 0
