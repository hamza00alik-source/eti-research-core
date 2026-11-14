from datetime import datetime, timezone

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
