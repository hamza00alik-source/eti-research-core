from datetime import datetime, timezone

from eti.solana_dataset import validate_snapshot_rows
from eti.storage import MemoryStore


def test_dataset_validator_flags_missing_negative_zero_and_duplicate_values():
    rows = [
        {
            "id": "one", "mint": "A" * 40, "pool_address": "PoolA",
            "fetched_at": "2026-01-01T00:00:00Z",
            "payload": {"evidence": {
                "price_usd": 0, "liquidity_usd": -1, "volume_h24_usd": 2,
                "txns_h1_buys": 1, "txns_h1_sells": 0,
            }},
        },
        {
            "id": "two", "mint": "A" * 40, "pool_address": "PoolA",
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
