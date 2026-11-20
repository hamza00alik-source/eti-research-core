import numpy as np
import sqlite3
import pytest

from eti.storage import MemoryStore


def _digest(char):
    return char * 64


def _timestamps(start="2026-01-01T00:00:00Z"):
    import pandas as pd
    return [item.isoformat().replace("+00:00", "Z") for item in
            pd.date_range(start, periods=8, freq="h", tz="UTC")]


def _register(store, workspace_id, trial_id, *, config="a", evaluation="b", cost="c",
               cost_digest=None, source_evidence=None):
    return store.register_research_trial(
        "owner-a", workspace_id, "family-2026-01", trial_id,
        "momentum", _digest(config), _digest("d"), _digest(evaluation),
        cost_digest or _digest(cost), "buy-and-hold-btc-usdt-1h", source_evidence,
    )


def test_trial_family_persists_every_attempt_and_builds_aligned_matrix(tmp_path):
    path = tmp_path / "trials.sqlite3"
    store = MemoryStore(path)
    workspace = store.create_workspace("owner-a", "research")
    workspace_id = workspace["id"]

    first = _register(store, workspace_id, "trial-001", config="a")
    second = _register(store, workspace_id, "trial-002", config="e")
    failed = _register(store, workspace_id, "trial-003", config="f")
    not_started = _register(store, workspace_id, "trial-004", config="1")
    assert first["status"] == second["status"] == "registered"

    store.start_research_trial("owner-a", workspace_id, "trial-001")
    store.start_research_trial("owner-a", workspace_id, "trial-002")
    store.start_research_trial("owner-a", workspace_id, "trial-003")
    times = _timestamps()
    store.complete_research_trial(
        "owner-a", workspace_id, "trial-001", times,
        [0.001, 0.002, -0.001, 0.003, 0.001, 0.0, 0.002, 0.001],
    )
    store.complete_research_trial(
        "owner-a", workspace_id, "trial-002", times,
        [0.0, 0.001, 0.001, 0.002, -0.001, 0.001, 0.003, 0.002],
    )
    store.fail_research_trial("owner-a", workspace_id, "trial-003", "optimizer diverged")

    reopened = MemoryStore(path)
    matrix = reopened.get_research_trial_matrix("owner-a", workspace_id, "family-2026-01")
    assert matrix["registered_trial_count"] == 4
    assert matrix["attempted_trial_count"] == 3
    assert matrix["completed_trial_count"] == 2
    assert matrix["failed_trial_count"] == 1
    assert matrix["not_started_trial_count"] == 1
    assert matrix["running_trial_count"] == 0
    assert matrix["trial_ids"] == ["trial-001", "trial-002"]
    assert len(matrix["net_return_differentials"]) == 8
    assert all(len(row) == 2 for row in matrix["net_return_differentials"])
    assert len(set(matrix["returns_sha256"])) == 2
    assert reopened.get_research_trial("owner-a", workspace_id, "trial-004")["status"] == "registered"


def test_trial_family_persists_and_verifies_source_evidence_manifest(tmp_path):
    import hashlib
    import json

    path = tmp_path / "source-evidence.sqlite3"
    store = MemoryStore(path)
    workspace = store.create_workspace("owner-a", "research")
    workspace_id = workspace["id"]
    cost_model = {"fee_bps": 6.0, "slippage_bps": 2.0}
    cost_digest = hashlib.sha256(json.dumps(
        cost_model, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    snapshot_digest = _digest("1")
    archive_digest = _digest("2")
    archive_receipt = {
        "schema_version": "eti.ohlcv_archive_receipt.v1",
        "sha256": archive_digest,
        "source_snapshot_sha256": snapshot_digest,
        "relative_path": f"{archive_digest[:2]}/{archive_digest}.json",
    }
    manifest = {
        "schema_version": "eti.research_source_evidence.v1",
        "capture_method": "MarketDataHub.fetch",
        "source_snapshot_sha256": snapshot_digest,
        "raw_data_archive": archive_receipt,
        "source_provenance": {
            "source_snapshot_sha256": snapshot_digest,
            "raw_data_archive": archive_receipt,
        },
        "cost_model": cost_model,
        "cost_model_sha256": cost_digest,
        "training_data_sha256": _digest("d"),
        "evaluation_data_sha256": _digest("b"),
    }
    for trial_id, config, value in (("trial-001", "a", 0.001), ("trial-002", "e", 0.002)):
        _register(store, workspace_id, trial_id, config=config, cost_digest=cost_digest,
                   source_evidence=manifest)
        store.start_research_trial("owner-a", workspace_id, trial_id)
        store.complete_research_trial(
            "owner-a", workspace_id, trial_id, _timestamps(), [value] * 8,
        )

    reopened = MemoryStore(path)
    matrix = reopened.get_research_trial_matrix("owner-a", workspace_id, "family-2026-01")
    assert matrix["source_evidence"] == manifest
    expected = hashlib.sha256(json.dumps(
        manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    assert matrix["source_evidence_sha256"] == expected

    with pytest.raises(ValueError, match="source evidence are immutable"):
        _register(reopened, workspace_id, "trial-003", config="f",
                  source_evidence={**manifest, "source_snapshot_sha256": _digest("2")})

    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE research_trial_families SET source_evidence_json=? WHERE family_id=?",
            ('{"tampered":true}', "family-2026-01"),
        )
    with pytest.raises(ValueError, match="source evidence manifest digest mismatch"):
        reopened.get_research_trial_matrix("owner-a", workspace_id, "family-2026-01")


def test_runner_capture_manifest_rejects_cross_binding_mismatch(tmp_path):
    import hashlib
    import json

    store = MemoryStore(tmp_path / "cross-binding.sqlite3")
    workspace = store.create_workspace("owner-a", "research")
    cost_model = {"fee_bps": 6.0, "slippage_bps": 2.0}
    cost_digest = hashlib.sha256(json.dumps(
        cost_model, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    snapshot_digest = _digest("1")
    archive_digest = _digest("2")
    archive_receipt = {
        "schema_version": "eti.ohlcv_archive_receipt.v1",
        "sha256": archive_digest,
        "source_snapshot_sha256": snapshot_digest,
        "relative_path": f"{archive_digest[:2]}/{archive_digest}.json",
    }
    manifest = {
        "schema_version": "eti.research_source_evidence.v1",
        "capture_method": "MarketDataHub.fetch",
        "source_snapshot_sha256": snapshot_digest,
        "raw_data_archive": archive_receipt,
        "source_provenance": {
            "source_snapshot_sha256": snapshot_digest,
            "raw_data_archive": archive_receipt,
        },
        "cost_model": cost_model,
        "cost_model_sha256": cost_digest,
        "training_data_sha256": _digest("d"),
        "evaluation_data_sha256": _digest("b"),
    }
    for trial_id, value in (("trial-001", 0.001), ("trial-002", 0.002)):
        _register(store, workspace["id"], trial_id, cost_digest=cost_digest,
                   source_evidence=manifest)
        store.start_research_trial("owner-a", workspace["id"], trial_id)
        store.complete_research_trial(
            "owner-a", workspace["id"], trial_id, _timestamps(), [value] * 8,
        )
    # A self-consistent outer digest cannot hide a contradictory archive receipt.
    manifest["raw_data_archive"]["source_snapshot_sha256"] = _digest("3")
    encoded = json.dumps(
        manifest, ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    )
    with sqlite3.connect(store.database_path) as conn:
        conn.execute(
            "UPDATE research_trial_families SET source_evidence_json=?,source_evidence_sha256=?",
            (encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()),
        )
    with pytest.raises(ValueError, match="raw-data archive receipt does not match"):
        store.get_research_trial_matrix("owner-a", workspace["id"], "family-2026-01")

    # Then test a separate internally contradictory transaction-cost receipt.
    manifest["raw_data_archive"]["source_snapshot_sha256"] = snapshot_digest
    manifest["cost_model_sha256"] = _digest("e")
    encoded = json.dumps(
        manifest, ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    )
    with sqlite3.connect(store.database_path) as conn:
        conn.execute(
            "UPDATE research_trial_families SET source_evidence_json=?,source_evidence_sha256=?",
            (encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()),
        )
    with pytest.raises(ValueError, match="transaction-cost model fingerprint mismatch"):
        store.get_research_trial_matrix("owner-a", workspace["id"], "family-2026-01")


def test_trial_family_identity_and_attempt_ids_are_immutable(tmp_path):
    store = MemoryStore(tmp_path / "trials.sqlite3")
    workspace = store.create_workspace("owner-a", "research")
    workspace_id = workspace["id"]
    _register(store, workspace_id, "trial-001")
    with pytest.raises(ValueError, match="immutable"):
        _register(store, workspace_id, "trial-002", cost="e")
    with pytest.raises(ValueError, match="already exists"):
        _register(store, workspace_id, "trial-001", config="f")
    assert store.get_research_trial("owner-a", workspace_id, "trial-002") is None


def test_trial_lifecycle_is_guarded_and_results_are_write_once(tmp_path):
    store = MemoryStore(tmp_path / "trials.sqlite3")
    workspace = store.create_workspace("owner-a", "research")
    workspace_id = workspace["id"]
    _register(store, workspace_id, "trial-001")
    with pytest.raises(ValueError, match="only a running"):
        store.complete_research_trial("owner-a", workspace_id, "trial-001", _timestamps(), [0.0] * 8)
    store.start_research_trial("owner-a", workspace_id, "trial-001")
    with pytest.raises(ValueError, match="only a registered"):
        store.start_research_trial("owner-a", workspace_id, "trial-001")
    store.complete_research_trial("owner-a", workspace_id, "trial-001", _timestamps(), [0.001] * 8)
    with pytest.raises(ValueError, match="only a running"):
        store.complete_research_trial("owner-a", workspace_id, "trial-001", _timestamps(), [0.002] * 8)
    with pytest.raises(ValueError, match="only a running"):
        store.fail_research_trial("owner-a", workspace_id, "trial-001", "late failure")


def test_trial_series_rejects_bad_timestamps_lengths_and_non_finite_values(tmp_path):
    store = MemoryStore(tmp_path / "trials.sqlite3")
    workspace = store.create_workspace("owner-a", "research")
    workspace_id = workspace["id"]
    _register(store, workspace_id, "trial-001")
    store.start_research_trial("owner-a", workspace_id, "trial-001")
    valid = _timestamps()

    with pytest.raises(ValueError, match="timezone"):
        store.complete_research_trial("owner-a", workspace_id, "trial-001",
                                      [item.replace("Z", "") for item in valid], [0.0] * 8)
    with pytest.raises(ValueError, match="strictly increasing"):
        store.complete_research_trial("owner-a", workspace_id, "trial-001",
                                      [valid[0], valid[0], *valid[2:]], [0.0] * 8)
    with pytest.raises(ValueError, match="strictly increasing"):
        store.complete_research_trial("owner-a", workspace_id, "trial-001",
                                      [valid[1], valid[0], *valid[2:]], [0.0] * 8)
    with pytest.raises(ValueError, match="paired observations"):
        store.complete_research_trial("owner-a", workspace_id, "trial-001", valid[:-1], [0.0] * 7)
    with pytest.raises(ValueError, match="finite"):
        store.complete_research_trial("owner-a", workspace_id, "trial-001", valid,
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan])
    with pytest.raises(ValueError, match="finite"):
        store.complete_research_trial("owner-a", workspace_id, "trial-001", valid,
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True])
    assert store.get_research_trial("owner-a", workspace_id, "trial-001")["status"] == "running"


def test_trial_matrix_fails_closed_on_misaligned_observation_timestamps(tmp_path):
    store = MemoryStore(tmp_path / "trials.sqlite3")
    workspace = store.create_workspace("owner-a", "research")
    workspace_id = workspace["id"]
    _register(store, workspace_id, "trial-001")
    _register(store, workspace_id, "trial-002", config="e")
    for trial_id in ("trial-001", "trial-002"):
        store.start_research_trial("owner-a", workspace_id, trial_id)
    store.complete_research_trial("owner-a", workspace_id, "trial-001", _timestamps(), [0.001] * 8)
    store.complete_research_trial(
        "owner-a", workspace_id, "trial-002",
        _timestamps("2026-01-02T00:00:00Z"), [0.002] * 8,
    )
    with pytest.raises(ValueError, match="not timestamp-aligned"):
        store.get_research_trial_matrix("owner-a", workspace_id, "family-2026-01")


def test_trial_ledger_is_owner_scoped_and_requires_completed_family(tmp_path):
    store = MemoryStore(tmp_path / "trials.sqlite3")
    workspace = store.create_workspace("owner-a", "research")
    workspace_id = workspace["id"]
    _register(store, workspace_id, "trial-001")
    assert store.get_research_trial("owner-b", workspace_id, "trial-001") is None
    with pytest.raises(LookupError, match="workspace not found"):
        store.register_research_trial(
            "owner-b", workspace_id, "other-family", "trial-002", "trend",
            _digest("a"), _digest("d"), _digest("b"), _digest("c"), "benchmark",
        )
    with pytest.raises(ValueError, match="at least two completed"):
        store.get_research_trial_matrix("owner-a", workspace_id, "family-2026-01")

def test_trial_matrix_detects_return_series_tampering(tmp_path):
    path = tmp_path / "tamper.sqlite3"
    store = MemoryStore(path)
    workspace = store.create_workspace("owner-a", "research")
    workspace_id = workspace["id"]
    for trial_id, config, value in (("trial-001", "a", 0.001), ("trial-002", "e", 0.002)):
        _register(store, workspace_id, trial_id, config=config)
        store.start_research_trial("owner-a", workspace_id, trial_id)
        store.complete_research_trial(
            "owner-a", workspace_id, trial_id, _timestamps(), [value] * 8,
        )

    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE research_trials SET net_return_differentials_json=? WHERE trial_id=?",
            ("[0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]", "trial-001"),
        )
    with pytest.raises(ValueError, match="digest mismatch"):
        store.get_research_trial_matrix("owner-a", workspace_id, "family-2026-01")


def test_legacy_trial_family_migration_backfills_empty_manifest_digest(tmp_path):
    path = tmp_path / "legacy-trials.sqlite3"
    store = MemoryStore(path)
    workspace = store.create_workspace("owner-a", "research")
    workspace_id = workspace["id"]
    _register(store, workspace_id, "trial-001")
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE research_trial_families SET source_evidence_sha256='' WHERE family_id=?",
            ("family-2026-01",),
        )

    migrated = MemoryStore(path)
    _register(migrated, workspace_id, "trial-002", config="e")
    with sqlite3.connect(path) as conn:
        digest = conn.execute(
            "SELECT source_evidence_sha256 FROM research_trial_families WHERE family_id=?",
            ("family-2026-01",),
        ).fetchone()[0]
    import hashlib
    assert digest == hashlib.sha256(b"{}").hexdigest()
