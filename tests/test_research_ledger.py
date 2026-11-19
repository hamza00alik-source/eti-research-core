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


def _register(store, workspace_id, trial_id, *, config="a", evaluation="b", cost="c"):
    return store.register_research_trial(
        "owner-a", workspace_id, "family-2026-01", trial_id,
        "momentum", _digest(config), _digest("d"), _digest(evaluation),
        _digest(cost), "buy-and-hold-btc-usdt-1h",
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
