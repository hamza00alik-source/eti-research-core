import json
import math
import pytest

from eti.storage import MemoryStore


def test_workspace_memory_persists_and_is_scoped_to_owner(tmp_path):
    path = tmp_path / "eti.sqlite3"
    store = MemoryStore(path)
    workspace = store.create_workspace("account-a", "research")
    store.put_memory("account-a", workspace["id"], "preferences", "markets", ["crypto", "gold"])

    reopened = MemoryStore(path)
    assert reopened.get_memory("account-a", workspace["id"], "preferences", "markets")["value"] == [
        "crypto", "gold"
    ]
    assert reopened.get_memory("account-b", workspace["id"], "preferences", "markets") is None
    assert reopened.list_workspaces("account-b") == []
    assert reopened.list_memory("account-b", workspace["id"]) == []


def test_cannot_write_to_another_accounts_workspace(tmp_path):
    store = MemoryStore(tmp_path / "eti.sqlite3")
    workspace = store.create_workspace("owner-a", "private")
    with pytest.raises(LookupError, match="workspace not found"):
        store.put_memory("owner-b", workspace["id"], "notes", "secret", "nope")
    with pytest.raises(LookupError, match="workspace not found"):
        store.append_audit_event("owner-b", "memory.write", {}, workspace["id"])


def test_memory_upsert_updates_value_without_duplicate_keys(tmp_path):
    store = MemoryStore(tmp_path / "eti.sqlite3")
    workspace = store.create_workspace("owner", "main")
    first = store.put_memory("owner", workspace["id"], "agent", "last_task", {"state": "queued"})
    second = store.put_memory("owner", workspace["id"], "agent", "last_task", {"state": "done"})
    items = store.list_memory("owner", workspace["id"])
    assert len(items) == 1
    assert second["id"] == first["id"]
    assert items[0]["value"] == {"state": "done"}


def test_memory_rejects_invalid_json_and_oversized_payloads(tmp_path):
    store = MemoryStore(tmp_path / "eti.sqlite3")
    workspace = store.create_workspace("owner", "main")
    with pytest.raises(ValueError):
        store.put_memory("owner", workspace["id"], "n", "nan", math.nan)
    with pytest.raises(ValueError, match="exceeds"):
        store.put_memory("owner", workspace["id"], "n", "large", "x" * 70000)


def test_audit_events_are_owner_scoped_and_append_only_api(tmp_path):
    store = MemoryStore(tmp_path / "eti.sqlite3")
    workspace = store.create_workspace("owner-a", "main")
    store.append_audit_event("owner-a", "agent.message.created", {"task_id": "t1"}, workspace["id"])
    store.append_audit_event("owner-b", "agent.message.created", {"task_id": "t2"})
    events = store.list_audit_events("owner-a")
    assert len(events) == 1
    assert events[0]["event_type"] == "agent.message.created"
    assert events[0]["payload"] == {"task_id": "t1"}


def test_workspace_name_is_unique_per_owner_but_not_across_owners(tmp_path):
    store = MemoryStore(tmp_path / "eti.sqlite3")
    store.create_workspace("owner-a", "default")
    store.create_workspace("owner-b", "default")
    with pytest.raises(Exception):
        store.create_workspace("owner-a", "default")


def test_store_validates_limits_and_identity(tmp_path):
    store = MemoryStore(tmp_path / "eti.sqlite3")
    with pytest.raises(ValueError):
        store.list_audit_events("owner", limit=0)
    with pytest.raises(ValueError):
        store.create_workspace("", "default")

def test_store_requires_file_backed_database():
    with pytest.raises(ValueError, match="file-backed"):
        MemoryStore(":memory:")


def test_agent_tasks_are_durable_and_have_guarded_state_transitions(tmp_path):
    path = tmp_path / "agents.sqlite3"
    store = MemoryStore(path)
    workspace = store.create_workspace("owner-a", "research")
    task = store.create_agent_task(
        "owner-a", workspace["id"], "orchestrator", "market.research",
        {"symbol": "BTC/USDT", "timeframe": "1h"}
    )
    assert task["status"] == "queued"
    reopened = MemoryStore(path)
    assert reopened.get_agent_task("owner-a", workspace["id"], task["id"])["input"]["symbol"] == "BTC/USDT"
    running = reopened.update_agent_task("owner-a", workspace["id"], task["id"], "running")
    assert running["status"] == "running"
    done = reopened.update_agent_task(
        "owner-a", workspace["id"], task["id"], "completed", {"finding": "test"}
    )
    assert done["result"] == {"finding": "test"}
    with pytest.raises(ValueError, match="invalid task transition"):
        reopened.update_agent_task("owner-a", workspace["id"], task["id"], "running")


def test_agent_tasks_and_messages_are_isolated_by_owner(tmp_path):
    store = MemoryStore(tmp_path / "agents.sqlite3")
    workspace = store.create_workspace("owner-a", "research")
    task = store.create_agent_task("owner-a", workspace["id"], "planner", "risk.check", {})
    assert store.get_agent_task("owner-b", workspace["id"], task["id"]) is None
    with pytest.raises(LookupError, match="workspace not found"):
        store.create_agent_task("owner-b", workspace["id"], "planner", "risk.check", {})
    with pytest.raises(LookupError, match="task not found"):
        store.send_agent_message(
            "owner-a", workspace["id"], "planner", "risk", "request", {}, "not-a-task"
        )


def test_agent_messages_are_persisted_as_data_and_task_scoped(tmp_path):
    path = tmp_path / "agents.sqlite3"
    store = MemoryStore(path)
    workspace = store.create_workspace("owner-a", "research")
    task = store.create_agent_task("owner-a", workspace["id"], "planner", "risk.check", {})
    message = store.send_agent_message(
        "owner-a", workspace["id"], "planner", "risk", "risk.request",
        {"max_leverage": 1, "note": "assess only"}, task["id"]
    )
    reopened = MemoryStore(path)
    messages = reopened.list_agent_messages("owner-a", workspace["id"], task["id"])
    assert len(messages) == 1
    assert messages[0]["id"] == message["id"]
    assert messages[0]["payload"]["max_leverage"] == 1
    assert reopened.list_agent_messages("owner-b", workspace["id"]) == []


def test_completed_agent_task_requires_result(tmp_path):
    store = MemoryStore(tmp_path / "agents.sqlite3")
    workspace = store.create_workspace("owner", "research")
    task = store.create_agent_task("owner", workspace["id"], "planner", "data.check", {})
    store.update_agent_task("owner", workspace["id"], task["id"], "running")
    with pytest.raises(ValueError, match="require a result"):
        store.update_agent_task("owner", workspace["id"], task["id"], "completed")


def test_model_registry_requires_checksum_and_registers_immutable_candidate(tmp_path):
    store = MemoryStore(tmp_path / "models.sqlite3")
    args = ("eti-core", "0.1.0", "artifact://eti-core/0.1.0", "a" * 64,
            {"features": ["close", "volume"]}, ["crypto_spot"], "dataset:train-001",
            {"passed": True, "balanced_accuracy": 0.61}, "calibrated", "trainer")
    with pytest.raises(ValueError, match="SHA-256"):
        store.register_model_version(*args[:3], "bad", *args[4:])
    model = store.register_model_version(*args)
    assert model["lifecycle"] == "candidate"
    assert model["artifact_sha256"] == "a" * 64
    with pytest.raises(Exception):
        store.register_model_version(*args)


def test_model_promotion_requires_pass_calibration_and_independent_approval(tmp_path):
    store = MemoryStore(tmp_path / "models.sqlite3")
    common = ("eti-core", "1.0", "artifact://eti-core/1.0", "b" * 64,
              {"features": ["close"]}, ["crypto_spot"], "dataset:1")
    store.register_model_version(*common, {"passed": False}, "calibrated", "trainer")
    with pytest.raises(ValueError, match="must explicitly pass"):
        store.approve_model_version("eti-core", "1.0", "reviewer")

    store.register_model_version(
        "eti-core", "1.1", "artifact://eti-core/1.1", "c" * 64,
        {"features": ["close"]}, ["crypto_spot"], "dataset:2",
        {"passed": True}, "unknown", "trainer"
    )
    with pytest.raises(ValueError, match="calibrated"):
        store.approve_model_version("eti-core", "1.1", "reviewer")

    store.register_model_version(
        "eti-core", "1.2", "artifact://eti-core/1.2", "d" * 64,
        {"features": ["close"]}, ["crypto_spot"], "dataset:3",
        {"passed": True}, "calibrated", "trainer"
    )
    with pytest.raises(ValueError, match="cannot approve their own"):
        store.approve_model_version("eti-core", "1.2", "trainer")
    approved = store.approve_model_version("eti-core", "1.2", "independent-reviewer")
    assert approved["lifecycle"] == "approved"
    assert approved["approved_by"] == "independent-reviewer"


def test_model_retirement_is_guarded_and_registry_is_queryable(tmp_path):
    store = MemoryStore(tmp_path / "models.sqlite3")
    store.register_model_version(
        "eti-core", "1.0", "artifact://eti-core/1.0", "e" * 64,
        {"features": ["close"]}, ["crypto_spot"], "dataset:1",
        {"passed": True}, "calibrated", "trainer"
    )
    with pytest.raises(ValueError, match="approved model"):
        store.retire_model_version("eti-core", "1.0")
    store.approve_model_version("eti-core", "1.0", "reviewer")
    retired = store.retire_model_version("eti-core", "1.0")
    assert retired["lifecycle"] == "retired"
    assert store.list_model_versions("retired")[0]["version"] == "1.0"


def test_agent_worker_leases_are_atomic_and_recover_after_expiry(tmp_path):
    store = MemoryStore(tmp_path / "worker.sqlite3")
    workspace = store.create_workspace("owner", "research")
    task = store.create_agent_task(
        "owner", workspace["id"], "user", "test.echo", {"value": 7}, max_attempts=2
    )

    first = store.claim_next_agent_task(lease_seconds=5, now=100)
    assert first["id"] == task["id"]
    assert first["status"] == "running"
    assert first["attempt_count"] == 1
    assert first["lease_expires_at"].startswith("1970-01-01T00:01:45")

    second = store.claim_next_agent_task(lease_seconds=5, now=106)
    assert second["id"] == task["id"]
    assert second["attempt_count"] == 2
    assert second["status"] == "running"

    assert store.claim_next_agent_task(lease_seconds=5, now=112) is None
    final = store.get_agent_task("owner", workspace["id"], task["id"])
    assert final["status"] == "failed"
    assert final["attempt_count"] == 2
    assert final["result"]["error"] == "worker lease expired"


def test_agent_task_claims_are_single_owner_under_concurrency(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    store = MemoryStore(tmp_path / "worker-concurrent.sqlite3")
    workspace = store.create_workspace("owner", "research")
    tasks = [
        store.create_agent_task("owner", workspace["id"], "user", "test.echo", {"n": n})
        for n in range(2)
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed = list(pool.map(lambda _: store.claim_next_agent_task(), range(2)))
    assert {item["id"] for item in claimed if item is not None} == {t["id"] for t in tasks}


def test_queued_task_cancellation_is_atomic_and_cannot_cancel_running_work(tmp_path):
    store = MemoryStore(tmp_path / "cancel-task.sqlite3")
    workspace = store.create_workspace("owner", "research")
    queued = store.create_agent_task("owner", workspace["id"], "user", "test.echo", {})
    cancelled = store.cancel_agent_task("owner", workspace["id"], queued["id"])
    assert cancelled["status"] == "cancelled"

    running = store.create_agent_task("owner", workspace["id"], "user", "test.echo", {})
    store.claim_next_agent_task(lease_seconds=60)
    with pytest.raises(ValueError, match="only queued tasks"):
        store.cancel_agent_task("owner", workspace["id"], running["id"])
