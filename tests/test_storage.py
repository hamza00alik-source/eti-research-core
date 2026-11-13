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
