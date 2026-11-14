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


def test_agent_task_completion_requires_current_unexpired_lease_token(tmp_path):
    store = MemoryStore(tmp_path / "lease-token.sqlite3")
    workspace = store.create_workspace("owner", "research")
    task = store.create_agent_task("owner", workspace["id"], "user", "test.echo", {})
    first = store.claim_next_agent_task(lease_seconds=5, now=100)
    with pytest.raises(ValueError, match="lease is invalid"):
        store.complete_claimed_agent_task(task["id"], "not-the-token", {"ok": True}, now=101)
    with pytest.raises(ValueError, match="lease is invalid"):
        store.complete_claimed_agent_task(task["id"], first["lease_token"], {"ok": True}, now=106)

    second = store.claim_next_agent_task(lease_seconds=5, now=106)
    assert second["attempt_count"] == 2
    with pytest.raises(ValueError, match="lease is invalid"):
        store.complete_claimed_agent_task(task["id"], first["lease_token"], {"ok": False}, now=107)
    done = store.complete_claimed_agent_task(task["id"], second["lease_token"], {"ok": True}, now=107)
    assert done["status"] == "completed"


def test_agent_worker_heartbeat_reports_online_and_offline_windows(tmp_path):
    store = MemoryStore(tmp_path / "worker-heartbeat.sqlite3")
    recorded = store.record_agent_worker_heartbeat(
        "worker-1", ["market.snapshot", "market.snapshot"], now=1000
    )
    assert recorded["capabilities"] == ["market.snapshot"]
    online = store.list_agent_workers(online_window_seconds=60, now=1059)
    assert len(online) == 1
    assert online[0]["online"] is True
    assert online[0]["capabilities"] == ["market.snapshot"]
    offline = store.list_agent_workers(online_window_seconds=60, now=1061)
    assert offline[0]["online"] is False


def test_autonomy_profile_schedules_owner_scoped_research_tasks_and_respects_interval(tmp_path):
    from datetime import datetime, timezone
    store = MemoryStore(tmp_path / "autonomy.sqlite3")
    workspace = store.create_workspace("owner-a", "research")
    profile = store.configure_autonomy(
        "owner-a", workspace["id"], True,
        venues=["mexc"], symbols=["BTC/USDT", "ETH/USDT"],
        timeframes=["1h"], interval_seconds=900,
    )
    assert profile["enabled"] is True
    first_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert store.enqueue_due_autonomy_tasks(first_time) == 6
    assert store.enqueue_due_autonomy_tasks(first_time) == 0
    assert len(store.list_agent_tasks("owner-a", workspace["id"], status="queued")) == 6
    assert store.list_agent_tasks("owner-b", workspace["id"]) == []
    assert store.enqueue_due_autonomy_tasks(
        datetime(2026, 1, 1, 0, 14, tzinfo=timezone.utc)
    ) == 0
    assert store.enqueue_due_autonomy_tasks(
        datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc)
    ) == 6


def test_autonomy_can_be_disabled_and_reconfigured(tmp_path):
    from datetime import datetime, timezone
    store = MemoryStore(tmp_path / "autonomy.sqlite3")
    workspace = store.create_workspace("owner-a", "research")
    store.configure_autonomy("owner-a", workspace["id"], True, symbols=["BTC/USDT"])
    store.configure_autonomy("owner-a", workspace["id"], False, symbols=["BTC/USDT"])
    assert store.enqueue_due_autonomy_tasks(datetime(2026, 1, 1, tzinfo=timezone.utc)) == 0
    assert store.get_autonomy_profile("owner-a")["enabled"] is False


def test_autonomy_profile_rejects_unsafe_or_invalid_scan_configuration(tmp_path):
    store = MemoryStore(tmp_path / "autonomy.sqlite3")
    workspace = store.create_workspace("owner-a", "research")
    with pytest.raises(ValueError, match="venues"):
        store.configure_autonomy("owner-a", workspace["id"], True, venues=["unknown"])
    with pytest.raises(ValueError, match="interval_seconds"):
        store.configure_autonomy("owner-a", workspace["id"], True, interval_seconds=10)


def test_autonomy_schedules_separate_daily_model_training_tasks(tmp_path):
    from datetime import datetime, timezone
    store = MemoryStore(tmp_path / "training-schedule.sqlite3")
    workspace = store.create_workspace("owner-a", "research")
    profile = store.configure_autonomy(
        "owner-a", workspace["id"], True,
        venues=["mexc"], symbols=["BTC/USDT"], timeframes=["1h"],
        interval_seconds=900, model_training_enabled=True,
        model_training_interval_seconds=86400,
    )
    assert profile["model_training_enabled"] is True
    assert profile["model_training_interval_seconds"] == 86400
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert store.enqueue_due_autonomy_tasks(start) == 15
    tasks = store.list_agent_tasks("owner-a", workspace["id"], status="queued")
    assert {task["target_capability"] for task in tasks} == {
        "market.snapshot", "market.discover", "model.train_research"
    }
    discoveries = [task for task in tasks if task["target_capability"] == "market.discover"]
    assert len(discoveries) == 6
    assert {(task["input"]["venue"], task["input"]["market_type"]) for task in discoveries} == {
        (venue, market_type) for venue in ("binance", "mexc")
        for market_type in ("spot", "swap", "future")
    }
    training = next(task for task in tasks if task["target_capability"] == "model.train_research")
    assert training["input"]["limit"] == 3000
    assert training["input"]["scheduled_by"] == "eti.autonomy.model_training"
    training_tasks = [task for task in tasks if task["target_capability"] == "model.train_research"]
    assert {task["input"]["asset_class"] for task in training_tasks} == {
        "forex", "equity", "commodity", "index"
    }
    assert len(training_tasks) == 4
    assert store.enqueue_due_autonomy_tasks(start) == 0
    assert store.enqueue_due_autonomy_tasks(
        datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc)
    ) == 5
    assert store.enqueue_due_autonomy_tasks(
        datetime(2026, 1, 2, tzinfo=timezone.utc)
    ) == 15
    all_tasks = store.list_agent_tasks("owner-a", workspace["id"])
    assert sum(t["target_capability"] == "model.train_research" for t in all_tasks) == 8
    assert sum(t["target_capability"] == "market.discover" for t in all_tasks) == 12


def test_model_training_schedule_is_bounded_and_disabled_by_default(tmp_path):
    store = MemoryStore(tmp_path / "training-bounds.sqlite3")
    workspace = store.create_workspace("owner-a", "research")
    profile = store.configure_autonomy("owner-a", workspace["id"], True)
    assert profile["model_training_enabled"] is False
    with pytest.raises(ValueError, match="at most 10"):
        store.configure_autonomy(
            "owner-a", workspace["id"], True,
            venues=["binance", "mexc"],
            symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"],
            timeframes=["1h"], market_types=["spot", "swap"], model_training_enabled=True,
        )


def test_exchange_credentials_are_owner_scoped_masked_and_audited(tmp_path):
    store = MemoryStore(tmp_path / "credentials.sqlite3")
    saved = store.upsert_exchange_credential(
        "owner-a", "binance", "encrypted-key", "encrypted-secret", "7890"
    )
    assert saved["venue"] == "binance"
    assert store.list_exchange_credentials("owner-a") == [{
        "venue": "binance",
        "masked_api_key": "••••7890",
        "created_at": saved["created_at"],
        "updated_at": saved["updated_at"],
    }]
    stored = store.get_exchange_credential("owner-a", "binance")
    assert stored["api_key_ciphertext"] == "encrypted-key"
    assert stored["api_secret_ciphertext"] == "encrypted-secret"
    assert store.get_exchange_credential("owner-b", "binance") is None
    assert store.list_exchange_credentials("owner-b") == []
    assert store.delete_exchange_credential("owner-a", "binance") is True
    assert store.delete_exchange_credential("owner-a", "binance") is False
    events = store.list_audit_events("owner-a")
    assert [event["event_type"] for event in events] == [
        "exchange.credentials.deleted", "exchange.credentials.saved"
    ]
    assert all("encrypted-secret" not in str(event["payload"]) for event in events)


def test_autonomy_schedules_spot_and_perpetual_swap_research(tmp_path):
    from datetime import datetime, timezone
    store = MemoryStore(tmp_path / "multi-market-autonomy.sqlite3")
    workspace = store.create_workspace("owner", "research")
    profile = store.configure_autonomy(
        "owner", workspace["id"], True, venues=["binance"],
        symbols=["BTC/USDT"], timeframes=["1h"],
        market_types=["spot", "swap"],
    )
    assert profile["market_types"] == ["spot", "swap"]
    assert store.enqueue_due_autonomy_tasks(datetime(2026, 1, 1, tzinfo=timezone.utc)) == 6
    tasks = store.list_agent_tasks("owner", workspace["id"], status="queued")
    crypto_inputs = {task["input"]["market_type"]: task["input"] for task in tasks
                     if task["input"].get("asset_class", "crypto") == "crypto"}
    assert crypto_inputs["spot"]["symbol"] == "BTC/USDT"
    assert crypto_inputs["swap"]["symbol"] == "BTC/USDT:USDT"
    assert {task["input"]["asset_class"] for task in tasks} == {
        "crypto", "forex", "equity", "commodity", "index"
    }



def test_autonomy_skips_dated_future_scan_without_exact_contract_symbol(tmp_path):
    from datetime import datetime, timezone
    store = MemoryStore(tmp_path / "future-symbol-guard.sqlite3")
    workspace = store.create_workspace("owner", "research")
    store.configure_autonomy(
        "owner", workspace["id"], True, venues=["binance"],
        symbols=["BTC/USDT"], timeframes=["1h"],
        market_types=["spot", "swap", "future"],
    )
    store.enqueue_due_autonomy_tasks(datetime(2026, 1, 1, tzinfo=timezone.utc))
    tasks = store.list_agent_tasks("owner", workspace["id"], status="queued")
    crypto_types = {task["input"]["market_type"] for task in tasks
                    if task["input"].get("asset_class") == "crypto"}
    assert crypto_types == {"spot", "swap"}


def test_solana_event_ledger_is_durable_idempotent_and_owner_scoped(tmp_path):
    path = tmp_path / "solana-events.sqlite3"
    store = MemoryStore(path)
    workspace = store.create_workspace("owner-a", "solana")
    payload = {"signature": "sig-001", "mint": "A" * 40, "slot": 12345}
    first = store.record_solana_research_event(
        "owner-a", workspace["id"], "pumpportal:sig-001",
        "pumpportal", "new_token", "A" * 40, payload, "2026-11-14T12:00:00Z",
    )
    duplicate = store.record_solana_research_event(
        "owner-a", workspace["id"], "pumpportal:sig-001",
        "pumpportal", "new_token", "A" * 40, payload, "2026-11-14T12:00:00Z",
    )
    assert first["id"] == duplicate["id"]
    assert duplicate["duplicate"] is True
    conflict = store.record_solana_research_event(
        "owner-a", workspace["id"], "pumpportal:sig-001",
        "pumpportal", "new_token", "A" * 40, {"signature": "sig-001", "slot": 99999},
        "2026-11-14T12:00:00Z",
    )
    assert conflict["duplicate"] is True
    assert conflict["payload_conflict"] is True
    reopened = MemoryStore(path)
    items = reopened.list_solana_research_events("owner-a", workspace["id"])
    assert len(items) == 1
    assert items[0]["payload"] == payload
    assert items[0]["payload_sha256"] == first["payload_sha256"]
    assert reopened.list_solana_research_events("owner-b", workspace["id"]) == []


def test_solana_event_ledger_rejects_invalid_event_data(tmp_path):
    store = MemoryStore(tmp_path / "solana-events.sqlite3")
    workspace = store.create_workspace("owner", "solana")
    with pytest.raises(ValueError, match="event_type"):
        store.record_solana_research_event(
            "owner", workspace["id"], "event-1", "pumpportal",
            "unknown", "A" * 40, {},
        )
    with pytest.raises(ValueError, match="source"):
        store.record_solana_research_event(
            "owner", workspace["id"], "event-2", "private-api",
            "new_token", "A" * 40, {},
        )
    with pytest.raises(LookupError, match="workspace not found"):
        store.record_solana_research_event(
            "owner", "not-your-workspace", "event-3", "pumpportal",
            "new_token", "A" * 40, {},
        )
