import pytest

from eti.agents import AgentCapabilityRegistry, default_capabilities
from eti.storage import MemoryStore
from eti.worker import AgentWorker


def _workspace(store):
    return store.create_workspace("owner-1", "worker tests")


def test_worker_claims_executes_and_completes_allowlisted_task(tmp_path):
    store = MemoryStore(tmp_path / "worker.sqlite3")
    workspace = _workspace(store)
    store.create_agent_task(
        "owner-1", workspace["id"], "test.requester", "test.echo",
        {"value": "safe data"}, max_attempts=2,
    )
    registry = AgentCapabilityRegistry()
    registry.register("test.echo", lambda payload: {"echo": payload["value"]})

    worker = AgentWorker(store, registry=registry)
    result = worker.run_once()

    assert result is not None
    assert result["status"] == "completed"
    assert result["attempt_count"] == 1
    assert result["result"] == {"echo": "safe data"}
    messages = store.list_agent_messages("owner-1", workspace["id"])
    assert len(messages) == 1
    assert messages[0]["message_type"] == "task.completed"


def test_worker_retries_transient_failure_then_completes(tmp_path):
    store = MemoryStore(tmp_path / "retry.sqlite3")
    workspace = _workspace(store)
    task = store.create_agent_task(
        "owner-1", workspace["id"], "test.requester", "test.flaky",
        {"value": 7}, max_attempts=2,
    )
    calls = {"count": 0}

    def flaky(payload):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary provider outage")
        return {"value": payload["value"] * 2}

    registry = AgentCapabilityRegistry()
    registry.register("test.flaky", flaky)
    worker = AgentWorker(store, registry=registry)

    first = worker.run_once()
    assert first["status"] == "queued"
    assert first["attempt_count"] == 1
    second = worker.run_once()
    assert second["status"] == "completed"
    assert second["attempt_count"] == 2
    assert second["result"] == {"value": 14}


def test_unknown_capability_fails_without_retry(tmp_path):
    store = MemoryStore(tmp_path / "unsupported.sqlite3")
    workspace = _workspace(store)
    task = store.create_agent_task(
        "owner-1", workspace["id"], "test.requester", "shell.execute",
        {"command": "echo unsafe"}, max_attempts=4,
    )
    worker = AgentWorker(store, registry=default_capabilities())

    result = worker.run_once()

    assert result["id"] == task["id"]
    assert result["status"] == "failed"
    assert result["attempt_count"] == 1
    assert result["result"]["error"]["type"] == "LookupError"


def test_expired_lease_requeues_and_stale_worker_cannot_complete(tmp_path):
    store = MemoryStore(tmp_path / "lease.sqlite3")
    workspace = _workspace(store)
    task = store.create_agent_task(
        "owner-1", workspace["id"], "test.requester", "test.echo",
        {"value": 1}, max_attempts=2,
    )
    first = store.claim_next_agent_task(lease_seconds=5, now=1000)
    assert first["id"] == task["id"]
    assert first["attempt_count"] == 1

    second = store.claim_next_agent_task(lease_seconds=5, now=1006)
    assert second["id"] == task["id"]
    assert second["attempt_count"] == 2
    assert second["lease_token"] != first["lease_token"]

    with pytest.raises(ValueError, match="lease is invalid or expired"):
        store.complete_claimed_agent_task(
            task["id"], first["lease_token"], {"stale": True}, now=1007
        )
    completed = store.complete_claimed_agent_task(
        task["id"], second["lease_token"], {"ok": True}, now=1007
    )
    assert completed["status"] == "completed"


def test_exhausted_expired_lease_becomes_failed(tmp_path):
    store = MemoryStore(tmp_path / "exhausted.sqlite3")
    workspace = _workspace(store)
    task = store.create_agent_task(
        "owner-1", workspace["id"], "test.requester", "test.echo",
        {"value": 1}, max_attempts=1,
    )
    claimed = store.claim_next_agent_task(lease_seconds=5, now=1000)
    assert claimed["id"] == task["id"]

    assert store.claim_next_agent_task(lease_seconds=5, now=1006) is None
    current = store.get_agent_task("owner-1", workspace["id"], task["id"])
    assert current["status"] == "failed"
    assert current["result"]["error"] == "worker lease expired"


def test_worker_registers_model_training_result_as_unapproved_candidate(tmp_path):
    store = MemoryStore(tmp_path / "candidate-registration.sqlite3")
    workspace = _workspace(store)
    worker = AgentWorker(store, registry=AgentCapabilityRegistry())
    result = {
        "venue": "mexc", "symbol": "BTC/USDT", "timeframe": "1h",
        "training_data_ref": "ccxt:mexc:BTC/USDT:1h:3000:" + "a" * 64,
        "candidate_artifact": {
            "artifact_uri": "artifact://sha256/" + "a" * 64,
            "artifact_sha256": "a" * 64,
        },
        "input_schema": {"features": ["ret_1", "volatility"]},
        "evaluation": {"fold_count": 4, "mean_test_balanced_accuracy": 0.56},
        "evaluation_passed": True,
        "quality_checks": {"calibration_passed": True},
        "calibration_status": "calibrated",
    }
    registered = worker._register_model_candidate(result)
    model = store.get_model_version(registered["registry"]["model_id"],
                                    registered["registry"]["version"])
    assert model is not None
    assert model["lifecycle"] == "candidate"
    assert model["created_by"] == "eti.worker"
    assert model["evaluation_report"]["passed"] is True
    assert registered["registry"]["approval_required"] is True
    assert registered["lifecycle"] == "candidate"
