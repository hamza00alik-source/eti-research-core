import time

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


def test_worker_renews_lease_during_long_running_capability(tmp_path):
    store = MemoryStore(tmp_path / "long-running.sqlite3")
    workspace = _workspace(store)
    store.create_agent_task(
        "owner-1", workspace["id"], "test.requester", "test.slow",
        {"value": 9}, max_attempts=2,
    )
    registry = AgentCapabilityRegistry()

    def slow(payload):
        # Longer than the 5-second lease; the worker must renew it in flight.
        time.sleep(6.2)
        return {"value": payload["value"]}

    registry.register("test.slow", slow)
    worker = AgentWorker(store, registry=registry, lease_seconds=5)

    result = worker.run_once()

    assert result is not None
    assert result["status"] == "completed"
    assert result["attempt_count"] == 1
    assert result["result"] == {"value": 9}


def test_worker_keeps_quality_failed_model_as_unapproved_candidate(tmp_path):
    store = MemoryStore(tmp_path / "candidate-registration.sqlite3")
    workspace = _workspace(store)
    worker = AgentWorker(store, registry=AgentCapabilityRegistry())
    result = {
        "venue": "mexc", "asset_class": "crypto", "market_type": "spot",
        "symbol": "BTC/USDT", "timeframe": "1h",
        "training_data_ref": "ccxt:mexc:BTC/USDT:1h:3000:" + "a" * 64,
        "candidate_artifact": {
            "artifact_uri": "artifact://sha256/" + "a" * 64,
            "artifact_sha256": "a" * 64,
        },
        "input_schema": {"features": ["ret_1", "volatility"]},
        "evaluation": {"fold_count": 4, "mean_test_balanced_accuracy": 0.56},
        "evaluation_passed": False,
        "quality_checks": {"calibration_passed": True},
        "calibration_status": "calibrated",
    }
    registered = worker._register_model_candidate(result)
    model = store.get_model_version(registered["registry"]["model_id"],
                                    registered["registry"]["version"])
    assert model is not None
    assert model["lifecycle"] == "candidate"
    assert model["created_by"] == "eti.worker"
    assert model["evaluation_report"]["passed"] is False
    assert registered["registry"]["approval_required"] is True
    assert registered["lifecycle"] == "candidate"


def test_worker_logs_task_lifecycle_without_payload_or_result(tmp_path, capsys):
    store = MemoryStore(tmp_path / "worker-events.sqlite3")
    workspace = _workspace(store)
    store.create_agent_task(
        "owner-1", workspace["id"], "test.requester", "test.echo",
        {"secret_payload": "must-not-be-logged"}, max_attempts=1,
    )
    registry = AgentCapabilityRegistry()
    registry.register("test.echo", lambda payload: {"private_result": payload["secret_payload"]})
    worker = AgentWorker(store, registry=registry)

    result = worker.run_once()
    output = capsys.readouterr().out

    assert result["status"] == "completed"
    assert '"event":"agent.task.claimed"' in output
    assert '"event":"agent.task.execution_started"' in output
    assert '"event":"agent.task.execution_finished"' in output
    assert '"capability":"test.echo"' in output
    assert '"status":"completed"' in output
    assert "must-not-be-logged" not in output
    assert "private_result" not in output


def test_user_created_live_execution_task_is_rejected_by_worker(tmp_path):
    store = MemoryStore(tmp_path / "manual-live-task.sqlite3")
    workspace = _workspace(store)
    task = store.create_agent_task(
        "owner-1", workspace["id"], "user.requester", "live.tick",
        {"owner_id": "owner-1", "venue": "binance", "symbol": "BTC/USDT",
         "timeframe": "1h", "scheduled_by": "user.requester"},
        max_attempts=1,
    )
    worker = AgentWorker(store)
    result = worker.run_once()
    assert result["id"] == task["id"]
    assert result["status"] == "failed"
    assert result["result"]["error"]["type"] == "ValueError"


def test_worker_auto_approves_only_quality_passed_calibrated_artifact(tmp_path, monkeypatch):
    import eti.worker as worker_module

    store = MemoryStore(tmp_path / "auto-promotion.sqlite3")
    store.create_workspace("owner-1", "worker tests")
    worker = AgentWorker(store, registry=AgentCapabilityRegistry())

    class FakeEstimator:
        feature_names_in_ = ["ret_1", "volatility"]

    class FakeModel:
        estimator = FakeEstimator()
        def has_valid_calibration(self):
            return True

    class FakeArtifacts:
        def load_model(self, uri, digest):
            assert digest == "b" * 64
            return FakeModel()

    monkeypatch.setattr(worker_module, "ModelArtifactStore", FakeArtifacts)
    result = {
        "venue": "mexc", "asset_class": "crypto", "market_type": "spot",
        "symbol": "ETH/USDT", "timeframe": "1h",
        "training_data_ref": "ccxt:mexc:ETH/USDT:1h:3000:" + "b" * 64,
        "candidate_artifact": {
            "artifact_uri": "artifact://sha256/" + "b" * 64,
            "artifact_sha256": "b" * 64,
        },
        "input_schema": {"features": ["ret_1", "volatility"]},
        "evaluation": {"fold_count": 4, "mean_test_balanced_accuracy": 0.60},
        "evaluation_passed": True,
        "quality_checks": {"all_checks_passed": True},
        "calibration_status": "calibrated",
    }
    registered = worker._register_model_candidate(result)
    model = store.get_model_version(registered["registry"]["model_id"],
                                    registered["registry"]["version"])
    assert model["lifecycle"] == "approved"
    assert model["created_by"] == "eti.worker"
    assert model["approved_by"] == "eti.quality_gate"
    assert registered["registry"]["approval_required"] is False
    assert registered["registry"]["promotion_authority"] == "eti.quality_gate"


def test_solana_scan_completion_queues_read_only_audit(tmp_path):
    store = MemoryStore(tmp_path / "solana-followup.sqlite3")
    workspace = _workspace(store)
    store.create_agent_task(
        "owner-1", workspace["id"], "eti.solana.ingest", "solana.market.scan",
        {"mint": "A" * 40, "source_event_id": "event-1",
         "scheduled_by": "eti.solana.ingest"}, max_attempts=2,
    )
    registry = AgentCapabilityRegistry()
    registry.register("solana.market.scan", lambda payload: {
        "mint": payload["mint"], "verdict": "PASS",
    })
    registry.register("solana.token.audit", lambda payload: {
        "mint": payload["mint"], "passed": False, "blocked": True,
    })
    worker = AgentWorker(store, registry=registry)

    scan = worker.run_once()
    assert scan["status"] == "completed"
    queued = store.list_agent_tasks("owner-1", workspace["id"], status="queued")
    assert len(queued) == 1
    assert queued[0]["target_capability"] == "solana.token.audit"
    assert queued[0]["input"]["source_scan_task_id"] == scan["id"]

    audit = worker.run_once()
    assert audit["status"] == "completed"
    assert audit["result"]["blocked"] is True
    assert store.list_agent_tasks("owner-1", workspace["id"], status="queued") == []


def test_solana_rejected_scan_does_not_queue_audit(tmp_path):
    store = MemoryStore(tmp_path / "solana-rejected.sqlite3")
    workspace = _workspace(store)
    store.create_agent_task(
        "owner-1", workspace["id"], "eti.solana.ingest", "solana.market.scan",
        {"mint": "B" * 40, "scheduled_by": "eti.solana.ingest"}, max_attempts=2,
    )
    registry = AgentCapabilityRegistry()
    registry.register("solana.market.scan", lambda payload: {
        "mint": payload["mint"], "verdict": "REJECT",
    })
    worker = AgentWorker(store, registry=registry)
    result = worker.run_once()
    assert result["status"] == "completed"
    assert store.list_agent_tasks("owner-1", workspace["id"], status="queued") == []
