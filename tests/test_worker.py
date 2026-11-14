import sqlite3
import time
from datetime import datetime, timezone

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
        "observed_at_ms": 1760000000000,
    })
    registry.register("solana.token.audit", lambda payload: {
        "mint": payload["mint"], "passed": False, "blocked": True,
    })
    worker = AgentWorker(store, registry=registry)

    scan = worker.run_once()
    assert scan["status"] == "completed"
    snapshots = store.list_solana_research_snapshots("owner-1", workspace["id"])
    assert len(snapshots) == 1
    assert snapshots[0]["task_id"] == scan["id"]
    assert snapshots[0]["capability"] == "solana.market.scan"
    assert snapshots[0]["mint"] == "A" * 40
    assert snapshots[0]["fetched_at"].startswith("2025-10-09T")
    assert len(snapshots[0]["payload_sha256"]) == 64
    with pytest.raises(LookupError, match="workspace not found"):
        store.list_solana_research_snapshots("owner-2", workspace["id"])
    queued = store.list_agent_tasks("owner-1", workspace["id"], status="queued")
    assert len(queued) == 1
    assert queued[0]["target_capability"] == "solana.token.audit"
    assert queued[0]["input"]["source_scan_task_id"] == scan["id"]

    audit = worker.run_once()
    assert audit["status"] == "completed"
    assert audit["result"]["blocked"] is True
    snapshots = store.list_solana_research_snapshots("owner-1", workspace["id"])
    assert len(snapshots) == 2
    assert {item["capability"] for item in snapshots} == {
        "solana.market.scan", "solana.token.audit",
    }
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


def test_solana_refresh_scheduler_is_idempotent_and_owner_scoped(tmp_path):
    store = MemoryStore(tmp_path / "solana-refresh.sqlite3")
    workspace = _workspace(store)
    job = store.schedule_solana_research_refresh(
        "owner-1", workspace["id"], "A" * 40, "Pool111", interval_seconds=300
    )
    with sqlite3.connect(store.database_path) as conn:
        conn.execute(
            "UPDATE solana_research_refresh_jobs SET next_due_at=? WHERE id=?",
            ("2000-01-01T00:00:00Z", job["id"]),
        )
    assert store.enqueue_due_solana_research_refreshes() == 1
    assert store.enqueue_due_solana_research_refreshes() == 0
    queued = store.list_agent_tasks("owner-1", workspace["id"], status="queued")
    assert len(queued) == 1
    assert queued[0]["target_capability"] == "solana.market.scan"
    assert queued[0]["requester"] == "eti.solana.refresh"
    assert queued[0]["input"]["scheduled_by"] == "eti.solana.refresh"


def test_solana_delayed_labels_are_persisted_once_and_only_after_future_sample(tmp_path):
    store = MemoryStore(tmp_path / "solana-labels.sqlite3")
    workspace = _workspace(store)
    base_ms = 1760000000000
    for index, observed_ms in enumerate((base_ms, base_ms + 301000)):
        task = store.create_agent_task(
            "owner-1", workspace["id"], "test.solver", "solana.market.scan",
            {"mint": "B" * 40}, max_attempts=1,
        )
        claimed = store.claim_next_agent_task(lease_seconds=120, now=1760000001 + index * 301)
        result = {
            "mint": "B" * 40, "verdict": "PASS", "observed_at_ms": observed_ms,
            "evidence": {
                "pair_address": "Pool222", "price_usd": 1.0 + index * 0.2,
                "liquidity_usd": 100000 - index * 10000, "volume_h24_usd": 50000,
            },
        }
        store.complete_claimed_agent_task(
            claimed["id"], claimed["lease_token"], result,
            now=1760000002 + index * 301,
        )
    before_horizon = store.generate_solana_research_labels(
        "owner-1", workspace["id"],
        now=datetime.fromtimestamp((base_ms + 299000) / 1000, timezone.utc),
    )
    assert before_horizon["generated"] == 0
    after_horizon = store.generate_solana_research_labels(
        "owner-1", workspace["id"],
        now=datetime.fromtimestamp((base_ms + 400000) / 1000, timezone.utc),
    )
    assert after_horizon["generated"] == 1
    repeated = store.generate_solana_research_labels(
        "owner-1", workspace["id"],
        now=datetime.fromtimestamp((base_ms + 400000) / 1000, timezone.utc),
    )
    assert repeated["generated"] == 0
    labels = store.list_solana_research_labels("owner-1", workspace["id"])
    assert len(labels) == 1
    label = labels[0]["label"]
    assert label["baseline_snapshot_id"] != label["target_snapshot_id"]
    assert label["sampled_price_return_pct"] == 20.0
    assert label["liquidity_change_pct"] == -10.0
    assert label["exitability_status"] == "UNKNOWN"


def test_new_mint_without_indexed_pool_is_rechecked_until_pool_appears(tmp_path):
    store = MemoryStore(tmp_path / "solana-pending-pool.sqlite3")
    workspace = _workspace(store)
    initial = store.create_agent_task(
        "owner-1", workspace["id"], "eti.solana.ingest", "solana.market.scan",
        {"mint": "C" * 40, "scheduled_by": "eti.solana.ingest"}, max_attempts=2,
    )
    claimed = store.claim_next_agent_task()
    assert claimed["id"] == initial["id"]
    store.complete_claimed_agent_task(
        claimed["id"], claimed["lease_token"],
        {"mint": "C" * 40, "verdict": "WAIT", "observed_at_ms": 1760000000000},
    )
    with sqlite3.connect(store.database_path) as conn:
        row = conn.execute(
            "SELECT pool_address,enabled FROM solana_research_refresh_jobs WHERE mint=?",
            ("C" * 40,),
        ).fetchone()
        assert row == ("__pending__", 1)
        conn.execute(
            "UPDATE solana_research_refresh_jobs SET next_due_at=? WHERE mint=?",
            ("2000-01-01T00:00:00Z", "C" * 40),
        )
    assert store.enqueue_due_solana_research_refreshes() == 1

    registry = AgentCapabilityRegistry()
    registry.register("solana.market.scan", lambda payload: {
        "mint": payload["mint"], "verdict": "PASS", "observed_at_ms": 1760000900000,
        "evidence": {"pair_address": "Pool333", "price_usd": 0.5,
                     "liquidity_usd": 60000, "volume_h24_usd": 10000},
    })
    registry.register("solana.token.audit", lambda payload: {
        "mint": payload["mint"], "passed": False, "blocked": True,
    })
    worker = AgentWorker(store, registry=registry)
    # The initial event's read-only audit task is ahead of the refresh task.
    assert worker.run_once()["target_capability"] == "solana.token.audit"
    refreshed = worker.run_once()
    assert refreshed["target_capability"] == "solana.market.scan"
    assert refreshed["status"] == "completed"
    with sqlite3.connect(store.database_path) as conn:
        rows = conn.execute(
            "SELECT pool_address,enabled FROM solana_research_refresh_jobs WHERE mint=? ORDER BY pool_address",
            ("C" * 40,),
        ).fetchall()
    assert ("__pending__", 0) in rows
    assert ("Pool333", 1) in rows
