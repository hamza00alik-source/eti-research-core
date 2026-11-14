import json
import numpy as np
import pytest
from fastapi.testclient import TestClient

from eti.api import app
from eti.artifacts import ModelArtifactStore
from eti.storage import MemoryStore


def _register(client, username, invite="private-invite-code-123456"):
    response = client.post("/v1/auth/register", json={
        "username": username,
        "password": "a-long-password-123",
        "invite_code": invite,
    })
    assert response.status_code == 201, response.text
    return response.json()


def test_workspace_memory_and_agent_apis_require_auth_and_isolate_accounts(tmp_path, monkeypatch):
    monkeypatch.setenv("ETI_DATABASE_PATH", str(tmp_path / "eti.sqlite3"))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    monkeypatch.setenv("ETI_REGISTRATION_KEY", "private-invite-code-123456")
    with TestClient(app) as client:
        assert client.get("/v1/workspaces").status_code == 401
        account_a = _register(client, "account_a")
        account_b = _register(client, "account_b")
        headers_a = {"Authorization": f"Bearer {account_a['access_token']}"}
        headers_b = {"Authorization": f"Bearer {account_b['access_token']}"}

        created = client.post("/v1/workspaces", headers=headers_a,
                              json={"name": "main research"})
        assert created.status_code == 201, created.text
        workspace_id = created.json()["id"]
        assert client.get("/v1/workspaces", headers=headers_b).json()["items"] == []

        put = client.put(
            f"/v1/workspaces/{workspace_id}/memory/preferences/markets",
            headers=headers_a, json={"value": ["crypto", "gold"]},
        )
        assert put.status_code == 200, put.text
        read = client.get(
            f"/v1/workspaces/{workspace_id}/memory/preferences/markets",
            headers=headers_a,
        )
        assert read.status_code == 200
        assert read.json()["value"] == ["crypto", "gold"]

        assert client.get(
            f"/v1/workspaces/{workspace_id}/memory",
            headers=headers_b,
        ).status_code == 404
        assert client.put(
            f"/v1/workspaces/{workspace_id}/memory/preferences/secret",
            headers=headers_b, json={"value": "not yours"},
        ).status_code == 404


def test_authenticated_agent_task_and_message_ledger_api(tmp_path, monkeypatch):
    monkeypatch.setenv("ETI_DATABASE_PATH", str(tmp_path / "agents.sqlite3"))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    monkeypatch.setenv("ETI_REGISTRATION_KEY", "private-invite-code-123456")
    with TestClient(app) as client:
        account = _register(client, "agent_user")
        headers = {"Authorization": f"Bearer {account['access_token']}"}
        workspace = client.post("/v1/workspaces", headers=headers,
                                json={"name": "agent lab"}).json()
        workspace_id = workspace["id"]

        capabilities = client.get("/v1/agents/capabilities", headers=headers)
        assert capabilities.status_code == 200
        assert "market.snapshot" in capabilities.json()["items"]

        created = client.post(
            f"/v1/workspaces/{workspace_id}/agents/tasks",
            headers=headers,
            json={"target_capability": "market.snapshot",
                  "input": {"symbol": "BTC/USDT", "timeframe": "1h"},
                  "max_attempts": 2},
        )
        assert created.status_code == 201, created.text
        task = created.json()
        assert task["status"] == "queued"
        assert task["max_attempts"] == 2

        listed = client.get(f"/v1/workspaces/{workspace_id}/agents/tasks",
                            headers=headers)
        assert len(listed.json()["items"]) == 1
        running = client.patch(
            f"/v1/workspaces/{workspace_id}/agents/tasks/{task['id']}",
            headers=headers, json={"status": "running"},
        )
        assert running.status_code == 403
        completed = client.patch(
            f"/v1/workspaces/{workspace_id}/agents/tasks/{task['id']}",
            headers=headers,
            json={"status": "completed", "result": {"summary": "research complete"}},
        )
        assert completed.status_code == 403
        cancelled = client.patch(
            f"/v1/workspaces/{workspace_id}/agents/tasks/{task['id']}",
            headers=headers, json={"status": "cancelled"},
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"

        message = client.post(
            f"/v1/workspaces/{workspace_id}/agents/messages",
            headers=headers,
            json={"recipient": "risk.agent", "message_type": "risk.request",
                  "payload": {"max_leverage": 1}, "task_id": task["id"]},
        )
        assert message.status_code == 201, message.text
        messages = client.get(
            f"/v1/workspaces/{workspace_id}/agents/messages",
            headers=headers,
        )
        assert len(messages.json()["items"]) == 1
        assert messages.json()["items"][0]["payload"]["max_leverage"] == 1


def test_api_security_headers_prevent_private_response_caching(tmp_path, monkeypatch):
    monkeypatch.setenv("ETI_DATABASE_PATH", str(tmp_path / "headers.sqlite3"))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    with TestClient(app) as client:
        response = client.get("/v1/profile")
        assert response.status_code == 401
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        health = client.get("/health")
        assert health.headers["cache-control"] == "no-store"


def test_service_worker_does_not_cache_authenticated_api_routes():
    with TestClient(app) as client:
        response = client.get("/sw.js")
        assert response.status_code == 200
        assert "url.pathname.startsWith('/v1/')" in response.text
        assert "cache:'no-store'" in response.text


def test_login_and_registration_are_rate_limited(tmp_path, monkeypatch):
    monkeypatch.setenv("ETI_DATABASE_PATH", str(tmp_path / "rate-limit.sqlite3"))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    monkeypatch.setenv("ETI_REGISTRATION_KEY", "private-invite-code-123456")
    with TestClient(app) as client:
        for _ in range(10):
            response = client.post("/v1/auth/token", json={
                "username": "limited_user", "password": "a-long-password-123"
            })
            assert response.status_code == 401
        limited = client.post("/v1/auth/token", json={
            "username": "limited_user", "password": "a-long-password-123"
        })
        assert limited.status_code == 429
        assert int(limited.headers["retry-after"]) > 0

        for index in range(5):
            response = client.post("/v1/auth/register", json={
                "username": f"register_{index}", "password": "a-long-password-123",
                "invite_code": "wrong-invite-code",
            })
            assert response.status_code == 401
        limited_register = client.post("/v1/auth/register", json={
            "username": "register_final", "password": "a-long-password-123",
            "invite_code": "wrong-invite-code",
        })
        assert limited_register.status_code == 429


def test_model_registry_is_admin_governed_and_hides_candidates(tmp_path, monkeypatch):
    class CalibratedStub:
        estimator = type("Estimator", (), {
            "feature_names_in_": np.array(["close", "volume"])
        })()
        calibration_status = "calibrated"
        calibration_report = {
            "passed": True, "method": "temperature_scaling",
            "sample_count": 50, "temperature": 1.2,
            "class_counts": {"0": 16, "1": 17, "2": 17},
            "nll_before": 1.2, "nll_after": 1.0,
        }

    monkeypatch.setattr(ModelArtifactStore, "load_model", lambda self, uri, checksum: CalibratedStub())
    monkeypatch.setenv("ETI_DATABASE_PATH", str(tmp_path / "models.sqlite3"))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    monkeypatch.setenv("ETI_REGISTRATION_KEY", "private-invite-code-123456")
    monkeypatch.setenv("ETI_ADMIN_USERNAMES", "trainer_admin,reviewer_admin")
    with TestClient(app) as client:
        trainer = _register(client, "trainer_admin")
        reviewer = _register(client, "reviewer_admin")
        regular = _register(client, "ordinary_user")
        trainer_headers = {"Authorization": f"Bearer {trainer['access_token']}"}
        reviewer_headers = {"Authorization": f"Bearer {reviewer['access_token']}"}
        regular_headers = {"Authorization": f"Bearer {regular['access_token']}"}

        payload = {
            "model_id": "eti-core",
            "version": "0.3.0",
            "artifact_uri": "artifact://eti-core/0.3.0",
            "artifact_sha256": "a" * 64,
            "input_schema": {"features": ["close", "volume"]},
            "supported_classes": ["crypto_spot"],
            "training_data_ref": "dataset:training-2026-11",
            "evaluation_report": {"passed": True, "balanced_accuracy": 0.61},
            "calibration_status": "calibrated",
        }
        assert client.post("/v1/models", headers=regular_headers, json=payload).status_code == 403
        created = client.post("/v1/models", headers=trainer_headers, json=payload)
        assert created.status_code == 201, created.text
        assert created.json()["lifecycle"] == "candidate"

        assert client.get("/v1/models", headers=regular_headers).json()["items"] == []
        assert client.get("/v1/models/eti-core/0.3.0", headers=regular_headers).status_code == 404
        assert client.post(
            "/v1/models/eti-core/0.3.0/approve", headers=trainer_headers
        ).status_code == 409

        CalibratedStub.estimator.feature_names_in_ = np.array(["bad", "volume"])
        schema_mismatch = client.post(
            "/v1/models/eti-core/0.3.0/approve", headers=reviewer_headers
        )
        assert schema_mismatch.status_code == 409
        CalibratedStub.estimator.feature_names_in_ = np.array(["close", "volume"])

        approved = client.post(
            "/v1/models/eti-core/0.3.0/approve", headers=reviewer_headers
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["approved_by"] == "reviewer_admin"
        visible = client.get("/v1/models", headers=regular_headers)
        assert len(visible.json()["items"]) == 1
        assert visible.json()["items"][0]["lifecycle"] == "approved"

        retired = client.post(
            "/v1/models/eti-core/0.3.0/retire", headers=trainer_headers
        )
        assert retired.status_code == 200
        assert retired.json()["lifecycle"] == "retired"


def test_audit_endpoint_is_authenticated_and_account_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("ETI_DATABASE_PATH", str(tmp_path / "audit.sqlite3"))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    monkeypatch.setenv("ETI_REGISTRATION_KEY", "private-invite-code-123456")
    with TestClient(app) as client:
        assert client.get("/v1/audit/events").status_code == 401
        first = _register(client, "audit_first")
        second = _register(client, "audit_second")
        h1 = {"Authorization": f"Bearer {first['access_token']}"}
        h2 = {"Authorization": f"Bearer {second['access_token']}"}
        workspace = client.post("/v1/workspaces", headers=h1,
                                json={"name": "audit workspace"}).json()
        task = client.post(
            f"/v1/workspaces/{workspace['id']}/agents/tasks",
            headers=h1,
            json={"target_capability": "market.snapshot", "input": {"symbol": "BTC/USDT"}},
        )
        assert task.status_code == 201
        events = client.get("/v1/audit/events", headers=h1)
        assert events.status_code == 200
        assert any(item["event_type"] == "agent.task.queued"
                   for item in events.json()["items"])
        assert client.get("/v1/audit/events", headers=h2).json()["items"] == []


def test_agent_worker_status_is_authenticated_and_reports_heartbeat(tmp_path, monkeypatch):
    monkeypatch.setenv("ETI_DATABASE_PATH", str(tmp_path / "worker-status.sqlite3"))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    monkeypatch.setenv("ETI_REGISTRATION_KEY", "private-invite-code-123456")
    with TestClient(app) as client:
        assert client.get("/v1/agents/status").status_code == 401
        account = _register(client, "worker_status")
        headers = {"Authorization": f"Bearer {account['access_token']}"}
        MemoryStore().record_agent_worker_heartbeat("worker-test", ["market.snapshot"])
        response = client.get("/v1/agents/status", headers=headers)
        assert response.status_code == 200
        assert response.json()["online_count"] == 1
        assert response.json()["items"][0]["capabilities"] == ["market.snapshot"]


def test_approved_model_inference_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("ETI_DATABASE_PATH", str(tmp_path / "inference.sqlite3"))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    monkeypatch.setenv("ETI_REGISTRATION_KEY", "private-invite-code-123456")
    monkeypatch.setenv("ETI_ADMIN_USERNAMES", "trainer_admin,reviewer_admin")

    class StubModel:
        estimator = type("Estimator", (), {
            "feature_names_in_": np.array(["feature_a", "feature_b"])
        })()
        probabilities = [0.2, 0.5, 0.3]
        calibration_status = "calibrated"
        calibration_report = {
            "passed": True, "method": "temperature_scaling",
            "sample_count": 50, "temperature": 1.2,
            "class_counts": {"0": 16, "1": 17, "2": 17},
            "nll_before": 1.2, "nll_after": 1.0,
        }

        def predict_proba(self, frame):
            return np.array([self.probabilities])

    monkeypatch.setattr(
        ModelArtifactStore, "load_model",
        lambda self, uri, checksum: StubModel(),
    )
    digest = "a" * 64
    uri = "artifact://sha256/" + digest

    with TestClient(app) as client:
        trainer = _register(client, "trainer_admin")
        reviewer = _register(client, "reviewer_admin")
        user = _register(client, "ordinary_user")
        trainer_headers = {"Authorization": f"Bearer {trainer['access_token']}"}
        reviewer_headers = {"Authorization": f"Bearer {reviewer['access_token']}"}
        user_headers = {"Authorization": f"Bearer {user['access_token']}"}

        registered = client.post("/v1/models", headers=trainer_headers, json={
            "model_id": "eti-inference-test", "version": "0.1.0",
            "artifact_uri": uri, "artifact_sha256": digest,
            "input_schema": {"features": ["feature_a", "feature_b"]},
            "supported_classes": ["crypto_spot"],
            "training_data_ref": "synthetic-fixture",
            "evaluation_report": {"passed": True},
            "calibration_status": "calibrated",
        })
        assert registered.status_code == 201, registered.text
        approved = client.post(
            "/v1/models/eti-inference-test/0.1.0/approve",
            headers=reviewer_headers,
        )
        assert approved.status_code == 200

        body = {"asset_class": "crypto_spot",
                "features": {"feature_a": 0.15, "feature_b": 0.7}}
        response = client.post(
            "/v1/inference/eti-inference-test/0.1.0",
            headers=user_headers, json=body,
        )
        assert response.status_code == 200, response.text
        assert response.json()["prediction"] == "flat"
        assert response.json()["probabilities"] == {
            "short": 0.2, "flat": 0.5, "long": 0.3
        }

        unsupported = client.post(
            "/v1/inference/eti-inference-test/0.1.0",
            headers=user_headers,
            json={"asset_class": "forex", "features": body["features"]},
        )
        assert unsupported.status_code == 422
        wrong_schema = client.post(
            "/v1/inference/eti-inference-test/0.1.0",
            headers=user_headers,
            json={"asset_class": "crypto_spot", "features": {"feature_a": 1.0}},
        )
        assert wrong_schema.status_code == 422

        non_finite_input = client.post(
            "/v1/inference/eti-inference-test/0.1.0",
            headers={**user_headers, "Content-Type": "application/json"},
            content='{"asset_class":"crypto_spot","features":{"feature_a":NaN,"feature_b":0.7}}',
        )
        assert non_finite_input.status_code == 422

        StubModel.probabilities = [0.2, 0.2, 0.2]
        invalid_distribution = client.post(
            "/v1/inference/eti-inference-test/0.1.0",
            headers=user_headers, json=body,
        )
        assert invalid_distribution.status_code == 503

        StubModel.probabilities = [0.2, float("nan"), 0.8]
        non_finite_output = client.post(
            "/v1/inference/eti-inference-test/0.1.0",
            headers=user_headers, json=body,
        )
        assert non_finite_output.status_code == 503


def test_model_approval_rejects_missing_server_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("ETI_DATABASE_PATH", str(tmp_path / "missing-artifact.sqlite3"))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    monkeypatch.setenv("ETI_REGISTRATION_KEY", "private-invite-code-123456")
    monkeypatch.setenv("ETI_ADMIN_USERNAMES", "trainer_admin,reviewer_admin")
    digest = "b" * 64
    with TestClient(app) as client:
        trainer = _register(client, "trainer_admin")
        reviewer = _register(client, "reviewer_admin")
        trainer_headers = {"Authorization": f"Bearer {trainer['access_token']}"}
        reviewer_headers = {"Authorization": f"Bearer {reviewer['access_token']}"}
        payload = {
            "model_id": "missing-artifact-model",
            "version": "0.1.0",
            "artifact_uri": "artifact://sha256/" + digest,
            "artifact_sha256": digest,
            "input_schema": {"features": ["x"]},
            "supported_classes": ["crypto_spot"],
            "training_data_ref": "test-fixture",
            "evaluation_report": {"passed": True},
            "calibration_status": "calibrated",
        }
        created = client.post("/v1/models", headers=trainer_headers, json=payload)
        assert created.status_code == 201
        denied = client.post(
            "/v1/models/missing-artifact-model/0.1.0/approve",
            headers=reviewer_headers,
        )
        assert denied.status_code == 409
        candidate = client.get(
            "/v1/models/missing-artifact-model/0.1.0",
            headers=trainer_headers,
        )
        assert candidate.status_code == 200
        assert candidate.json()["lifecycle"] == "candidate"


def test_account_export_is_authenticated_and_contains_only_callers_data(tmp_path, monkeypatch):
    monkeypatch.setenv("ETI_DATABASE_PATH", str(tmp_path / "export.sqlite3"))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    monkeypatch.setenv("ETI_REGISTRATION_KEY", "private-invite-code-123456")
    with TestClient(app) as client:
        assert client.get("/v1/account/export").status_code == 401
        first = _register(client, "export_first")
        second = _register(client, "export_second")
        h1 = {"Authorization": f"Bearer {first['access_token']}"}
        h2 = {"Authorization": f"Bearer {second['access_token']}"}
        workspace = client.post("/v1/workspaces", headers=h1, json={"name": "private"}).json()
        client.put(f"/v1/workspaces/{workspace['id']}/memory/prefs/market",
                   headers=h1, json={"value": ["crypto", "gold"]})
        exported = client.get("/v1/account/export", headers=h1)
        assert exported.status_code == 200
        payload = exported.json()
        assert payload["format"] == "eti-account-export-v1"
        assert payload["workspaces"][0]["id"] == workspace["id"]
        assert payload["memory"][0]["value"] == ["crypto", "gold"]
        assert "password_hash" not in exported.text
        assert client.get("/v1/account/export", headers=h2).json()["workspaces"] == []


def test_workspace_deletion_is_owner_scoped_cascades_data_and_keeps_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("ETI_DATABASE_PATH", str(tmp_path / "delete-workspace.sqlite3"))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    monkeypatch.setenv("ETI_REGISTRATION_KEY", "private-invite-code-123456")
    with TestClient(app) as client:
        first = _register(client, "delete_first")
        second = _register(client, "delete_second")
        h1 = {"Authorization": f"Bearer {first['access_token']}"}
        h2 = {"Authorization": f"Bearer {second['access_token']}"}
        workspace = client.post("/v1/workspaces", headers=h1, json={"name": "to-delete"}).json()
        client.put(f"/v1/workspaces/{workspace['id']}/memory/prefs/key",
                   headers=h1, json={"value": "private"})
        assert client.delete(f"/v1/workspaces/{workspace['id']}", headers=h2).status_code == 404
        deleted = client.delete(f"/v1/workspaces/{workspace['id']}", headers=h1)
        assert deleted.status_code == 204
        assert client.get("/v1/workspaces", headers=h1).json()["items"] == []
        events = client.get("/v1/audit/events", headers=h1).json()["items"]
        assert any(event["event_type"] == "workspace.deleted" for event in events)


def test_registry_cannot_approve_artifact_without_embedded_calibration(tmp_path, monkeypatch):
    monkeypatch.setenv("ETI_DATABASE_PATH", str(tmp_path / "uncalibrated.sqlite3"))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    monkeypatch.setenv("ETI_REGISTRATION_KEY", "private-invite-code-123456")
    monkeypatch.setenv("ETI_ADMIN_USERNAMES", "trainer_admin,reviewer_admin")

    class UncalibratedStub:
        calibration_status = "uncalibrated"
        calibration_report = None

    monkeypatch.setattr(
        ModelArtifactStore, "load_model",
        lambda self, uri, checksum: UncalibratedStub(),
    )
    digest = "d" * 64
    with TestClient(app) as client:
        trainer = _register(client, "trainer_admin")
        reviewer = _register(client, "reviewer_admin")
        trainer_headers = {"Authorization": f"Bearer {trainer['access_token']}"}
        reviewer_headers = {"Authorization": f"Bearer {reviewer['access_token']}"}
        payload = {
            "model_id": "uncalibrated-model", "version": "0.1.0",
            "artifact_uri": "artifact://sha256/" + digest, "artifact_sha256": digest,
            "input_schema": {"features": ["x"]}, "supported_classes": ["crypto_spot"],
            "training_data_ref": "test-fixture",
            "evaluation_report": {"passed": True}, "calibration_status": "calibrated",
        }
        assert client.post("/v1/models", headers=trainer_headers, json=payload).status_code == 201
        denied = client.post(
            "/v1/models/uncalibrated-model/0.1.0/approve", headers=reviewer_headers
        )
        assert denied.status_code == 409
        candidate = client.get(
            "/v1/models/uncalibrated-model/0.1.0", headers=trainer_headers
        )
        assert candidate.json()["lifecycle"] == "candidate"


def test_authenticated_solana_event_ledger_is_idempotent_and_workspace_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("ETI_DATABASE_PATH", str(tmp_path / "solana-api.sqlite3"))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    monkeypatch.setenv("ETI_REGISTRATION_KEY", "private-invite-code-123456")
    with TestClient(app) as client:
        owner = _register(client, "solana_owner")
        other = _register(client, "solana_other")
        headers = {"Authorization": f"Bearer {owner['access_token']}"}
        other_headers = {"Authorization": f"Bearer {other['access_token']}"}
        workspace = client.post("/v1/workspaces", headers=headers,
                                json={"name": "solana research"}).json()
        workspace_id = workspace["id"]
        body = {
            "event_key": "pumpportal:sig-001",
            "source": "pumpportal",
            "event_type": "new_token",
            "mint": "A" * 40,
            "source_timestamp": "2026-11-14T12:00:00Z",
            "payload": {"signature": "sig-001", "slot": 12345},
        }
        first = client.post(
            f"/v1/workspaces/{workspace_id}/solana/events",
            headers=headers, json=body,
        )
        assert first.status_code == 201, first.text
        duplicate = client.post(
            f"/v1/workspaces/{workspace_id}/solana/events",
            headers=headers, json=body,
        )
        assert duplicate.status_code == 201
        assert duplicate.json()["id"] == first.json()["id"]
        assert duplicate.json()["duplicate"] is True
        listed = client.get(
            f"/v1/workspaces/{workspace_id}/solana/events",
            headers=headers,
        )
        assert listed.status_code == 200
        assert len(listed.json()["items"]) == 1
        assert client.get(
            f"/v1/workspaces/{workspace_id}/solana/events",
            headers=other_headers,
        ).status_code == 404
        assert client.get(
            f"/v1/workspaces/{workspace_id}/solana/events",
        ).status_code == 401

def test_solana_dataset_export_api_requires_auth_and_isolates_workspaces(tmp_path, monkeypatch):
    database = tmp_path / "solana-export-api.sqlite3"
    monkeypatch.setenv("ETI_DATABASE_PATH", str(database))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    monkeypatch.setenv("ETI_REGISTRATION_KEY", "private-invite-code-123456")
    with TestClient(app) as client:
        assert client.get("/v1/workspaces/not-owned/solana/dataset-export").status_code == 401
        owner = _register(client, "solana_export_owner")
        other = _register(client, "solana_export_other")
        owner_headers = {"Authorization": f"Bearer {owner['access_token']}"}
        other_headers = {"Authorization": f"Bearer {other['access_token']}"}
        workspace_response = client.post(
            "/v1/workspaces", headers=owner_headers, json={"name": "Solana export"}
        )
        assert workspace_response.status_code == 201, workspace_response.text
        workspace_id = workspace_response.json()["id"]

        own_export = client.get(
            f"/v1/workspaces/{workspace_id}/solana/dataset-export",
            headers=owner_headers,
        )
        assert own_export.status_code == 200, own_export.text
        assert own_export.json()["schema"] == "eti.solana.research_dataset.v1"
        assert own_export.json()["snapshot_count"] == 0

        foreign_export = client.get(
            f"/v1/workspaces/{workspace_id}/solana/dataset-export",
            headers=other_headers,
        )
        assert foreign_export.status_code == 404

        invalid_limit = client.get(
            f"/v1/workspaces/{workspace_id}/solana/dataset-export?limit=5001",
            headers=owner_headers,
        )
        assert invalid_limit.status_code == 422
        invalid_mint = client.get(
            f"/v1/workspaces/{workspace_id}/solana/dataset-export?mint=short",
            headers=owner_headers,
        )
        assert invalid_mint.status_code == 422


def test_solana_cost_evidence_api_is_authenticated_and_owner_scoped(tmp_path, monkeypatch):
    database = tmp_path / "solana-cost-api.sqlite3"
    monkeypatch.setenv("ETI_DATABASE_PATH", str(database))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    monkeypatch.setenv("ETI_REGISTRATION_KEY", "private-invite-code-123456")
    with TestClient(app) as client:
        account_a = _register(client, "cost_owner_a")
        account_b = _register(client, "cost_owner_b")
        headers_a = {"Authorization": f"Bearer {account_a['access_token']}"}
        headers_b = {"Authorization": f"Bearer {account_b['access_token']}"}
        workspace = client.post(
            "/v1/workspaces", headers=headers_a, json={"name": "cost research"}
        ).json()
        workspace_id = workspace["id"]
        mint, pool, label_id = "E" * 32, "pool-api", "cost-api-label"
        store = MemoryStore(database)
        with store._connection() as conn:
            conn.execute(
                "INSERT INTO solana_research_labels "
                "(id,owner_id,workspace_id,mint,pool_address,baseline_snapshot_id,target_snapshot_id,"
                "horizon_seconds,label_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (label_id, account_a["account"]["id"], workspace_id, mint, pool,
                 "baseline", "target", 3600, json.dumps({}), "2026-01-01T00:00:00Z"),
            )
        body = {
            "label_id": label_id, "mint": mint, "pool_address": pool,
            "evidence": {"label_id": label_id, "mint": mint, "pool_address": pool,
                         "cost_components": {}},
        }
        url = f"/v1/workspaces/{workspace_id}/solana/cost-evidence"
        assert client.post(url, json=body).status_code == 401
        saved = client.post(url, headers=headers_a, json=body)
        assert saved.status_code == 201, saved.text
        assert saved.json()["integrity_valid"] is True
        assert client.get(url, headers=headers_a).json()["items"][0]["id"] == saved.json()["id"]
        assert client.get(url, headers=headers_b).status_code == 404
        assert client.post(url, headers=headers_b, json=body).status_code == 404

def test_solana_transaction_verification_api_is_authenticated_and_workspace_scoped(tmp_path, monkeypatch):
    import eti.api as api_module

    monkeypatch.setenv("ETI_DATABASE_PATH", str(tmp_path / "solana-tx-verify.sqlite3"))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    monkeypatch.setenv("ETI_REGISTRATION_KEY", "private-invite-code-123456")

    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

    def b58(data):
        number = int.from_bytes(data, "big")
        encoded = ""
        while number:
            number, remainder = divmod(number, 58)
            encoded = alphabet[remainder] + encoded
        return "1" * (len(data) - len(data.lstrip(b"\\x00"))) + encoded

    signature = b58(bytes(range(1, 65)))
    mint = b58(bytes(range(65, 97)))
    pool = b58(bytes(range(129, 161)))
    wallet = b58(bytes(range(161, 193)))

    class FakeRpc:
        def call(self, method, params):
            if method == "getSignatureStatuses":
                return {"value": [{
                    "slot": 123,
                    "err": None,
                    "confirmationStatus": "finalized",
                }]}
            if method == "getTransaction":
                return {
                    "slot": 123,
                    "blockTime": 1794672000,
                    "transaction": {
                        "signatures": [signature],
                        "message": {"accountKeys": [{"pubkey": pool}, {"pubkey": wallet}]},
                    },
                    "meta": {
                        "err": None,
                        "preTokenBalances": [{
                            "mint": mint, "owner": wallet,
                            "uiTokenAmount": {"amount": "100", "decimals": 6},
                        }],
                        "postTokenBalances": [{
                            "mint": mint, "owner": wallet,
                            "uiTokenAmount": {"amount": "50", "decimals": 6},
                        }],
                    },
                }
            raise AssertionError(method)

    monkeypatch.setattr(api_module, "SolanaRpcClient", lambda: FakeRpc())
    with TestClient(app) as client:
        owner = _register(client, "tx_verify_owner")
        other = _register(client, "tx_verify_other")
        owner_headers = {"Authorization": f"Bearer {owner['access_token']}"}
        other_headers = {"Authorization": f"Bearer {other['access_token']}"}
        workspace = client.post(
            "/v1/workspaces", headers=owner_headers, json={"name": "tx verification"}
        ).json()
        url = f"/v1/workspaces/{workspace['id']}/solana/transaction-verification"
        payload = {
            "signature": signature, "mint": mint,
            "expected_pool_address": pool, "wallet_address": wallet,
        }

        assert client.post(url, json=payload).status_code == 401
        assert client.post(url, headers=other_headers, json=payload).status_code == 404
        response = client.post(url, headers=owner_headers, json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["verified"] is False
        assert response.json()["transaction_receipt_verified"] is True
        assert response.json()["status"] == "FINALIZED_MINT_BALANCE_DEBIT_OBSERVED"



def test_solana_transaction_attestation_api_verifies_and_persists_server_report(tmp_path, monkeypatch):
    import json
    import eti.api as api_module

    monkeypatch.setenv("ETI_DATABASE_PATH", str(tmp_path / "solana-tx-attest.sqlite3"))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    monkeypatch.setenv("ETI_REGISTRATION_KEY", "private-invite-code-123456")
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

    def b58(data):
        number = int.from_bytes(data, "big")
        encoded = ""
        while number:
            number, remainder = divmod(number, 58)
            encoded = alphabet[remainder] + encoded
        return "1" * (len(data) - len(data.lstrip(b"\\x00"))) + encoded

    signature = b58(bytes(range(1, 65)))
    mint = b58(bytes(range(65, 97)))
    pool = b58(bytes(range(129, 161)))
    wallet = b58(bytes(range(161, 193)))

    class FakeRpc:
        def call(self, method, params):
            if method == "getSignatureStatuses":
                return {"value": [{"slot": 7, "err": None, "confirmationStatus": "finalized"}]}
            if method == "getTransaction":
                return {
                    "slot": 7, "blockTime": 1794672000,
                    "transaction": {
                        "signatures": [signature],
                        "message": {"accountKeys": [{"pubkey": pool}, {"pubkey": wallet}]},
                    },
                    "meta": {
                        "err": None,
                        "preTokenBalances": [{"mint": mint, "owner": wallet,
                            "uiTokenAmount": {"amount": "80", "decimals": 6}}],
                        "postTokenBalances": [{"mint": mint, "owner": wallet,
                            "uiTokenAmount": {"amount": "30", "decimals": 6}}],
                    },
                }
            raise AssertionError(method)

    monkeypatch.setattr(api_module, "SolanaRpcClient", lambda: FakeRpc())
    with TestClient(app) as client:
        owner = _register(client, "tx_attest_owner")
        other = _register(client, "tx_attest_other")
        headers = {"Authorization": f"Bearer {owner['access_token']}"}
        other_headers = {"Authorization": f"Bearer {other['access_token']}"}
        workspace = client.post("/v1/workspaces", headers=headers, json={"name": "attest"}).json()
        workspace_id = workspace["id"]
        label_id = "attest-label"
        with api_module._memory_store()._connection() as conn:
            conn.execute(
                "INSERT INTO solana_research_labels "
                "(id,owner_id,workspace_id,mint,pool_address,baseline_snapshot_id,target_snapshot_id,"
                "horizon_seconds,label_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (label_id, owner["account"]["id"], workspace_id, mint, pool, "base", "target", 3600,
                 json.dumps({"baseline_at": "2026-01-01T00:00:00Z"}), "2026-01-01T00:00:00Z"),
            )
        url = f"/v1/workspaces/{workspace_id}/solana/transaction-attestations"
        body = {"label_id": label_id, "signature": signature, "mint": mint,
                "expected_pool_address": pool, "wallet_address": wallet}
        assert client.post(url, json=body).status_code == 401
        assert client.post(url, headers=other_headers, json=body).status_code == 404
        saved = client.post(url, headers=headers, json=body)
        assert saved.status_code == 200, saved.text
        assert saved.json()["recorded"] is True
        assert saved.json()["verification"]["verified"] is False
        assert saved.json()["attestation"]["integrity_valid"] is True
        listed = client.get(url, headers=headers)
        assert listed.status_code == 200
        assert len(listed.json()["items"]) == 1
        assert client.get(url, headers=other_headers).status_code == 404
