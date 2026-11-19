import numpy as np
import pandas as pd
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




def _promotion_payload(challenger_sha):
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    folds = []
    for index in range(5):
        folds.append({
            "test_start": (start + pd.Timedelta(days=30 * index)).isoformat(),
            "test_end": (start + pd.Timedelta(days=30 * index + 20)).isoformat(),
            "test_bars": 200, "net_return_bps": 20.0,
            "max_drawdown": 0.02, "out_of_sample_ece": 0.03,
        })
    champion_folds = [{**item, "net_return_bps": 10.0} for item in folds]
    return {
        "comparison_family_size": 1,
        "independent_cost_evidence_verified": True,
        "cost_evidence_sha256": "2" * 64,
        "cost_evidence_source": "independent exchange fee schedule and observed spread sample",
        "cost_evidence_as_of": "2025-06-01T00:00:00Z",
        "champion_report": {
            "model_sha256": "c" * 64, "training_data_sha256": "b" * 64,
            "evaluation_data_sha256": "d" * 64, "cost_model_sha256": "e" * 64,
            "cost_adjusted": True, "evaluation_locked": True, "folds": champion_folds,
        },
        "challenger_report": {
            "model_sha256": challenger_sha, "training_data_sha256": "b" * 64,
            "evaluation_data_sha256": "d" * 64, "cost_model_sha256": "e" * 64,
            "code_sha256": "f" * 64, "config_sha256": "1" * 64,
            "experiment_id": "eti-exp-test-001",
            "cost_adjusted": True, "evaluation_locked": True, "folds": folds,
        },
    }

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
            "input_schema": {
                "features": ["close", "volume"],
                "lineage": {
                    "dataset_sha256": "b" * 64, "cost_model_sha256": "e" * 64,
                    "code_sha256": "f" * 64, "config_sha256": "1" * 64,
                },
            },
            "experiment_id": "eti-exp-test-001",
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
            "/v1/models/eti-core/0.3.0/approve", headers=reviewer_headers,
            json=_promotion_payload("a" * 64),
        )
        assert schema_mismatch.status_code == 409
        CalibratedStub.estimator.feature_names_in_ = np.array(["close", "volume"])
        mismatched_evidence = client.post(
            "/v1/models/eti-core/0.3.0/approve", headers=reviewer_headers,
            json=_promotion_payload("f" * 64),
        )
        assert mismatched_evidence.status_code == 409
        wrong_dataset = _promotion_payload("a" * 64)
        wrong_dataset["challenger_report"]["training_data_sha256"] = "9" * 64
        assert client.post(
            "/v1/models/eti-core/0.3.0/approve", headers=reviewer_headers,
            json=wrong_dataset,
        ).status_code == 409
        wrong_code = _promotion_payload("a" * 64)
        wrong_code["challenger_report"]["code_sha256"] = "2" * 64
        assert client.post(
            "/v1/models/eti-core/0.3.0/approve", headers=reviewer_headers,
            json=wrong_code,
        ).status_code == 409
        wrong_experiment = _promotion_payload("a" * 64)
        wrong_experiment["challenger_report"]["experiment_id"] = "eti-exp-other"
        assert client.post(
            "/v1/models/eti-core/0.3.0/approve", headers=reviewer_headers,
            json=wrong_experiment,
        ).status_code == 409
        unverified_costs = _promotion_payload("a" * 64)
        unverified_costs["independent_cost_evidence_verified"] = False
        assert client.post(
            "/v1/models/eti-core/0.3.0/approve", headers=reviewer_headers,
            json=unverified_costs,
        ).status_code == 409

        no_lineage_payload = {
            **payload, "model_id": "eti-no-lineage", "version": "0.1.0",
            "input_schema": {"features": ["close", "volume"]},
            "experiment_id": "eti-exp-no-lineage",
        }
        assert client.post(
            "/v1/models", headers=trainer_headers, json=no_lineage_payload
        ).status_code == 201
        no_lineage_approval = client.post(
            "/v1/models/eti-no-lineage/0.1.0/approve",
            headers=reviewer_headers, json=_promotion_payload("a" * 64),
        )
        assert no_lineage_approval.status_code == 409
        assert "complete code/config/data/cost lineage" in no_lineage_approval.text
        assert client.get(
            "/v1/models/eti-no-lineage/0.1.0", headers=trainer_headers
        ).json()["lifecycle"] == "candidate"

        approved = client.post(
            "/v1/models/eti-core/0.3.0/approve", headers=reviewer_headers,
            json=_promotion_payload("a" * 64),
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["approved_by"] == "reviewer_admin"
        assert approved.json()["promotion_assessment"]["promotion_eligible"] is True
        audit = client.get("/v1/audit/events", headers=reviewer_headers).json()["items"]
        promotion_event = next(item for item in audit if item["event_type"] == "model.promotion.approved")
        assert promotion_event["payload"]["model_id"] == "eti-core"
        assert promotion_event["payload"]["independent_cost_evidence_verified"] is True
        assert promotion_event["payload"]["cost_evidence_sha256"] == "2" * 64
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
            "input_schema": {
                "features": ["feature_a", "feature_b"],
                "department": "prediction", "model_role": "direction_classifier",
                "market_types": ["spot"], "timeframes": ["1h"],
                "lineage": {
                    "dataset_sha256": "b" * 64, "cost_model_sha256": "e" * 64,
                    "code_sha256": "f" * 64, "config_sha256": "1" * 64,
                },
            },
            "experiment_id": "eti-exp-test-001",
            "supported_classes": ["crypto_spot"],
            "training_data_ref": "synthetic-fixture",
            "evaluation_report": {"passed": True},
            "calibration_status": "calibrated",
        })
        assert registered.status_code == 201, registered.text
        approved = client.post(
            "/v1/models/eti-inference-test/0.1.0/approve",
            headers=reviewer_headers, json=_promotion_payload(digest),
        )
        assert approved.status_code == 200

        body = {
            "asset_class": "crypto_spot", "model_role": "direction_classifier",
            "market_type": "spot", "timeframe": "1h",
            "features": {"feature_a": 0.15, "feature_b": 0.7},
        }
        response = client.post(
            "/v1/inference/eti-inference-test/0.1.0",
            headers=user_headers, json=body,
        )
        assert response.status_code == 200, response.text
        assert response.json()["prediction"] == "flat"
        assert response.json()["probabilities"] == {
            "short": 0.2, "flat": 0.5, "long": 0.3
        }

        wrong_role = client.post(
            "/v1/inference/eti-inference-test/0.1.0",
            headers=user_headers, json={**body, "model_role": "volatility_forecast"},
        )
        assert wrong_role.status_code == 422
        wrong_market_type = client.post(
            "/v1/inference/eti-inference-test/0.1.0",
            headers=user_headers, json={**body, "market_type": "swap"},
        )
        assert wrong_market_type.status_code == 422
        wrong_timeframe = client.post(
            "/v1/inference/eti-inference-test/0.1.0",
            headers=user_headers, json={**body, "timeframe": "4h"},
        )
        assert wrong_timeframe.status_code == 422

        unsupported = client.post(
            "/v1/inference/eti-inference-test/0.1.0",
            headers=user_headers,
            json={**body, "asset_class": "forex"},
        )
        assert unsupported.status_code == 422
        wrong_schema = client.post(
            "/v1/inference/eti-inference-test/0.1.0",
            headers=user_headers,
            json={**body, "features": {"feature_a": 1.0}},
        )
        assert wrong_schema.status_code == 422

        non_finite_input = client.post(
            "/v1/inference/eti-inference-test/0.1.0",
            headers={**user_headers, "Content-Type": "application/json"},
            content='{"asset_class":"crypto_spot","model_role":"direction_classifier","market_type":"spot","timeframe":"1h","features":{"feature_a":NaN,"feature_b":0.7}}',
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


def test_paper_portfolio_api_is_authenticated_owner_scoped_and_explicitly_attested(tmp_path, monkeypatch):
    monkeypatch.setenv("ETI_DATABASE_PATH", str(tmp_path / "portfolio-api.sqlite3"))
    monkeypatch.setenv("ETI_AUTH_SECRET", "server-only-secret-value-at-least-32-bytes")
    monkeypatch.setenv("ETI_REGISTRATION_KEY", "private-invite-code-123456")
    with TestClient(app) as client:
        assert client.get("/v1/paper/portfolios").status_code == 401
        owner = _register(client, "portfolio_owner")
        other = _register(client, "portfolio_other")
        owner_headers = {"Authorization": f"Bearer {owner['access_token']}"}
        other_headers = {"Authorization": f"Bearer {other['access_token']}"}
        workspace = client.post("/v1/workspaces", headers=owner_headers,
                                json={"name": "portfolio lab"}).json()
        workspace_id = workspace["id"]
        created = client.post("/v1/paper/portfolios", headers=owner_headers, json={
            "name": "primary paper", "base_currency": "USD",
            "initial_equity": 10000, "workspace_id": workspace_id,
        })
        assert created.status_code == 201, created.text
        account_id = created.json()["account"]["id"]
        assert created.json()["live_execution_enabled"] is False
        assert "not connected to an exchange" in created.json()["note"]
        snapshot = client.post(
            f"/v1/paper/portfolios/{account_id}/snapshots",
            headers=owner_headers, params={"workspace_id": workspace_id},
            json={
                "as_of": "2026-11-16T10:00:00Z", "equity": 10000, "cash": 7000,
                "positions": [
                    {"symbol": "BTC/USD", "notional": 2000, "correlation_group": "crypto_major"},
                    {"symbol": "ETH/USD", "notional": 1000, "correlation_group": "crypto_major"},
                ],
            },
        )
        assert snapshot.status_code == 201, snapshot.text
        assert snapshot.json()["source"] == "owner_attested"
        latest = client.get(
            f"/v1/paper/portfolios/{account_id}/snapshots/latest",
            headers=owner_headers, params={"workspace_id": workspace_id},
        )
        assert latest.status_code == 200
        assert latest.json()["snapshot"]["id"] == snapshot.json()["snapshot"]["id"]
        denied = client.get(
            f"/v1/paper/portfolios/{account_id}/snapshots/latest",
            headers=other_headers, params={"workspace_id": workspace_id},
        )
        assert denied.status_code == 404
        invalid = client.post(
            f"/v1/paper/portfolios/{account_id}/snapshots",
            headers=owner_headers, params={"workspace_id": workspace_id},
            json={"as_of": "2026-11-16T10:01:00Z", "equity": 1000, "cash": 1001, "positions": []},
        )
        assert invalid.status_code == 422
