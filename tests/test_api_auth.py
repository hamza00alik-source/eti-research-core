from fastapi.testclient import TestClient

from eti.api import app


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
        assert running.status_code == 200
        completed = client.patch(
            f"/v1/workspaces/{workspace_id}/agents/tasks/{task['id']}",
            headers=headers,
            json={"status": "completed", "result": {"summary": "research complete"}},
        )
        assert completed.status_code == 200
        assert completed.json()["status"] == "completed"

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
