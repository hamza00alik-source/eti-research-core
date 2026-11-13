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

        created = client.post(
            f"/v1/workspaces/{workspace_id}/agents/tasks",
            headers=headers,
            json={"target_capability": "market.research",
                  "input": {"symbol": "BTC/USDT", "timeframe": "1h"}},
        )
        assert created.status_code == 201, created.text
        task = created.json()
        assert task["status"] == "queued"

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
