from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pwa_account_controls_are_connected_to_authenticated_api():
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    for selector in (
        "login-form", "register-form", "account-view", "workspace-form",
        "workspace-select", "capability-select", "memory-form", "task-form", "audit-list", "worker-status", "export-account", "delete-workspace",
    ):
        assert selector in html
        assert selector in script
    for endpoint in (
        "/v1/auth/token", "/v1/auth/register", "/v1/profile",
        "/v1/workspaces", "/v1/account/export", "/v1/agents/capabilities", "/v1/agents/status", "/v1/audit/events",
    ):
        assert endpoint in script


def test_pwa_does_not_persist_bearer_token_in_local_storage_or_cache():
    script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    service_worker = (ROOT / "web" / "sw.js").read_text(encoding="utf-8")
    assert "sessionStorage" in script
    assert "localStorage" not in script
    assert "url.pathname.startsWith('/v1/')" in service_worker
    assert "cache:'no-store'" in service_worker


def test_task_form_defaults_to_registered_mexc_snapshot_capability():
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert 'name="capability" value="market.snapshot"' not in html
    assert 'id="capability-select" name="capability"' in html
    assert '"venue":"mexc"' in html


def test_capability_picker_is_populated_from_server_allowlist():
    script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    assert "async function loadCapabilities()" in script
    assert "select.add(new Option(capability,capability))" in script


def test_pwa_exports_account_json_and_confirms_workspace_deletion():
    script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    assert "URL.createObjectURL(blob)" in script
    assert "eti-account-export-" in script
    assert "window.confirm(" in script
    assert "method:'DELETE'" in script
    assert "'/v1/workspaces/'+encodeURIComponent(activeWorkspace)" in script
