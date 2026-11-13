from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_container_runs_as_non_root_and_has_healthcheck():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "useradd --system" in dockerfile
    entrypoint = (ROOT / "docker-entrypoint.sh").read_text(encoding="utf-8")
    assert "chown -R eti:eti /app/data" in entrypoint
    assert "exec gosu eti" in entrypoint
    assert "HEALTHCHECK" in dockerfile
    assert "PORT" in dockerfile
    assert "${PORT:-8000}" in dockerfile
    assert "exec uvicorn eti.api:app" in dockerfile


def test_compose_separates_api_and_worker_with_shared_persistent_data():
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert "  api:" in compose
    assert "  worker:" in compose
    assert '["python", "-m", "eti.worker"' in compose
    assert "eti-data:/app/data" in compose
    assert "${ETI_AUTH_SECRET:?Set ETI_AUTH_SECRET in .env}" in compose
    assert "${ETI_REGISTRATION_KEY:?Set ETI_REGISTRATION_KEY in .env}" in compose
    assert "127.0.0.1" in compose
    assert "no-new-privileges:true" in compose


def test_example_environment_contains_placeholders_not_live_secrets():
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "REPLACE_WITH_RANDOM_SECRET" in env
    assert "REPLACE_WITH_PRIVATE_INVITE_KEY" in env
    assert "127.0.0.1" in env
    assert "-----BEGIN" not in env
