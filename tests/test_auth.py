import pytest

from eti.auth import AuthManager, AuthenticationError, RegistrationDisabled


def test_invite_only_registration_and_login_persist(tmp_path):
    path = tmp_path / "auth.sqlite3"
    auth = AuthManager(path, secret="s" * 40, registration_key="invite-key-123456")
    with pytest.raises(AuthenticationError):
        auth.register("trader01", "a-long-password-123", "wrong-key")
    account = auth.register("Trader01", "a-long-password-123", "invite-key-123456")
    assert account["username"] == "trader01"
    reopened = AuthManager(path, secret="s" * 40, registration_key="invite-key-123456")
    assert reopened.authenticate("TRADER01", "a-long-password-123") == account
    with pytest.raises(ValueError, match="already registered"):
        reopened.register("trader01", "another-long-password", "invite-key-123456")


def test_password_policy_and_unknown_user(tmp_path):
    auth = AuthManager(tmp_path / "auth.sqlite3", secret="s" * 40,
                       registration_key="invite-key-123456")
    with pytest.raises(ValueError, match="12-1024"):
        auth.register("trader01", "short", "invite-key-123456")
    with pytest.raises(AuthenticationError):
        auth.authenticate("missing1", "a-long-password-123")


def test_tokens_are_signed_expiring_and_bound_to_account(tmp_path):
    auth = AuthManager(tmp_path / "auth.sqlite3", secret="s" * 40,
                       registration_key="invite-key-123456", token_ttl_seconds=300)
    account = auth.register("trader01", "a-long-password-123", "invite-key-123456")
    token = auth.issue_token(account, now=1000)
    assert auth.verify_token(token, now=1001) == account
    with pytest.raises(AuthenticationError, match="invalid or expired"):
        auth.verify_token(token, now=1300)
    parts = token.split(".")
    signature = parts[2]
    parts[2] = ("A" if signature[0] != "A" else "B") + signature[1:]
    with pytest.raises(AuthenticationError):
        auth.verify_token(".".join(parts), now=1001)
    other = AuthManager(tmp_path / "auth.sqlite3", secret="x" * 40,
                        registration_key="invite-key-123456")
    with pytest.raises(AuthenticationError):
        other.verify_token(token, now=1001)


def test_auth_fails_closed_without_server_secret_or_invite(tmp_path):
    with pytest.raises(RuntimeError, match="ETI_AUTH_SECRET"):
        AuthManager(tmp_path / "auth.sqlite3", secret="", registration_key="invite-key-123456")
    auth = AuthManager(tmp_path / "auth.sqlite3", secret="s" * 40, registration_key=None)
    with pytest.raises(RegistrationDisabled):
        auth.register("trader01", "a-long-password-123", "anything")


def test_username_and_token_policy_validation(tmp_path):
    auth = AuthManager(tmp_path / "auth.sqlite3", secret="s" * 40,
                       registration_key="invite-key-123456")
    with pytest.raises(ValueError, match="username"):
        auth.register("x", "a-long-password-123", "invite-key-123456")
    with pytest.raises(ValueError, match="token_ttl"):
        AuthManager(tmp_path / "bad.sqlite3", secret="s" * 40,
                    registration_key="invite-key-123456", token_ttl_seconds=10)


def test_persistent_rate_limit_is_atomic_and_hides_subject(tmp_path):
    from eti.auth import RateLimitExceeded

    path = tmp_path / "auth.sqlite3"
    auth = AuthManager(path, secret="s" * 40, registration_key="invite-key-123456")
    auth.consume_rate_limit("login", "192.0.2.55", limit=2, window_seconds=60, now=100)
    auth.consume_rate_limit("login", "192.0.2.55", limit=2, window_seconds=60, now=101)
    with pytest.raises(RateLimitExceeded) as exc:
        auth.consume_rate_limit("login", "192.0.2.55", limit=2, window_seconds=60, now=102)
    assert exc.value.retry_after == 58
    with __import__("sqlite3").connect(path) as conn:
        rows = conn.execute("SELECT bucket_key FROM auth_rate_limits").fetchall()
    assert len(rows) == 2
    assert "192.0.2.55" not in rows[0][0]


def test_rate_limit_window_expires(tmp_path):
    from eti.auth import RateLimitExceeded

    auth = AuthManager(tmp_path / "auth.sqlite3", secret="s" * 40,
                       registration_key="invite-key-123456")
    auth.consume_rate_limit("login", "user-a", limit=1, window_seconds=60, now=100)
    with pytest.raises(RateLimitExceeded):
        auth.consume_rate_limit("login", "user-a", limit=1, window_seconds=60, now=120)
    auth.consume_rate_limit("login", "user-a", limit=1, window_seconds=60, now=161)


def test_password_change_requires_current_password_and_persists(tmp_path):
    auth = AuthManager(tmp_path / "auth.sqlite3", secret="s" * 40,
                       registration_key="invite-key-123456")
    account = auth.register("trader01", "old-password-123", "invite-key-123456")
    with pytest.raises(AuthenticationError, match="current password"):
        auth.change_password(account["id"], "wrong-password-123", "new-password-456")
    auth.change_password(account["id"], "old-password-123", "new-password-456")
    with pytest.raises(ValueError, match="invalid username or password"):
        auth.authenticate("trader01", "old-password-123")
    assert auth.authenticate("trader01", "new-password-456") == account
    reopened = AuthManager(tmp_path / "auth.sqlite3", secret="s" * 40,
                           registration_key="invite-key-123456")
    assert reopened.authenticate("trader01", "new-password-456") == account
