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
