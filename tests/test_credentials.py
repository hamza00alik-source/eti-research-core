import base64
import sys
from types import SimpleNamespace

import pytest

from eti.credentials import CredentialVault, CredentialVaultError, verify_exchange_credentials


def test_aes_gcm_vault_round_trip_and_context_binding():
    key = base64.urlsafe_b64encode(b"k" * 32).decode("ascii")
    vault = CredentialVault(key)
    encrypted = vault.encrypt("example-api-secret", "owner-a", "binance", "api_secret")
    assert "example-api-secret" not in encrypted
    assert vault.decrypt(encrypted, "owner-a", "binance", "api_secret") == "example-api-secret"
    with pytest.raises(CredentialVaultError, match="failed authentication"):
        vault.decrypt(encrypted, "owner-b", "binance", "api_secret")
    with pytest.raises(CredentialVaultError, match="failed authentication"):
        vault.decrypt(encrypted, "owner-a", "mexc", "api_secret")


def test_vault_rejects_missing_or_wrong_length_key():
    with pytest.raises(CredentialVaultError, match="not configured"):
        CredentialVault("")
    with pytest.raises(CredentialVaultError, match="exactly 32 bytes"):
        CredentialVault(base64.urlsafe_b64encode(b"short").decode("ascii"))


def test_exchange_verification_checks_read_access_without_order_creation(monkeypatch):
    calls = {"balance": 0, "orders": 0}

    class FakeExchange:
        def __init__(self, config):
            assert config["apiKey"] == "api-key-123456"
            assert config["secret"] == "api-secret-123456"
        def fetch_balance(self):
            calls["balance"] += 1
            return {"total": {"USDT": 10}}
        def create_order(self, *args, **kwargs):
            calls["orders"] += 1
            raise AssertionError("verification must never place an order")

    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(binance=FakeExchange))
    result = verify_exchange_credentials("binance", "api-key-123456", "api-secret-123456")
    assert result == {
        "venue": "binance",
        "read_access_verified": True,
        "trading_permission_verified": False,
    }
    assert calls == {"balance": 1, "orders": 0}


def test_exchange_verification_does_not_expose_provider_exception(monkeypatch):
    class FakeExchange:
        def __init__(self, config):
            pass
        def fetch_balance(self):
            raise RuntimeError("secret=do-not-leak")

    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(mexc=FakeExchange))
    with pytest.raises(CredentialVaultError) as exc:
        verify_exchange_credentials("mexc", "api-key-123456", "api-secret-123456")
    assert "do-not-leak" not in str(exc.value)
