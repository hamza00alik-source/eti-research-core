"""Encrypted owner-managed exchange credential vault.

Credentials are encrypted at rest with AES-256-GCM. The encryption key is
provided only through ETI_CREDENTIAL_ENCRYPTION_KEY and is never stored in
SQLite or returned by an API route.
"""
from __future__ import annotations

import base64
import os
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class CredentialVaultError(RuntimeError):
    pass


class CredentialVault:
    def __init__(self, key: str | None = None):
        encoded = key if key is not None else os.environ.get("ETI_CREDENTIAL_ENCRYPTION_KEY", "")
        if not encoded:
            raise CredentialVaultError("ETI_CREDENTIAL_ENCRYPTION_KEY is not configured")
        try:
            raw = base64.urlsafe_b64decode(encoded.encode("ascii"))
        except Exception as exc:
            raise CredentialVaultError("ETI_CREDENTIAL_ENCRYPTION_KEY must be URL-safe base64") from exc
        if len(raw) != 32:
            raise CredentialVaultError("ETI_CREDENTIAL_ENCRYPTION_KEY must decode to exactly 32 bytes")
        self._cipher = AESGCM(raw)

    @staticmethod
    def _aad(owner_id: str, venue: str, field: str) -> bytes:
        return f"eti-exchange-credential:v1:{owner_id}:{venue}:{field}".encode("utf-8")

    def encrypt(self, plaintext: str, owner_id: str, venue: str, field: str) -> str:
        if not isinstance(plaintext, str) or not plaintext:
            raise ValueError("credential value must be a non-empty string")
        nonce = os.urandom(12)
        ciphertext = self._cipher.encrypt(nonce, plaintext.encode("utf-8"), self._aad(owner_id, venue, field))
        return base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")

    def decrypt(self, encoded: str, owner_id: str, venue: str, field: str) -> str:
        try:
            raw = base64.urlsafe_b64decode(encoded.encode("ascii"))
            if len(raw) < 29:
                raise ValueError("ciphertext is too short")
            return self._cipher.decrypt(raw[:12], raw[12:], self._aad(owner_id, venue, field)).decode("utf-8")
        except (InvalidTag, ValueError, UnicodeDecodeError) as exc:
            raise CredentialVaultError("stored exchange credential failed authentication") from exc


def verify_exchange_credentials(venue: str, api_key: str, api_secret: str) -> dict[str, Any]:
    """Verify private read access without placing an order or returning balances."""
    if venue not in {"binance", "mexc"}:
        raise ValueError("Only Binance and MEXC spot accounts are supported")
    if not isinstance(api_key, str) or not 8 <= len(api_key) <= 256:
        raise ValueError("API key length is invalid")
    if not isinstance(api_secret, str) or not 8 <= len(api_secret) <= 256:
        raise ValueError("API secret length is invalid")
    try:
        import ccxt
        exchange_class = getattr(ccxt, venue)
        exchange = exchange_class({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "timeout": 15000,
        })
        try:
            balance = exchange.fetch_balance()
            if not isinstance(balance, dict):
                raise RuntimeError("Exchange returned an invalid account response")
        finally:
            close = getattr(exchange, "close", None)
            if callable(close):
                close()
    except Exception as exc:
        # Never return provider exception strings: they can contain request data.
        raise CredentialVaultError("Exchange connection or private read permission check failed") from exc
    return {"venue": venue, "read_access_verified": True, "trading_permission_verified": False}
