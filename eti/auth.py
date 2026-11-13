"""Invite-only account authentication for the ETI research API.

Uses scrypt password hashing and HMAC-SHA256 signed, short-lived bearer tokens.
Secrets are supplied only through environment variables; no secret is generated
or persisted automatically. This is a foundation, not a complete public-Internet
identity service (email verification and edge rate limiting are still required).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


_USERNAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_TOKEN_ISSUER = "eti-research-core"


class AuthenticationError(ValueError):
    """Invalid credentials or bearer token."""


class RegistrationDisabled(RuntimeError):
    """Required server-side registration configuration is absent."""


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


class AuthManager:
    """SQLite account store with invite-only registration and signed sessions."""

    def __init__(
        self,
        database_path: str | os.PathLike[str] | None = None,
        *,
        secret: str | None = None,
        registration_key: str | None = None,
        token_ttl_seconds: int = 900,
    ):
        self.database_path = str(
            database_path or os.environ.get("ETI_DATABASE_PATH", "data/eti.sqlite3")
        )
        self.secret = secret if secret is not None else os.environ.get("ETI_AUTH_SECRET")
        self.registration_key = (
            registration_key if registration_key is not None
            else os.environ.get("ETI_REGISTRATION_KEY")
        )
        if not isinstance(self.secret, str) or len(self.secret.encode("utf-8")) < 32:
            raise RuntimeError("ETI_AUTH_SECRET must be configured with at least 32 bytes")
        if isinstance(token_ttl_seconds, bool) or not 60 <= token_ttl_seconds <= 3600:
            raise ValueError("token_ttl_seconds must be between 60 and 3600")
        self.token_ttl_seconds = token_ttl_seconds
        Path(self.database_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS auth_accounts (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    created_at INTEGER NOT NULL
                )"""
            )

    @staticmethod
    def _validate_username(username: str) -> str:
        if not isinstance(username, str):
            raise ValueError("username must be a string")
        normalized = username.strip().lower()
        if not _USERNAME.fullmatch(normalized):
            raise ValueError("username must be 3-32 characters: letters, numbers, . _ -")
        return normalized

    @staticmethod
    def _password_bytes(password: str) -> bytes:
        if not isinstance(password, str) or not 12 <= len(password) <= 1024:
            raise ValueError("password must contain 12-1024 characters")
        return password.encode("utf-8")

    @staticmethod
    def _hash_password(password: bytes, salt: bytes) -> bytes:
        return hashlib.scrypt(password, salt=salt, n=_SCRYPT_N, r=_SCRYPT_R,
                              p=_SCRYPT_P, dklen=32)

    def register(self, username: str, password: str, invite_code: str) -> dict[str, str]:
        if not isinstance(self.registration_key, str) or len(self.registration_key) < 16:
            raise RegistrationDisabled("Invite-only registration is not configured")
        if not isinstance(invite_code, str) or not hmac.compare_digest(
            invite_code.encode("utf-8"), self.registration_key.encode("utf-8")
        ):
            raise AuthenticationError("invalid registration credentials")
        username = self._validate_username(username)
        password_bytes = self._password_bytes(password)
        salt = secrets.token_bytes(16)
        password_hash = self._hash_password(password_bytes, salt)
        account = {"id": str(uuid4()), "username": username}
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute(
                    "INSERT INTO auth_accounts(id, username, password_salt, password_hash, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (account["id"], username, salt, password_hash, int(time.time())),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("username is already registered") from exc
        return account

    def authenticate(self, username: str, password: str) -> dict[str, str]:
        username = self._validate_username(username)
        password_bytes = self._password_bytes(password)
        with sqlite3.connect(self.database_path) as conn:
            row = conn.execute(
                "SELECT id, username, password_salt, password_hash "
                "FROM auth_accounts WHERE username = ?", (username,)
            ).fetchone()
        if row is None:
            # Do a same-cost hash for unknown usernames to reduce enumeration timing.
            self._hash_password(password_bytes, b"ETI-auth-salt!!")
            raise AuthenticationError("invalid username or password")
        candidate = self._hash_password(password_bytes, row[2])
        if not hmac.compare_digest(candidate, row[3]):
            raise AuthenticationError("invalid username or password")
        return {"id": row[0], "username": row[1]}

    def issue_token(self, account: dict[str, str], *, now: int | None = None) -> str:
        issued = int(time.time()) if now is None else int(now)
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "iss": _TOKEN_ISSUER, "sub": account["id"],
            "username": account["username"], "iat": issued,
            "exp": issued + self.token_ttl_seconds,
        }
        signing_input = _b64encode(_json_bytes(header)) + "." + _b64encode(_json_bytes(payload))
        signature = hmac.new(self.secret.encode("utf-8"), signing_input.encode("ascii"),
                             hashlib.sha256).digest()
        return signing_input + "." + _b64encode(signature)

    def verify_token(self, token: str, *, now: int | None = None) -> dict[str, str]:
        try:
            if not isinstance(token, str) or len(token) > 4096:
                raise ValueError("invalid token")
            header_part, payload_part, signature_part = token.split(".")
            signing_input = header_part + "." + payload_part
            expected = hmac.new(self.secret.encode("utf-8"), signing_input.encode("ascii"),
                                hashlib.sha256).digest()
            supplied = _b64decode(signature_part)
            if not hmac.compare_digest(expected, supplied):
                raise ValueError("invalid signature")
            header = json.loads(_b64decode(header_part))
            payload = json.loads(_b64decode(payload_part))
            current = int(time.time()) if now is None else int(now)
            if header != {"alg": "HS256", "typ": "JWT"}:
                raise ValueError("unsupported token")
            if payload.get("iss") != _TOKEN_ISSUER:
                raise ValueError("invalid issuer")
            if not isinstance(payload.get("sub"), str) or not payload["sub"]:
                raise ValueError("invalid subject")
            if not isinstance(payload.get("username"), str):
                raise ValueError("invalid username")
            if isinstance(payload.get("exp"), bool) or not isinstance(payload.get("exp"), int):
                raise ValueError("invalid expiry")
            if isinstance(payload.get("iat"), bool) or not isinstance(payload.get("iat"), int):
                raise ValueError("invalid issued-at")
            if payload["exp"] <= current or payload["iat"] > current + 30:
                raise ValueError("expired or not-yet-valid token")
            if payload["exp"] - payload["iat"] > 3600:
                raise ValueError("token lifetime exceeds policy")
            return {"id": payload["sub"], "username": payload["username"]}
        except (ValueError, TypeError, KeyError, UnicodeError, json.JSONDecodeError) as exc:
            raise AuthenticationError("invalid or expired bearer token") from exc
