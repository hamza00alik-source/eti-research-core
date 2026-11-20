"""Read-only RugCheck report client for Solana token intelligence.

The response is preserved as provider evidence; normalization into ETI's
policy schema is deliberately explicit and rejects absent critical fields.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Mapping, Protocol
from urllib.parse import quote


RUGCHECK_BASE_URL = "https://api.rugcheck.xyz"


class RugCheckError(RuntimeError):
    """Raised when RugCheck evidence cannot be safely retrieved or parsed."""


class JsonGetTransport(Protocol):
    def get(self, url: str, headers: Mapping[str, str]) -> Mapping[str, Any]: ...


class UrllibGetTransport:
    def get(self, url: str, headers: Mapping[str, str]) -> Mapping[str, Any]:
        request = urllib.request.Request(url, headers=dict(headers), method="GET")
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise RugCheckError("RugCheck returned a non-object response")
        return payload


class RugCheckClient:
    """Fetch full read-only token reports; this client has no wallet capability."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        transport: JsonGetTransport | None = None,
        base_url: str = RUGCHECK_BASE_URL,
    ) -> None:
        self.api_key = api_key or os.getenv("RUGCHECK_API_KEY")
        self.transport = transport or UrllibGetTransport()
        self.base_url = base_url.rstrip("/")

    def get_token_report(self, mint: str) -> Mapping[str, Any]:
        if not isinstance(mint, str) or not mint.strip() or len(mint.strip()) > 64:
            raise ValueError("A non-empty Solana mint address is required")
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-KEY"] = self.api_key
        url = f"{self.base_url}/v1/tokens/{quote(mint.strip(), safe='')}/report"
        try:
            report = self.transport.get(url, headers)
        except Exception as exc:
            raise RugCheckError(f"RugCheck report request failed: {type(exc).__name__}") from exc
        if not isinstance(report, Mapping):
            raise RugCheckError("RugCheck returned an invalid report")
        returned_mint = report.get("mint")
        if returned_mint is not None and returned_mint != mint.strip():
            raise RugCheckError("RugCheck response mint does not match the requested mint")
        return dict(report)

    @staticmethod
    def authority_state(report: Mapping[str, Any], field: str) -> bool | None:
        """RugCheck null authority means revoked; absent/malformed means unknown."""
        token = report.get("token")
        if not isinstance(token, Mapping) or field not in token:
            return None
        value = token[field]
        if value is None:
            return False
        if isinstance(value, str) and value.strip():
            return True
        return None

    @staticmethod
    def top_holder_fraction(report: Mapping[str, Any]) -> float | None:
        """Convert RugCheck topHoldersPct (percentage points) to a 0..1 fraction."""
        value = report.get("topHoldersPct")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not 0 <= value <= 100:
            return None
        return float(value) / 100.0
