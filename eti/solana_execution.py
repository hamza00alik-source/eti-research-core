"""Solana mainnet swap execution foundation via Jupiter Swap API v2.

This module deliberately does not load or store private keys. Production signing
must be supplied by an isolated signer/KMS integration. Live submission is
fail-closed until the project-level safety review explicitly changes the gate.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

JUPITER_BASE_URL = "https://api.jup.ag/swap/v2"
SOLANA_LIVE_EXECUTION_ENABLED = False


class SolanaExecutionError(RuntimeError):
    """Raised when a Solana swap cannot safely proceed."""


class JsonTransport(Protocol):
    def get(self, url: str, headers: Mapping[str, str]) -> Mapping[str, object]: ...
    def post(self, url: str, body: Mapping[str, object], headers: Mapping[str, str]) -> Mapping[str, object]: ...


class UrllibJsonTransport:
    """Small stdlib transport; injectable in tests."""

    def get(self, url: str, headers: Mapping[str, str]) -> Mapping[str, object]:
        request = urllib.request.Request(url, headers=dict(headers), method="GET")
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))

    def post(self, url: str, body: Mapping[str, object], headers: Mapping[str, str]) -> Mapping[str, object]:
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            url, data=payload, headers={**dict(headers), "Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))


@dataclass(frozen=True)
class SwapReceipt:
    status: str
    signature: str
    input_amount: str
    output_amount: str
    router: str
    request_id: str


class JupiterSwapClient:
    """Quote, sign via an injected custody boundary, and execute Jupiter orders.

    The signer receives the decoded unsigned transaction and must return the
    fully signed transaction bytes. It must never expose key material to ETI.
    """

    def __init__(
        self,
        wallet_address: str,
        signer: Callable[[bytes], bytes],
        *,
        api_key: str | None = None,
        transport: JsonTransport | None = None,
        base_url: str = JUPITER_BASE_URL,
    ) -> None:
        if not wallet_address or len(wallet_address) < 32:
            raise ValueError("A valid Solana taker wallet address is required")
        self.wallet_address = wallet_address
        self.signer = signer
        self.api_key = api_key or os.getenv("JUPITER_API_KEY")
        self.transport = transport or UrllibJsonTransport()
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def get_order(self, input_mint: str, output_mint: str, amount: int) -> Mapping[str, object]:
        if not input_mint or not output_mint or input_mint == output_mint:
            raise ValueError("Distinct input and output mint addresses are required")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            raise ValueError("amount must be a positive integer in the input token's base units")
        query = urllib.parse.urlencode({
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "taker": self.wallet_address,
        })
        order = self.transport.get(f"{self.base_url}/order?{query}", self._headers())
        transaction = order.get("transaction")
        request_id = order.get("requestId")
        out_amount = order.get("outAmount")
        if not isinstance(transaction, str) or not transaction:
            raise SolanaExecutionError("Jupiter did not return a signable transaction")
        if not isinstance(request_id, str) or not request_id:
            raise SolanaExecutionError("Jupiter order is missing requestId")
        if not isinstance(out_amount, str) or not out_amount.isdigit() or int(out_amount) <= 0:
            raise SolanaExecutionError("Jupiter order has an invalid output amount")
        return order

    def execute_order(self, order: Mapping[str, object], *, live_authorized: bool = False) -> SwapReceipt:
        if not SOLANA_LIVE_EXECUTION_ENABLED:
            raise SolanaExecutionError("Solana live execution is hard-disabled in this build")
        if not live_authorized:
            raise SolanaExecutionError("Explicit owner live-trading authorization is required")
        transaction = order.get("transaction")
        request_id = order.get("requestId")
        if not isinstance(transaction, str) or not isinstance(request_id, str):
            raise SolanaExecutionError("Invalid Jupiter order")
        try:
            unsigned = base64.b64decode(transaction, validate=True)
        except (ValueError, TypeError) as exc:
            raise SolanaExecutionError("Jupiter transaction is not valid base64") from exc
        signed = self.signer(unsigned)
        if not isinstance(signed, bytes) or not signed:
            raise SolanaExecutionError("Signer did not return signed transaction bytes")
        result = self.transport.post(
            f"{self.base_url}/execute",
            {"signedTransaction": base64.b64encode(signed).decode("ascii"), "requestId": request_id},
            self._headers(),
        )
        if result.get("status") != "Success":
            raise SolanaExecutionError(f"Jupiter execution failed: {result.get('error') or result.get('code')}")
        signature = result.get("signature")
        input_amount = result.get("inputAmountResult") or result.get("totalInputAmount")
        output_amount = result.get("outputAmountResult") or result.get("totalOutputAmount")
        if not all(isinstance(value, str) and value for value in (signature, input_amount, output_amount)):
            raise SolanaExecutionError("Jupiter success response is missing transaction receipt fields")
        return SwapReceipt(
            status="Success",
            signature=signature,
            input_amount=input_amount,
            output_amount=output_amount,
            router=str(order.get("router") or "unknown"),
            request_id=request_id,
        )
