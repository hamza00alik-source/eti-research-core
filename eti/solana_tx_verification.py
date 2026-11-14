"""Read-only public Solana RPC verification for transaction receipts.

This module verifies finalized transaction inclusion and mint association. It
does not establish trade direction, fill price, wallet ownership, or P&L.
"""
from __future__ import annotations

from typing import Any


_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _base58_decode(value: str) -> bytes:
    number = 0
    for char in value:
        try:
            digit = _ALPHABET.index(char)
        except ValueError as exc:
            raise ValueError("invalid base58 value") from exc
        number = number * 58 + digit
    raw = number.to_bytes((number.bit_length() + 7) // 8, "big") if number else b""
    return b"\x00" * (len(value) - len(value.lstrip("1"))) + raw


def _valid_pubkey(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return len(_base58_decode(value)) == 32
    except ValueError:
        return False


def _valid_signature(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return len(_base58_decode(value)) == 64
    except ValueError:
        return False


def verify_finalized_transaction_for_mint(
    rpc: Any, signature: str, mint: str,
) -> dict[str, Any]:
    """Verify finalized, successful transaction inclusion and mint association."""
    if not _valid_signature(signature):
        raise ValueError("signature must be a valid 64-byte base58 transaction signature")
    if not _valid_pubkey(mint):
        raise ValueError("mint must be a valid 32-byte base58 public key")

    status_result = rpc.call(
        "getSignatureStatuses",
        [[signature], {"searchTransactionHistory": True}],
    )
    values = status_result.get("value") if isinstance(status_result, dict) else None
    status = values[0] if isinstance(values, list) and values else None
    if not isinstance(status, dict):
        return {
            "status": "NOT_FOUND",
            "verified": False,
            "signature": signature,
            "mint": mint,
            "reason": "RPC has no transaction status for this signature",
        }
    if status.get("err") is not None:
        return {
            "status": "TRANSACTION_FAILED",
            "verified": False,
            "signature": signature,
            "mint": mint,
            "slot": status.get("slot"),
            "reason": "RPC reports a transaction execution error",
        }
    if status.get("confirmationStatus") != "finalized":
        return {
            "status": "NOT_FINALIZED",
            "verified": False,
            "signature": signature,
            "mint": mint,
            "slot": status.get("slot"),
            "confirmation_status": status.get("confirmationStatus"),
            "reason": "Transaction is not finalized",
        }

    transaction = rpc.call(
        "getTransaction",
        [signature, {
            "encoding": "jsonParsed",
            "commitment": "finalized",
            "maxSupportedTransactionVersion": 0,
        }],
    )
    if not isinstance(transaction, dict):
        return {
            "status": "TRANSACTION_DETAILS_UNAVAILABLE",
            "verified": False,
            "signature": signature,
            "mint": mint,
            "slot": status.get("slot"),
            "reason": "Finalized transaction details are unavailable",
        }

    meta = transaction.get("meta")
    tx = transaction.get("transaction")
    signatures = tx.get("signatures") if isinstance(tx, dict) else None
    if not isinstance(meta, dict) or not isinstance(signatures, list) or signature not in signatures:
        return {
            "status": "TRANSACTION_DETAILS_INVALID",
            "verified": False,
            "signature": signature,
            "mint": mint,
            "slot": transaction.get("slot"),
            "reason": "Transaction response did not bind the requested signature",
        }
    if meta.get("err") is not None:
        return {
            "status": "TRANSACTION_FAILED",
            "verified": False,
            "signature": signature,
            "mint": mint,
            "slot": transaction.get("slot"),
            "reason": "Finalized transaction metadata reports an execution error",
        }

    balances = []
    for field in ("preTokenBalances", "postTokenBalances"):
        values = meta.get(field)
        if isinstance(values, list):
            balances.extend(
                item for item in values
                if isinstance(item, dict) and item.get("mint") == mint
            )
    if not balances:
        return {
            "status": "MINT_NOT_PRESENT",
            "verified": False,
            "signature": signature,
            "mint": mint,
            "slot": transaction.get("slot"),
            "reason": "Transaction token balances do not reference the requested mint",
        }

    status_slot = status.get("slot")
    transaction_slot = transaction.get("slot")
    if status_slot is not None and transaction_slot != status_slot:
        return {
            "status": "SLOT_MISMATCH",
            "verified": False,
            "signature": signature,
            "mint": mint,
            "slot": transaction_slot,
            "reason": "Signature status and transaction detail slots disagree",
        }

    return {
        "status": "FINALIZED_MINT_TRANSACTION_VERIFIED",
        "verified": True,
        "signature": signature,
        "mint": mint,
        "slot": transaction_slot,
        "block_time": transaction.get("blockTime"),
        "commitment": "finalized",
        "mint_balance_entries": len(balances),
        "limitations": [
            "Confirms finalized successful transaction inclusion and mint presence only.",
            "Does not prove this transaction was an entry or exit for the strategy.",
            "Does not establish wallet ownership, swap direction, fill price, fees, or realized P&L.",
            "RPC provider responses are trusted as the queried public node's view and are not independently cross-validated.",
        ],
    }
