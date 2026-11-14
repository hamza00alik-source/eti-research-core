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
    expected_pool_address: str | None = None,
    wallet_address: str | None = None,
) -> dict[str, Any]:
    """Verify a finalized successful mint exit, pool reference, and wallet debit."""
    if not _valid_signature(signature):
        raise ValueError("signature must be a valid 64-byte base58 transaction signature")
    if not _valid_pubkey(mint):
        raise ValueError("mint must be a valid 32-byte base58 public key")
    if expected_pool_address is None or not _valid_pubkey(expected_pool_address):
        raise ValueError("expected_pool_address must be a valid 32-byte base58 public key")
    if wallet_address is None or not _valid_pubkey(wallet_address):
        raise ValueError("wallet_address must be a valid 32-byte base58 public key")

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

    message = tx.get("message")
    account_keys = message.get("accountKeys") if isinstance(message, dict) else None
    key_values = []
    if isinstance(account_keys, list):
        for key in account_keys:
            if isinstance(key, str):
                key_values.append(key)
            elif isinstance(key, dict) and isinstance(key.get("pubkey"), str):
                key_values.append(key["pubkey"])
    if wallet_address not in key_values:
        return {
            "status": "WALLET_NOT_REFERENCED",
            "verified": False,
            "signature": signature,
            "mint": mint,
            "wallet_address": wallet_address,
            "slot": transaction.get("slot"),
            "reason": "Wallet address is not present in transaction account keys",
        }
    if expected_pool_address not in key_values:
        return {
            "status": "EXPECTED_POOL_NOT_REFERENCED",
            "verified": False,
            "signature": signature,
            "mint": mint,
            "expected_pool_address": expected_pool_address,
            "slot": transaction.get("slot"),
            "reason": "Expected pool address is not present in transaction account keys",
        }

    mint_balance_rows = []
    for field in ("preTokenBalances", "postTokenBalances"):
        entries = meta.get(field)
        if isinstance(entries, list):
            mint_balance_rows.extend(
                item for item in entries
                if isinstance(item, dict) and item.get("mint") == mint
            )
    if not mint_balance_rows:
        return {
            "status": "MINT_NOT_PRESENT",
            "verified": False,
            "signature": signature,
            "mint": mint,
            "slot": transaction.get("slot"),
            "reason": "Transaction token balances do not reference the requested mint",
        }

    def wallet_amount(field):
        entries = meta.get(field)
        if not isinstance(entries, list):
            return None
        total = 0
        found = False
        for item in entries:
            if not isinstance(item, dict) or item.get("mint") != mint:
                continue
            if not isinstance(item.get("owner"), str) or not item.get("owner"):
                return None
            if item.get("owner") != wallet_address:
                continue
            amount = item.get("uiTokenAmount")
            raw = amount.get("amount") if isinstance(amount, dict) else None
            if not isinstance(raw, str) or not raw.isdigit():
                return None
            total += int(raw)
            found = True
        return total if found else 0

    pre_amount = wallet_amount("preTokenBalances")
    post_amount = wallet_amount("postTokenBalances")
    if pre_amount is None or post_amount is None:
        return {
            "status": "WALLET_BALANCE_EVIDENCE_UNAVAILABLE",
            "verified": False,
            "signature": signature,
            "mint": mint,
            "wallet_address": wallet_address,
            "slot": transaction.get("slot"),
            "reason": "RPC did not provide parseable owner-attributed token balances for both sides",
        }
    if post_amount >= pre_amount:
        return {
            "status": "WALLET_TOKEN_BALANCE_NOT_DEBITED",
            "verified": False,
            "signature": signature,
            "mint": mint,
            "wallet_address": wallet_address,
            "pre_token_amount": str(pre_amount),
            "post_token_amount": str(post_amount),
            "slot": transaction.get("slot"),
            "reason": "Wallet's aggregate balance for the mint did not decrease",
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
        "expected_pool_address": expected_pool_address,
        "wallet_address": wallet_address,
        "pre_token_amount_raw": str(pre_amount),
        "post_token_amount_raw": str(post_amount),
        "wallet_token_debit_raw": str(pre_amount - post_amount),
        "direction": "wallet_mint_balance_decreased",
        "limitations": [
            "Confirms finalized successful transaction inclusion, expected pool account-key reference, and owner-attributed wallet mint-balance decrease.",
            "Pool account-key presence alone does not prove a specific pool instruction caused the balance change.",
            "Does not establish fill price, complete transaction costs, or realized P&L.",
            "RPC provider responses are trusted as the queried public node's view and are not independently cross-validated.",
        ],
    }
