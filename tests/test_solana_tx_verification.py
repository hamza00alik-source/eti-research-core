import pytest

from eti.solana_tx_verification import verify_finalized_transaction_for_mint


_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58(data):
    number = int.from_bytes(data, "big")
    encoded = ""
    while number:
        number, remainder = divmod(number, 58)
        encoded = _ALPHABET[remainder] + encoded
    leading = len(data) - len(data.lstrip(b"\\x00"))
    return "1" * leading + (encoded or ("" if leading else "1"))


SIGNATURE = _b58(bytes(range(1, 65)))
MINT = _b58(bytes(range(65, 97)))
OTHER_MINT = _b58(bytes(range(97, 129)))


class FakeRpc:
    def __init__(self, status=None, transaction=None):
        self.status = status
        self.transaction = transaction
        self.calls = []

    def call(self, method, params):
        self.calls.append((method, params))
        if method == "getSignatureStatuses":
            return {"value": [self.status]}
        if method == "getTransaction":
            return self.transaction
        raise AssertionError(method)


def _status(**overrides):
    value = {
        "slot": 12345,
        "err": None,
        "confirmationStatus": "finalized",
    }
    value.update(overrides)
    return value


def _transaction(mint=MINT, **overrides):
    value = {
        "slot": 12345,
        "blockTime": 1794672000,
        "transaction": {"signatures": [SIGNATURE]},
        "meta": {
            "err": None,
            "preTokenBalances": [],
            "postTokenBalances": [{"mint": mint, "uiTokenAmount": {"amount": "10"}}],
        },
    }
    value.update(overrides)
    return value


def test_finalized_successful_transaction_is_bound_to_signature_and_mint():
    rpc = FakeRpc(_status(), _transaction())
    result = verify_finalized_transaction_for_mint(rpc, SIGNATURE, MINT)

    assert result["status"] == "FINALIZED_MINT_TRANSACTION_VERIFIED"
    assert result["verified"] is True
    assert result["slot"] == 12345
    assert result["mint_balance_entries"] == 1
    assert result["limitations"]
    assert rpc.calls[0][0] == "getSignatureStatuses"
    assert rpc.calls[1][0] == "getTransaction"
    assert rpc.calls[1][1][1]["commitment"] == "finalized"


def test_unknown_signature_is_not_verified():
    result = verify_finalized_transaction_for_mint(
        FakeRpc(None, None), SIGNATURE, MINT
    )
    assert result["status"] == "NOT_FOUND"
    assert result["verified"] is False


def test_confirmed_but_not_finalized_transaction_is_not_verified():
    result = verify_finalized_transaction_for_mint(
        FakeRpc(_status(confirmationStatus="confirmed"), None), SIGNATURE, MINT
    )
    assert result["status"] == "NOT_FINALIZED"
    assert result["verified"] is False


def test_failed_transaction_is_not_verified():
    result = verify_finalized_transaction_for_mint(
        FakeRpc(_status(err={"InstructionError": [0, "Custom"]}), None), SIGNATURE, MINT
    )
    assert result["status"] == "TRANSACTION_FAILED"
    assert result["verified"] is False


def test_transaction_without_requested_mint_is_not_verified():
    result = verify_finalized_transaction_for_mint(
        FakeRpc(_status(), _transaction(OTHER_MINT)), SIGNATURE, MINT
    )
    assert result["status"] == "MINT_NOT_PRESENT"
    assert result["verified"] is False


def test_transaction_slot_must_match_signature_status_slot():
    result = verify_finalized_transaction_for_mint(
        FakeRpc(_status(), _transaction(slot=12346)), SIGNATURE, MINT
    )
    assert result["status"] == "SLOT_MISMATCH"
    assert result["verified"] is False


@pytest.mark.parametrize("signature,mint", [
    ("not-a-signature", MINT),
    (SIGNATURE, "not-a-mint"),
])
def test_invalid_signature_or_mint_rejected(signature, mint):
    with pytest.raises(ValueError):
        verify_finalized_transaction_for_mint(FakeRpc(), signature, mint)


def test_transaction_response_must_include_requested_signature():
    tx = _transaction()
    tx["transaction"]["signatures"] = [_b58(bytes([3] * 64))]
    result = verify_finalized_transaction_for_mint(FakeRpc(_status(), tx), SIGNATURE, MINT)
    assert result["status"] == "TRANSACTION_DETAILS_INVALID"
    assert result["verified"] is False
