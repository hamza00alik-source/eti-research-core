import pytest

from eti.solana_tx_verification import verify_finalized_transaction_for_mint


_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58(data):
    number = int.from_bytes(data, "big")
    encoded = ""
    while number:
        number, remainder = divmod(number, 58)
        encoded = _ALPHABET[remainder] + encoded
    leading = len(data) - len(data.lstrip(bytes([0])))
    return "1" * leading + (encoded or ("" if leading else "1"))


SIGNATURE = _b58(bytes(range(1, 65)))
MINT = _b58(bytes(range(65, 97)))
OTHER_MINT = _b58(bytes(range(97, 129)))
POOL = _b58(bytes(range(129, 161)))
WALLET = _b58(bytes(range(161, 193)))
OTHER_SIGNATURE = _b58(bytes([3] * 64))


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
    value = {"slot": 12345, "err": None, "confirmationStatus": "finalized"}
    value.update(overrides)
    return value


def _balance(mint, amount):
    return {
        "mint": mint,
        "owner": WALLET,
        "uiTokenAmount": {"amount": str(amount), "decimals": 6, "uiAmount": amount / 1_000_000},
    }


def _transaction(mint=MINT, **overrides):
    value = {
        "slot": 12345,
        "blockTime": 1794672000,
        "transaction": {
            "signatures": [SIGNATURE],
            "message": {"accountKeys": [{"pubkey": POOL}, {"pubkey": WALLET}]},
        },
        "meta": {
            "err": None,
            "preTokenBalances": [_balance(mint, 100)],
            "postTokenBalances": [_balance(mint, 50)],
        },
    }
    value.update(overrides)
    return value


def _verify(rpc, signature=SIGNATURE, mint=MINT, pool=POOL, wallet=WALLET):
    return verify_finalized_transaction_for_mint(
        rpc, signature, mint, expected_pool_address=pool, wallet_address=wallet
    )


def test_finalized_successful_wallet_debit_and_expected_pool_are_verified():
    rpc = FakeRpc(_status(), _transaction())
    result = _verify(rpc)

    assert result["status"] == "FINALIZED_MINT_TRANSACTION_VERIFIED"
    assert result["verified"] is True
    assert result["slot"] == 12345
    assert result["wallet_token_debit_raw"] == "50"
    assert result["direction"] == "wallet_mint_balance_decreased"
    assert result["limitations"]
    assert rpc.calls[0][0] == "getSignatureStatuses"
    assert rpc.calls[1][0] == "getTransaction"
    assert rpc.calls[1][1][1]["commitment"] == "finalized"


def test_unknown_signature_is_not_verified():
    result = _verify(FakeRpc(None, None))
    assert result["status"] == "NOT_FOUND"
    assert result["verified"] is False


def test_confirmed_but_not_finalized_transaction_is_not_verified():
    result = _verify(FakeRpc(_status(confirmationStatus="confirmed"), None))
    assert result["status"] == "NOT_FINALIZED"
    assert result["verified"] is False


def test_failed_transaction_is_not_verified():
    result = _verify(FakeRpc(_status(err={"InstructionError": [0, "Custom"]}), None))
    assert result["status"] == "TRANSACTION_FAILED"
    assert result["verified"] is False


def test_transaction_without_requested_mint_is_not_verified():
    result = _verify(FakeRpc(_status(), _transaction(OTHER_MINT)))
    assert result["status"] == "MINT_NOT_PRESENT"
    assert result["verified"] is False


def test_expected_pool_must_be_referenced_by_transaction():
    tx = _transaction()
    tx["transaction"]["message"]["accountKeys"] = [{"pubkey": WALLET}]
    result = _verify(FakeRpc(_status(), tx))
    assert result["status"] == "EXPECTED_POOL_NOT_REFERENCED"
    assert result["verified"] is False


def test_wallet_must_have_a_net_mint_debit():
    tx = _transaction()
    tx["meta"]["postTokenBalances"] = [_balance(MINT, 110)]
    result = _verify(FakeRpc(_status(), tx))
    assert result["status"] == "WALLET_TOKEN_BALANCE_NOT_DEBITED"
    assert result["verified"] is False


def test_wallet_owner_attribution_is_required():
    tx = _transaction()
    tx["meta"]["preTokenBalances"][0].pop("owner")
    result = _verify(FakeRpc(_status(), tx))
    assert result["status"] == "WALLET_BALANCE_EVIDENCE_UNAVAILABLE"
    assert result["verified"] is False


def test_transaction_slot_must_match_signature_status_slot():
    result = _verify(FakeRpc(_status(), _transaction(slot=12346)))
    assert result["status"] == "SLOT_MISMATCH"
    assert result["verified"] is False


@pytest.mark.parametrize("signature,mint,pool,wallet", [
    ("not-a-signature", MINT, POOL, WALLET),
    (SIGNATURE, "not-a-mint", POOL, WALLET),
    (SIGNATURE, MINT, "not-a-pool", WALLET),
    (SIGNATURE, MINT, POOL, "not-a-wallet"),
])
def test_invalid_identifiers_rejected(signature, mint, pool, wallet):
    with pytest.raises(ValueError):
        verify_finalized_transaction_for_mint(
            FakeRpc(), signature, mint, expected_pool_address=pool, wallet_address=wallet
        )


def test_transaction_response_must_include_requested_signature():
    tx = _transaction()
    tx["transaction"]["signatures"] = [OTHER_SIGNATURE]
    result = _verify(FakeRpc(_status(), tx))
    assert result["status"] == "TRANSACTION_DETAILS_INVALID"
    assert result["verified"] is False

def test_wallet_must_be_referenced_by_transaction():
    tx = _transaction()
    tx["transaction"]["message"]["accountKeys"] = [{"pubkey": POOL}]
    result = _verify(FakeRpc(_status(), tx))
    assert result["status"] == "WALLET_NOT_REFERENCED"
    assert result["verified"] is False
\n