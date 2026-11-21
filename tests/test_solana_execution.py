import base64

import pytest

from eti import solana_execution as sol


class FakeTransport:
    def __init__(self):
        self.get_result = {
            "transaction": base64.b64encode(b"unsigned-tx").decode(),
            "requestId": "req-123",
            "outAmount": "2500",
            "router": "metis",
        }
        self.post_result = {
            "status": "Success",
            "signature": "5ig",
            "inputAmountResult": "1000",
            "outputAmountResult": "2500",
        }
        self.calls = []

    def get(self, url, headers):
        self.calls.append(("GET", url, headers))
        return self.get_result

    def post(self, url, body, headers):
        self.calls.append(("POST", url, body, headers))
        return self.post_result


def test_get_order_requests_taker_transaction():
    transport = FakeTransport()
    client = sol.JupiterSwapClient("A" * 40, lambda tx: tx, transport=transport)
    order = client.get_order("So111", "EPj", 1000)
    assert order["requestId"] == "req-123"
    assert "taker=" in transport.calls[0][1]


@pytest.mark.parametrize("amount", [0, -1, True, 1.2])
def test_rejects_invalid_amount(amount):
    client = sol.JupiterSwapClient("A" * 40, lambda tx: tx, transport=FakeTransport())
    with pytest.raises(ValueError):
        client.get_order("So111", "EPj", amount)


def test_rejects_quote_without_transaction():
    transport = FakeTransport()
    transport.get_result["transaction"] = None
    client = sol.JupiterSwapClient("A" * 40, lambda tx: tx, transport=transport)
    with pytest.raises(sol.SolanaExecutionError):
        client.get_order("So111", "EPj", 1000)


def test_live_execution_is_fail_closed(monkeypatch):
    transport = FakeTransport()
    client = sol.JupiterSwapClient("A" * 40, lambda tx: b"signed", transport=transport)
    order = client.get_order("So111", "EPj", 1000)
    with pytest.raises(sol.SolanaExecutionError, match="hard-disabled"):
        client.execute_order(order, live_authorized=True)
    assert not any(call[0] == "POST" for call in transport.calls)


def test_execution_flow_with_test_only_gate_override(monkeypatch):
    transport = FakeTransport()
    signed_inputs = []
    def signer(raw):
        signed_inputs.append(raw)
        return b"signed-tx"
    client = sol.JupiterSwapClient("A" * 40, signer, transport=transport)
    order = client.get_order("So111", "EPj", 1000)
    monkeypatch.setattr(sol, "SOLANA_LIVE_EXECUTION_ENABLED", True)
    receipt = client.execute_order(order, live_authorized=True)
    assert signed_inputs == [b"unsigned-tx"]
    assert receipt.signature == "5ig"
    post = next(call for call in transport.calls if call[0] == "POST")
    assert base64.b64decode(post[2]["signedTransaction"]) == b"signed-tx"
