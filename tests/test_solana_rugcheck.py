import pytest

from eti.solana_rugcheck import RugCheckClient, RugCheckError


class FakeTransport:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, headers):
        self.calls.append((url, headers))
        return self.payload


def test_fetches_full_report_and_matches_mint():
    mint = "So11111111111111111111111111111111111111112"
    transport = FakeTransport({"mint": mint, "token": {"mintAuthority": None}})
    client = RugCheckClient(api_key="test-key", transport=transport)
    report = client.get_token_report(mint)
    assert report["mint"] == mint
    assert "/v1/tokens/" + mint + "/report" in transport.calls[0][0]
    assert transport.calls[0][1]["X-API-KEY"] == "test-key"


def test_rejects_report_for_different_mint():
    client = RugCheckClient(transport=FakeTransport({"mint": "different"}))
    with pytest.raises(RugCheckError, match="does not match"):
        client.get_token_report("requested-mint")


def test_provider_failure_is_wrapped_without_leaking_response():
    class Broken:
        def get(self, url, headers):
            raise TimeoutError("secret endpoint detail")
    client = RugCheckClient(transport=Broken())
    with pytest.raises(RugCheckError, match="TimeoutError"):
        client.get_token_report("mint")


@pytest.mark.parametrize(
    ("report", "field", "expected"),
    [
        ({"token": {"mintAuthority": None}}, "mintAuthority", False),
        ({"token": {"mintAuthority": "authority"}}, "mintAuthority", True),
        ({"token": {}}, "mintAuthority", None),
        ({}, "mintAuthority", None),
    ],
)
def test_authority_normalization(report, field, expected):
    assert RugCheckClient.authority_state(report, field) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(15.2, 0.152), (100, 1.0), (-1, None), (101, None), ("15", None), (None, None)],
)
def test_top_holder_percentage_normalization(value, expected):
    assert RugCheckClient.top_holder_fraction({"topHoldersPct": value}) == expected
