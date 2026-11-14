import pytest

from eti.solana_research import (
    DexScreenerClient,
    Evidence,
    audit_mint,
    scan_pair,
)


def _pair(mint="Mint111"):
    return {
        "chainId": "solana",
        "pairAddress": "Pool111",
        "baseToken": {"address": mint, "symbol": "TEST"},
        "liquidity": {"usd": 50000},
        "priceChange": {"h1": 12},
        "pairCreatedAt": 1_799_999_000_000,
    }


def test_scan_pair_passes_only_when_all_required_market_fields_are_present():
    result = scan_pair("Mint111", _pair(), now_ms=1_800_000_000_000)
    assert result["verdict"] == "PASS"
    assert result["evidence"]["liquidity_usd"] == 50000
    assert result["source"] == "dexscreener"


def test_scan_pair_rejects_below_liquidity_and_excessive_chase():
    pair = _pair()
    pair["liquidity"]["usd"] = 1000
    pair["priceChange"]["h1"] = 120
    result = scan_pair("Mint111", pair, now_ms=1_800_000_000_000)
    assert result["verdict"] == "REJECT"
    assert "liquidity" in result["reason"]
    assert "chase" in result["reason"]


def test_scan_pair_keeps_missing_data_unknown_not_pass():
    pair = _pair()
    pair["liquidity"] = {}
    pair.pop("pairCreatedAt")
    result = scan_pair("Mint111", pair)
    assert result["verdict"] == "UNKNOWN"


def test_scan_pair_waits_when_no_pair_is_indexed():
    result = scan_pair("Mint111", None)
    assert result["verdict"] == "WAIT"


def test_dexscreener_selects_highest_liquidity_pair_for_exact_mint():
    mint = "Mint111"
    pairs = [_pair(mint), _pair(mint)]
    pairs[0]["liquidity"]["usd"] = 10
    pairs[1]["liquidity"]["usd"] = 100
    pairs.append(_pair("DifferentMint"))
    selected = DexScreenerClient.select_pair(mint, pairs)
    assert selected["liquidity"]["usd"] == 100


class FakeRpc:
    def account_info(self, _mint):
        return {"data": {"parsed": {"info": {
            "mintAuthority": None, "freezeAuthority": None
        }}}}

    def token_supply(self, _mint):
        return {"amount": "1000000", "decimals": 6}

    def token_program_accounts(self, _mint):
        rows = []
        for i in range(12):
            rows.append({
                "pubkey": f"TokenAccount{i}",
                "account": {"data": {"parsed": {"info": {
                    "owner": f"Owner{i}",
                    "tokenAmount": {"amount": "5000"},
                    "delegate": None,
                    "delegatedAmount": {"amount": "0"},
                }}}},
            })
        return rows


class FakeRugCheck:
    def report(self, _mint):
        return {"markets": []}

    def insider_graph(self, _mint):
        return {"nodes": [], "links": []}


def test_audit_makes_unknown_layers_block_a_pass():
    result = audit_mint(
        "Mint111",
        rpc=FakeRpc(),
        rugcheck=FakeRugCheck(),
        pool_created_at_ms=1_799_999_000_000,
        now_ms=1_800_000_000_000,
    )
    assert len(result["layers"]) == 8
    assert result["status_counts"]["PASS"] == 4
    assert result["status_counts"]["UNKNOWN"] == 4
    assert result["passed"] is False
    assert result["blocked"] is True


def test_audit_fails_when_mint_authority_is_still_present():
    class UnsafeRpc(FakeRpc):
        def account_info(self, _mint):
            return {"data": {"parsed": {"info": {
                "mintAuthority": "Authority111", "freezeAuthority": None
            }}}}

    result = audit_mint(
        "Mint111", rpc=UnsafeRpc(), rugcheck=FakeRugCheck(),
        pool_created_at_ms=1_799_999_000_000, now_ms=1_800_000_000_000,
    )
    mint_layer = next(layer for layer in result["layers"] if layer["name"] == "mint_authority")
    assert mint_layer["status"] == "FAIL"
    assert result["passed"] is False


def test_audit_provider_failure_is_unknown_not_pass():
    class BrokenRpc(FakeRpc):
        def account_info(self, _mint):
            raise RuntimeError("RPC unavailable")

        def token_supply(self, _mint):
            raise RuntimeError("RPC unavailable")

    # Only ProviderError is normalized by the real client. A fake provider exception
    # is intentionally not swallowed: programming errors must remain visible.
    with pytest.raises(RuntimeError):
        audit_mint("Mint111", rpc=BrokenRpc(), rugcheck=FakeRugCheck())


def test_solana_research_capabilities_are_allowlisted():
    from eti.agents import default_capabilities

    capabilities = default_capabilities().capabilities()
    assert "solana.market.scan" in capabilities
    assert "solana.token.audit" in capabilities
    assert "live.tick" not in capabilities


def test_pumpportal_event_normalizer_accepts_only_known_event_shapes():
    from eti.solana_ingest import normalize_pumpportal_event

    base = {"mint": "A" * 40, "signature": "sig-001"}
    created = normalize_pumpportal_event({**base, "txType": "create", "timestamp": 123})
    assert created["event_type"] == "new_token"
    assert created["event_key"] == "pumpportal:sig-001:new_token"
    assert created["source_timestamp"] == "123"
    migrated = normalize_pumpportal_event({**base, "txType": "migrate"})
    assert migrated["event_type"] == "migration"
    assert normalize_pumpportal_event({**base, "txType": "buy"}) is None
    assert normalize_pumpportal_event({"mint": "short", "signature": "sig", "txType": "create"}) is None
    assert normalize_pumpportal_event({**base, "txType": "create"}) is not None
