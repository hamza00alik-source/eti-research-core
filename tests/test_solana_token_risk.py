import pytest

from eti.solana_token_risk import (
    SolanaTokenEvidence,
    SolanaTokenRiskPolicy,
    assess_solana_token,
    evidence_from_mapping,
)


def clean_evidence(**overrides):
    values = {
        "mint": "So11111111111111111111111111111111111111112",
        "mint_authority_active": False,
        "freeze_authority_active": False,
        "top_holder_fraction": 0.08,
        "liquidity_usd": 100_000.0,
        "pool_age_hours": 48.0,
        "sell_simulation_passed": True,
        "verified": True,
    }
    values.update(overrides)
    return SolanaTokenEvidence(**values)


def test_clean_evidence_passes_first_pass_policy():
    result = assess_solana_token(clean_evidence())
    assert result.verdict == "PASS"
    assert result.eligible
    assert result.score == 100
    assert result.evidence_completeness == 1.0


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"mint_authority_active": True}, "active_mint_authority"),
        ({"freeze_authority_active": True}, "active_freeze_authority"),
        ({"sell_simulation_passed": False}, "sell_simulation_failed"),
    ],
)
def test_critical_authority_or_sellability_risk_rejects(overrides, reason):
    result = assess_solana_token(clean_evidence(**overrides))
    assert result.verdict == "REJECT"
    assert not result.eligible
    assert reason in result.reasons


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"top_holder_fraction": 0.35}, "top_holder_concentration_exceeded"),
        ({"liquidity_usd": 1_000}, "liquidity_below_minimum"),
        ({"pool_age_hours": 0.5}, "pool_too_new"),
        ({"liquidity_usd": None}, "unknown_or_invalid_liquidity"),
    ],
)
def test_threshold_and_missing_evidence_watch(overrides, reason):
    result = assess_solana_token(clean_evidence(**overrides))
    assert result.verdict == "WATCH"
    assert not result.eligible
    assert reason in result.reasons


def test_missing_critical_evidence_never_defaults_to_safe():
    result = assess_solana_token(SolanaTokenEvidence(mint="mint", mint_authority_active=None,
        freeze_authority_active=None, top_holder_fraction=None, liquidity_usd=None,
        pool_age_hours=None, sell_simulation_passed=None))
    assert result.verdict == "WATCH"
    assert result.evidence_completeness == 0
    assert not result.eligible


def test_invalid_numeric_values_fail_closed():
    result = assess_solana_token(clean_evidence(top_holder_fraction=float("nan")))
    assert not result.eligible
    assert "unknown_or_invalid_top_holder_fraction" in result.reasons


def test_verified_token_can_be_required_by_policy():
    result = assess_solana_token(
        clean_evidence(verified=False),
        SolanaTokenRiskPolicy(require_verified_token=True),
    )
    assert result.verdict == "WATCH"
    assert "token_not_verified" in result.reasons


def test_mapping_adapter_preserves_missing_fields_as_unknown():
    evidence = evidence_from_mapping({"mint": "mint", "liquidity_usd": 50_000})
    assert evidence.mint == "mint"
    assert evidence.mint_authority_active is None
    assert evidence.liquidity_usd == 50_000


def test_mapping_adapter_rejects_non_mapping():
    with pytest.raises(TypeError):
        evidence_from_mapping([])
