"""Deterministic first-pass risk screening for Solana meme-token candidates.

This module consumes evidence supplied by independent read-only providers. It
never treats missing evidence as safe and never signs or submits transactions.
A PASS means only that configured screening thresholds were met; it is not a
guarantee against rugs, manipulation, or losses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SolanaTokenEvidence:
    mint: str
    mint_authority_active: bool | None
    freeze_authority_active: bool | None
    top_holder_fraction: float | None
    liquidity_usd: float | None
    pool_age_hours: float | None
    sell_simulation_passed: bool | None
    token_program: str | None = None
    verified: bool | None = None
    source: str | None = None
    observed_at: str | None = None


@dataclass(frozen=True)
class SolanaTokenRiskDecision:
    verdict: str
    score: int
    eligible: bool
    reasons: tuple[str, ...]
    evidence_completeness: float


@dataclass(frozen=True)
class SolanaTokenRiskPolicy:
    max_top_holder_fraction: float = 0.20
    min_liquidity_usd: float = 25_000.0
    min_pool_age_hours: float = 2.0
    reject_active_mint_authority: bool = True
    reject_active_freeze_authority: bool = True
    require_sell_simulation: bool = True
    require_verified_token: bool = False


def _valid_fraction(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _valid_nonnegative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0.0
    )


def assess_solana_token(
    evidence: SolanaTokenEvidence,
    policy: SolanaTokenRiskPolicy | None = None,
) -> SolanaTokenRiskDecision:
    """Evaluate explicit risk evidence; unknown or malformed critical fields fail closed."""
    if not isinstance(evidence, SolanaTokenEvidence):
        raise TypeError("evidence must be SolanaTokenEvidence")
    p = policy or SolanaTokenRiskPolicy()
    reasons: list[str] = []
    score = 100
    required = (
        evidence.mint_authority_active,
        evidence.freeze_authority_active,
        evidence.top_holder_fraction,
        evidence.liquidity_usd,
        evidence.pool_age_hours,
        evidence.sell_simulation_passed,
    )
    completeness = sum(value is not None for value in required) / len(required)

    if not isinstance(evidence.mint, str) or not evidence.mint.strip():
        reasons.append("missing_mint_address")
        score -= 100

    for field, value in (
        ("mint_authority_active", evidence.mint_authority_active),
        ("freeze_authority_active", evidence.freeze_authority_active),
        ("sell_simulation_passed", evidence.sell_simulation_passed),
    ):
        if value is None:
            reasons.append(f"unknown_{field}")
            score -= 15
        elif not isinstance(value, bool):
            reasons.append(f"invalid_{field}")
            score -= 100

    if evidence.mint_authority_active is True and p.reject_active_mint_authority:
        reasons.append("active_mint_authority")
        score -= 35
    if evidence.freeze_authority_active is True and p.reject_active_freeze_authority:
        reasons.append("active_freeze_authority")
        score -= 35

    if not _valid_fraction(evidence.top_holder_fraction):
        reasons.append("unknown_or_invalid_top_holder_fraction")
        score -= 25
    elif evidence.top_holder_fraction > p.max_top_holder_fraction:
        reasons.append("top_holder_concentration_exceeded")
        score -= 35

    if not _valid_nonnegative(evidence.liquidity_usd):
        reasons.append("unknown_or_invalid_liquidity")
        score -= 25
    elif evidence.liquidity_usd < p.min_liquidity_usd:
        reasons.append("liquidity_below_minimum")
        score -= 35

    if not _valid_nonnegative(evidence.pool_age_hours):
        reasons.append("unknown_or_invalid_pool_age")
        score -= 15
    elif evidence.pool_age_hours < p.min_pool_age_hours:
        reasons.append("pool_too_new")
        score -= 20

    if evidence.sell_simulation_passed is False and p.require_sell_simulation:
        reasons.append("sell_simulation_failed")
        score -= 100

    if p.require_verified_token and evidence.verified is not True:
        reasons.append("token_not_verified")
        score -= 20

    score = max(0, min(100, score))
    hard_block = any(reason in {
        "missing_mint_address", "invalid_mint_authority_active",
        "invalid_freeze_authority_active", "invalid_sell_simulation_passed",
        "active_mint_authority", "active_freeze_authority",
        "sell_simulation_failed",
    } for reason in reasons)
    eligible = not hard_block and not reasons
    verdict = "PASS" if eligible else "REJECT" if hard_block else "WATCH"
    return SolanaTokenRiskDecision(
        verdict=verdict,
        score=score,
        eligible=eligible,
        reasons=tuple(dict.fromkeys(reasons)),
        evidence_completeness=completeness,
    )


def evidence_from_mapping(payload: Mapping[str, Any]) -> SolanaTokenEvidence:
    """Strict mapping adapter for provider responses; absent values stay unknown."""
    if not isinstance(payload, Mapping):
        raise TypeError("provider payload must be a mapping")
    allowed = set(SolanaTokenEvidence.__dataclass_fields__)
    return SolanaTokenEvidence(**{key: payload[key] for key in allowed if key in payload})
