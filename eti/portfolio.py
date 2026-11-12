"""Portfolio-wide deterministic exposure checks for paper research."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioLimits:
    max_positions: int = 5
    max_gross_exposure_fraction: float = 0.25
    max_single_asset_fraction: float = 0.10
    max_correlated_group_fraction: float = 0.15


@dataclass(frozen=True)
class PortfolioAssessment:
    approved: bool
    reasons: tuple[str, ...]
    resulting_gross_fraction: float


def assess_portfolio(
    equity: float,
    current_positions: dict[str, float],
    proposed_symbol: str,
    proposed_notional: float,
    correlated_group_exposure: float = 0.0,
    limits: PortfolioLimits | None = None,
) -> PortfolioAssessment:
    limits = limits or PortfolioLimits()
    reasons = []
    if equity <= 0 or proposed_notional <= 0:
        return PortfolioAssessment(False, ("invalid_equity_or_notional",), 0.0)
    positions = {k: abs(float(v)) for k, v in current_positions.items() if abs(float(v)) > 0}
    existing = positions.get(proposed_symbol, 0.0)
    resulting = sum(positions.values()) + proposed_notional
    if proposed_symbol not in positions and len(positions) >= limits.max_positions:
        reasons.append("max_positions_reached")
    if resulting / equity > limits.max_gross_exposure_fraction:
        reasons.append("gross_exposure_limit")
    if (existing + proposed_notional) / equity > limits.max_single_asset_fraction:
        reasons.append("single_asset_limit")
    if (correlated_group_exposure + proposed_notional) / equity > limits.max_correlated_group_fraction:
        reasons.append("correlated_group_limit")
    return PortfolioAssessment(not reasons, tuple(reasons), resulting / equity)
