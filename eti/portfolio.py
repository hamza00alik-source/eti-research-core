"""Portfolio-wide deterministic exposure checks for paper research."""
from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class PortfolioLimits:
    max_positions: int = 5
    max_gross_exposure_fraction: float = 0.25
    max_single_asset_fraction: float = 0.10
    max_correlated_group_fraction: float = 0.15

    def __post_init__(self):
        if isinstance(self.max_positions, bool) or not isinstance(self.max_positions, int) or self.max_positions < 1:
            raise ValueError("max_positions must be a positive integer")
        fractions = (self.max_gross_exposure_fraction, self.max_single_asset_fraction,
                     self.max_correlated_group_fraction)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or not 0 < v <= 1 for v in fractions):
            raise ValueError("Portfolio exposure limits must be finite fractions in (0, 1]")

@dataclass(frozen=True)
class PortfolioAssessment:
    approved: bool
    reasons: tuple[str, ...]
    resulting_gross_fraction: float

def assess_portfolio(equity: float, current_positions: dict[str, float],
                     proposed_symbol: str, proposed_notional: float,
                     correlated_group_exposure: float = 0.0,
                     limits: PortfolioLimits | None = None) -> PortfolioAssessment:
    limits = limits or PortfolioLimits()
    numeric = (equity, proposed_notional, correlated_group_exposure)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in numeric):
        return PortfolioAssessment(False, ("non_finite_or_invalid_numeric_input",), 0.0)
    if equity <= 0 or proposed_notional <= 0 or correlated_group_exposure < 0:
        return PortfolioAssessment(False, ("invalid_equity_or_notional",), 0.0)
    if not isinstance(proposed_symbol, str) or not proposed_symbol.strip():
        return PortfolioAssessment(False, ("missing_symbol",), 0.0)
    if not isinstance(current_positions, dict):
        return PortfolioAssessment(False, ("invalid_existing_positions",), 0.0)
    positions: dict[str, float] = {}
    for symbol, value in current_positions.items():
        if not isinstance(symbol, str) or not symbol.strip():
            return PortfolioAssessment(False, ("invalid_existing_position_symbol",), 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return PortfolioAssessment(False, ("invalid_existing_position",), 0.0)
        positions[symbol] = abs(float(value))
    positions = {k: v for k, v in positions.items() if v > 0}
    existing = positions.get(proposed_symbol, 0.0)
    resulting = sum(positions.values()) + proposed_notional
    reasons = []
    if proposed_symbol not in positions and len(positions) >= limits.max_positions:
        reasons.append("max_positions_reached")
    if resulting / equity > limits.max_gross_exposure_fraction:
        reasons.append("gross_exposure_limit")
    if (existing + proposed_notional) / equity > limits.max_single_asset_fraction:
        reasons.append("single_asset_limit")
    if (correlated_group_exposure + proposed_notional) / equity > limits.max_correlated_group_fraction:
        reasons.append("correlated_group_limit")
    return PortfolioAssessment(not reasons, tuple(reasons), resulting / equity)
