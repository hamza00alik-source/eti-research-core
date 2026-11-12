"""Risk- and portfolio-gated research decision orchestration; never submits orders."""
from __future__ import annotations

import math
from dataclasses import asdict

from .portfolio import PortfolioLimits, assess_portfolio
from .risk import RiskEngine, RiskRequest


def build_research_decision(
    model_probabilities: dict[str, float],
    risk_request: RiskRequest,
    risk_engine: RiskEngine | None = None,
    confidence_threshold: float = 0.55,
    current_positions: dict[str, float] | None = None,
    proposed_symbol: str = "UNKNOWN",
    correlated_group_exposure: float = 0.0,
    portfolio_limits: PortfolioLimits | None = None,
) -> dict:
    """A trade is paper-eligible only after both independent gates approve it."""
    if not math.isfinite(confidence_threshold) or not 0.5 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be finite and between 0.5 and 1")
    required = {"short", "flat", "long"}
    if set(model_probabilities) != required:
        raise ValueError("Probabilities must contain short, flat, and long")
    try:
        probs = {k: float(model_probabilities[k]) for k in required}
    except (TypeError, ValueError):
        raise ValueError("Probabilities must be numeric") from None
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probs.values()):
        raise ValueError("Probabilities must be finite and in [0,1]")
    if abs(sum(probs.values()) - 1) > 1e-6:
        raise ValueError("Probabilities must sum to one")
    action = max(("short", "flat", "long"), key=lambda k: probs[k])
    confidence = probs[action]
    base = {"action": "no_trade", "confidence": confidence, "risk": None,
            "portfolio": None, "execution": "disabled"}
    if action == "flat" or confidence < confidence_threshold:
        return {**base, "reason": "flat_or_below_confidence_threshold"}
    request = RiskRequest(**{**asdict(risk_request), "side": action})
    risk = (risk_engine or RiskEngine()).assess(request)
    if not risk.approved:
        return {**base, "risk": asdict(risk), "reason": "risk_rejected"}
    portfolio = assess_portfolio(
        equity=request.equity,
        current_positions=current_positions or {},
        proposed_symbol=proposed_symbol,
        proposed_notional=risk.approved_notional,
        correlated_group_exposure=correlated_group_exposure,
        limits=portfolio_limits,
    )
    if not portfolio.approved:
        return {**base, "risk": asdict(risk), "portfolio": asdict(portfolio),
                "reason": "portfolio_rejected"}
    return {"action": action, "confidence": confidence, "risk": asdict(risk),
            "portfolio": asdict(portfolio), "execution": "disabled",
            "reason": "risk_and_portfolio_approved_for_paper_simulation"}
