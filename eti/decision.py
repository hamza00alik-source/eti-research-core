"""Research decision orchestration with mandatory independent risk approval."""
from __future__ import annotations

from dataclasses import asdict

from .risk import RiskEngine, RiskRequest


def build_research_decision(
    model_probabilities: dict[str, float],
    risk_request: RiskRequest,
    risk_engine: RiskEngine | None = None,
    confidence_threshold: float = 0.55,
) -> dict:
    """Convert model probabilities to a paper decision; never submits orders."""
    if not 0.5 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0.5 and 1")
    required = {"short", "flat", "long"}
    if set(model_probabilities) != required:
        raise ValueError("Probabilities must contain short, flat, and long")
    probs = [float(model_probabilities[k]) for k in ("short", "flat", "long")]
    if any(p < 0 or p > 1 for p in probs) or abs(sum(probs) - 1) > 1e-6:
        raise ValueError("Probabilities must be in [0,1] and sum to one")
    action = max(model_probabilities, key=model_probabilities.get)
    confidence = model_probabilities[action]
    if action == "flat" or confidence < confidence_threshold:
        return {"action": "no_trade", "confidence": confidence, "risk": None, "execution": "disabled"}
    if action == "short":
        risk_request = RiskRequest(**{**asdict(risk_request), "side": "short"})
    risk = (risk_engine or RiskEngine()).assess(risk_request)
    return {
        "action": action if risk.approved else "no_trade",
        "confidence": confidence,
        "risk": asdict(risk),
        "execution": "disabled",
        "reason": "risk_approved_for_paper_simulation" if risk.approved else "risk_rejected",
    }
