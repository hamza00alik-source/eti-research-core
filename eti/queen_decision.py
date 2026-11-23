"""Evidence-gated Queen decision protocol.

This module turns specialist outputs into a bounded research disposition. It
never creates an order, selects an order side, or grants execution authority.
Qwen may add a caution but can never approve a candidate or override hard gates.
"""
from __future__ import annotations

import math
from typing import Any


def _number(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def queen_trade_review(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate research evidence; output is paper-candidate disposition only.

    Required inputs are explicit, independently generated specialist results:
    market_evidence, model_evidence, strategy_evidence, cost_evidence and
    risk_evidence. Optional Qwen advice is advisory-only and can add a veto.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    market = _object(payload, "market_evidence")
    model = _object(payload, "model_evidence")
    strategy = _object(payload, "strategy_evidence")
    costs = _object(payload, "cost_evidence")
    risk = _object(payload, "risk_evidence")
    qwen = _object(payload, "qwen_advisory")

    blockers: list[str] = []
    warnings: list[str] = []

    if market.get("status") != "valid":
        blockers.append("market_data_not_valid")
    age = _number(market, "candle_age_seconds")
    if age is None or age < 0 or age > 7200:
        blockers.append("market_data_stale_or_unknown")
    spread = _number(market, "spread_bps")
    if spread is None or spread < 0 or spread > 50:
        blockers.append("spread_unknown_or_excessive")

    if model.get("status") != "validated":
        blockers.append("model_not_validated")
    probability = _number(model, "probability_up")
    if probability is None or not 0.0 <= probability <= 1.0 or probability < 0.60:
        blockers.append("model_confidence_below_threshold")
    model_edge = _number(model, "oos_net_edge_bps")
    if model_edge is None or model_edge <= 0:
        blockers.append("model_has_no_positive_oos_net_edge")
    model_samples = model.get("oos_observations")
    if isinstance(model_samples, bool) or not isinstance(model_samples, int) or model_samples < 100:
        blockers.append("model_oos_sample_too_small")

    if strategy.get("status") != "validated":
        blockers.append("strategy_not_validated")
    strategy_edge = _number(strategy, "oos_net_return_bps")
    if strategy_edge is None or strategy_edge <= 0:
        blockers.append("strategy_has_no_positive_oos_net_return")
    trades = strategy.get("oos_trades")
    if isinstance(trades, bool) or not isinstance(trades, int) or trades < 30:
        blockers.append("strategy_oos_trade_sample_too_small")
    drawdown = _number(strategy, "oos_max_drawdown")
    if drawdown is None or not 0 <= drawdown <= 0.10:
        blockers.append("strategy_drawdown_unknown_or_excessive")

    if costs.get("status") != "validated":
        blockers.append("transaction_costs_not_validated")
    cost_bps = _number(costs, "round_trip_cost_bps")
    if cost_bps is None or cost_bps < 0 or cost_bps > 100:
        blockers.append("transaction_costs_unknown_or_excessive")
    expected_edge = _number(model, "expected_edge_bps")
    if (expected_edge is None or cost_bps is None
            or expected_edge <= cost_bps * 1.5):
        blockers.append("expected_edge_does_not_clear_cost_buffer")

    if risk.get("approved") is not True:
        blockers.append("deterministic_risk_gate_not_approved")
    hard_blocks = risk.get("hard_blocks", [])
    if not isinstance(hard_blocks, list) or hard_blocks:
        blockers.append("deterministic_risk_gate_has_hard_blocks")

    qwen_status = qwen.get("status")
    qwen_recommendation = qwen.get("recommendation")
    if qwen_status == "completed" and qwen_recommendation == "avoid":
        blockers.append("qwen_advisory_caution")
    elif qwen_status not in {None, "disabled", "completed"}:
        warnings.append("qwen_advisory_unavailable")
    elif qwen_status == "completed" and qwen_recommendation not in {
        "watch", "avoid", "insufficient_data"
    }:
        warnings.append("qwen_advisory_schema_unrecognized")
    if qwen_recommendation == "insufficient_data":
        warnings.append("qwen_reports_insufficient_context")

    disposition = "paper_candidate" if not blockers else "hold"
    return {
        "kind": "queen_trade_review",
        "disposition": disposition,
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
        "evidence_summary": {
            "market_data_valid": market.get("status") == "valid",
            "model_status": model.get("status"),
            "strategy_status": strategy.get("status"),
            "cost_status": costs.get("status"),
            "risk_approved": risk.get("approved") is True,
            "qwen_advisory": qwen_recommendation if qwen_status == "completed" else "not_used",
        },
        "execution_authority": False,
        "live_orders_allowed": False,
        "model_promotion_authority": False,
        "qwen_can_approve": False,
        "requires_separate_paper_engine": True,
    }
