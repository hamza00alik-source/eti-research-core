"""Deterministic, evidence-gated portfolio allocation for ETI paper research.

This planner converts independently assessed research candidates into bounded
paper-allocation proposals. It never places orders, approves models, or enables
live execution. Missing or failed economic evidence is rejected by default.
"""
from __future__ import annotations
import math
from typing import Any


def _number(value: Any, field: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return result


def _exposure_map(value: Any, field: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    result: dict[str, float] = {}
    for key, amount in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field} keys must be non-empty strings")
        result[key] = _number(amount, f"{field}.{key}", minimum=0)
    return result


def plan_paper_allocations(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic, long-only, unleveraged paper allocation plan.

    Candidates must carry a successful independent economic disposition
    (human_review_required) and explicit base/2x cost-adjusted out-of-sample
    evidence. This status is not approval to trade; the result remains a paper
    proposal and requires a separate owner decision.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    equity = _number(payload.get("equity"), "equity", minimum=0.0000001)
    cash_buffer = _number(payload.get("cash_buffer_fraction", 0.20),
                          "cash_buffer_fraction", minimum=0, maximum=0.95)
    max_gross = _number(payload.get("max_gross_exposure_fraction", 0.25),
                        "max_gross_exposure_fraction", minimum=0.01, maximum=1)
    max_single = _number(payload.get("max_single_asset_fraction", 0.10),
                         "max_single_asset_fraction", minimum=0.01, maximum=1)
    max_group = _number(payload.get("max_correlated_group_fraction", 0.15),
                        "max_correlated_group_fraction", minimum=0.01, maximum=1)
    risk_per_trade = _number(payload.get("max_risk_per_trade", 0.005),
                             "max_risk_per_trade", minimum=0.0001, maximum=0.05)
    max_positions = payload.get("max_positions", 5)
    if isinstance(max_positions, bool) or not isinstance(max_positions, int) or not 1 <= max_positions <= 100:
        raise ValueError("max_positions must be an integer from 1 to 100")

    positions = _exposure_map(payload.get("current_positions", {}), "current_positions")
    group_exposure = _exposure_map(payload.get("correlated_group_exposure", {}),
                                   "correlated_group_exposure")
    if sum(amount > 0 for amount in positions.values()) > max_positions:
        return {
            "kind": "paper_portfolio_allocation_plan", "status": "blocked",
            "reason": "existing_position_count_exceeds_limit", "allocations": [],
            "rejected": [], "execution_enabled": False, "paper_only": True,
        }

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")
    if len(candidates) > 200:
        raise ValueError("candidates cannot exceed 200 entries")

    rejected: list[dict[str, str]] = []
    eligible: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            raise ValueError(f"candidates[{index}] must be an object")
        symbol, strategy, group = (item.get("symbol"), item.get("strategy"),
                                   item.get("correlation_group"))
        if any(not isinstance(value, str) or not value.strip()
               for value in (symbol, strategy, group)):
            raise ValueError(f"candidates[{index}] requires symbol, strategy and correlation_group")
        if symbol in seen:
            rejected.append({"symbol": symbol, "reason": "duplicate_candidate_symbol"})
            continue
        seen.add(symbol)
        if symbol in positions and positions[symbol] > 0:
            rejected.append({"symbol": symbol, "reason": "already_held_position"})
            continue

        assessment = item.get("economic_assessment")
        if not isinstance(assessment, dict):
            rejected.append({"symbol": symbol, "reason": "missing_economic_assessment"})
            continue
        if (assessment.get("status") != "human_review_required"
                or assessment.get("live_execution_eligible") is not False):
            rejected.append({"symbol": symbol, "reason": "economic_evidence_not_eligible_for_paper_review"})
            continue
        folds = assessment.get("fold_count")
        if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
            rejected.append({"symbol": symbol, "reason": "insufficient_outer_test_folds"})
            continue
        fold_counts = (
            assessment.get("base_cost_positive_return_folds"),
            assessment.get("two_x_positive_return_folds"),
            assessment.get("folds_beating_no_trade"),
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value != folds
               for value in fold_counts):
            rejected.append({"symbol": symbol, "reason": "outer_test_fold_evidence_incomplete"})
            continue
        try:
            base_return = _number(assessment.get("mean_net_return"), "mean_net_return")
            stressed_return = _number(assessment.get("two_x_mean_net_return"), "two_x_mean_net_return")
            drawdown = _number(item.get("mean_max_drawdown"), "mean_max_drawdown",
                               minimum=0, maximum=1)
            stop_distance = _number(item.get("stop_distance_fraction"),
                                    "stop_distance_fraction", minimum=0.0001, maximum=0.99)
        except ValueError:
            rejected.append({"symbol": symbol, "reason": "invalid_allocation_evidence"})
            continue
        if base_return <= 0 or stressed_return <= 0:
            rejected.append({"symbol": symbol, "reason": "non_positive_cost_adjusted_test_return"})
            continue
        score = min(base_return, stressed_return) / max(drawdown, 0.01)
        if not math.isfinite(score) or score <= 0:
            rejected.append({"symbol": symbol, "reason": "invalid_risk_adjusted_score"})
            continue
        eligible.append({
            "symbol": symbol, "strategy": strategy, "group": group,
            "score": score, "stop_distance": stop_distance,
            "fold_count": folds, "base_return": base_return,
            "stressed_return": stressed_return, "mean_max_drawdown": drawdown,
        })

    existing_gross = sum(positions.values())
    gross_room = max(0.0, equity * max_gross - existing_gross)
    deployable_room = max(0.0, equity * (1.0 - cash_buffer) - existing_gross)
    total_budget = min(gross_room, deployable_room)
    active_count = sum(amount > 0 for amount in positions.values())
    slots = max(0, max_positions - active_count)
    eligible.sort(key=lambda row: (-row["score"], row["symbol"], row["strategy"]))
    selected = eligible[:slots]
    score_total = sum(row["score"] for row in selected)
    allocations = []
    planned_by_group: dict[str, float] = {}
    planned_total = 0.0
    for row in selected:
        remaining_budget = max(0.0, total_budget - planned_total)
        target = remaining_budget * row["score"] / score_total if score_total > 0 else 0.0
        risk_cap = equity * risk_per_trade / row["stop_distance"]
        single_room = max(0.0, equity * max_single - positions.get(row["symbol"], 0.0))
        group_room = max(0.0, equity * max_group - group_exposure.get(row["group"], 0.0)
                         - planned_by_group.get(row["group"], 0.0))
        notional = max(0.0, min(target, risk_cap, single_room, group_room))
        if notional <= 0:
            rejected.append({"symbol": row["symbol"], "reason": "no_remaining_risk_or_exposure_capacity"})
            continue
        planned_total += notional
        planned_by_group[row["group"]] = planned_by_group.get(row["group"], 0.0) + notional
        allocations.append({
            "symbol": row["symbol"], "strategy": row["strategy"],
            "correlation_group": row["group"],
            "risk_adjusted_score": round(row["score"], 8),
            "proposed_notional": round(notional, 8),
            "equity_fraction": round(notional / equity, 8),
            "risk_budget_fraction": round(notional * row["stop_distance"] / equity, 8),
            "evidence": {
                "outer_test_folds": row["fold_count"],
                "mean_net_return": row["base_return"],
                "two_x_cost_mean_net_return": row["stressed_return"],
                "mean_max_drawdown": row["mean_max_drawdown"],
            },
        })

    allocated = sum(item["proposed_notional"] for item in allocations)
    return {
        "kind": "paper_portfolio_allocation_plan",
        "status": "proposals_ready" if allocations else "no_eligible_allocations",
        "equity": equity, "cash_buffer_fraction": cash_buffer,
        "proposed_total_notional": round(allocated, 8),
        "proposed_gross_exposure_fraction": round((existing_gross + allocated) / equity, 8),
        "unallocated_budget": round(max(0.0, total_budget - allocated), 8),
        "allocations": allocations, "rejected": rejected,
        "selection_method": "cost_adjusted_outer_test_return_over_drawdown_v1",
        "limits": {
            "max_positions": max_positions,
            "max_gross_exposure_fraction": max_gross,
            "max_single_asset_fraction": max_single,
            "max_correlated_group_fraction": max_group,
            "max_risk_per_trade": risk_per_trade,
            "leverage": 1.0, "shorts_allowed": False,
        },
        "owner_review_required": True, "orders_submitted": False,
        "paper_only": True, "execution_enabled": False,
        "limitations": [
            "A paper allocation proposal is not a model approval, trade instruction or live order.",
            "The risk-adjusted score is a transparent ranking heuristic, not a calibrated forecast.",
            "Correlation groups are supplied labels; pairwise covariance estimation is not implemented here.",
            "Caps may leave budget unused; the planner does not relax limits to fill the portfolio.",
        ],
    }
