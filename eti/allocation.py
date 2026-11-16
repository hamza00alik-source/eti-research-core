"""Deterministic, evidence-gated portfolio allocation for ETI paper research.

This planner converts independently assessed research candidates into bounded
paper-allocation proposals. It never places orders, approves models, or enables
live execution. Missing or failed economic evidence is rejected by default.
"""
from __future__ import annotations
import math
from typing import Any

from .config import RiskConfig
from .risk import RiskEngine, RiskRequest


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
    available_cash = _number(payload.get("available_cash", equity), "available_cash",
                              minimum=0, maximum=equity)
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

        research_report = item.get("research_report")
        if not isinstance(research_report, dict):
            rejected.append({"symbol": symbol, "reason": "missing_research_report"})
            continue
        # Recompute the disposition from the supplied source report. Never trust
        # a caller-provided status/summary object as evidence of economic viability.
        from .economic_disposition import assess_economic_viability
        assessment = assess_economic_viability(research_report)
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
        # Bind the evidence to the exact instrument and completed candle used by
        # the strategy evaluator; a candidate cannot borrow another symbol's report.
        identity = ("venue", "asset_class", "market_type", "symbol", "timeframe")
        if any(not isinstance(research_report.get(key), str) or not research_report.get(key)
               for key in identity) or research_report.get("symbol") != symbol:
            rejected.append({"symbol": symbol, "reason": "research_instrument_identity_mismatch"})
            continue
        risk_evidence = item.get("risk_evidence")
        if not isinstance(risk_evidence, dict):
            rejected.append({"symbol": symbol, "reason": "missing_stop_risk_provenance"})
            continue
        if (risk_evidence.get("symbol") != symbol
                or risk_evidence.get("venue") != research_report.get("venue")
                or risk_evidence.get("timeframe") != research_report.get("timeframe")
                or risk_evidence.get("price_as_of") != research_report.get("data_end")
                or risk_evidence.get("source") != "owner_defined_stop"):
            rejected.append({"symbol": symbol, "reason": "stop_risk_provenance_mismatch"})
            continue
        try:
            base_return = _number(assessment.get("mean_net_return"), "mean_net_return")
            stressed_return = _number(assessment.get("two_x_mean_net_return"), "two_x_mean_net_return")
            drawdown = _number(assessment.get("mean_max_drawdown"), "mean_max_drawdown",
                               minimum=0, maximum=1)
            entry_price = _number(research_report.get("latest_price"), "latest_price", minimum=0.0000001)
            risk_entry = _number(risk_evidence.get("entry_price"), "risk_evidence.entry_price",
                                 minimum=0.0000001)
            stop_price = _number(risk_evidence.get("stop_price"), "risk_evidence.stop_price",
                                 minimum=0.0000001)
            if not math.isclose(entry_price, risk_entry, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("risk entry price differs from the research snapshot")
            if stop_price >= entry_price:
                raise ValueError("long-only paper allocation requires a stop below entry")
            stop_distance = (entry_price - stop_price) / entry_price
            if not 0.0001 <= stop_distance <= 0.99:
                raise ValueError("stop distance is outside supported limits")
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
            "entry_price": entry_price, "stop_price": stop_price,
            "fold_count": folds, "base_return": base_return,
            "stressed_return": stressed_return, "mean_max_drawdown": drawdown,
        })

    existing_gross = sum(positions.values())
    if existing_gross > equity * max_gross + 1e-9:
        return {
            "kind": "paper_portfolio_allocation_plan", "status": "blocked",
            "reason": "existing_gross_exposure_exceeds_limit", "allocations": [],
            "rejected": rejected, "execution_enabled": False, "paper_only": True,
        }
    if existing_gross > equity * (1.0 - cash_buffer) + 1e-9:
        return {
            "kind": "paper_portfolio_allocation_plan", "status": "blocked",
            "reason": "existing_exposure_breaches_cash_buffer", "allocations": [],
            "rejected": rejected, "execution_enabled": False, "paper_only": True,
        }
    gross_room = max(0.0, equity * max_gross - existing_gross)
    deployable_room = max(0.0, equity * (1.0 - cash_buffer) - existing_gross)
    total_budget = min(gross_room, deployable_room, available_cash)
    active_count = sum(amount > 0 for amount in positions.values())
    slots = max(0, max_positions - active_count)
    eligible.sort(key=lambda row: (-row["score"], row["symbol"], row["strategy"]))
    selected = eligible[:slots]
    score_total = sum(row["score"] for row in selected)
    allocations = []
    planned_by_group: dict[str, float] = {}
    planned_total = 0.0
    for row in selected:
        # Allocate from the original budget, not the shrinking remainder.
        # Caps may leave cash unused; the planner never relaxes a limit.
        target = total_budget * row["score"] / score_total if score_total > 0 else 0.0
        risk_cap = equity * risk_per_trade / row["stop_distance"]
        single_room = max(0.0, equity * max_single - positions.get(row["symbol"], 0.0))
        group_room = max(0.0, equity * max_group - group_exposure.get(row["group"], 0.0)
                         - planned_by_group.get(row["group"], 0.0))
        raw_notional = max(0.0, min(target, risk_cap, single_room, group_room))
        # A second, independent deterministic gate checks the proposed amount
        # against the same account equity, stop and cumulative gross exposure.
        gate = RiskEngine(RiskConfig(
            max_risk_per_trade=risk_per_trade,
            max_position_fraction=max_single,
            max_gross_exposure=max_gross,
            allow_short=False,
            allow_live_orders=False,
        ))
        decision = gate.assess(RiskRequest(
            equity=equity, entry_price=row["entry_price"],
            stop_price=row["stop_price"], requested_notional=raw_notional,
            gross_exposure=existing_gross + planned_total,
            leverage=1.0, side="long",
        ))
        if not decision.approved:
            rejected.append({"symbol": row["symbol"],
                             "reason": "independent_risk_gate_rejected:" + ",".join(decision.reasons)})
            continue
        raw_notional = min(raw_notional, decision.approved_notional)
        # Round down, never to nearest: the serialized plan must not exceed a cap.
        notional = math.floor(raw_notional * 100_000_000) / 100_000_000
        if notional <= 0:
            rejected.append({"symbol": row["symbol"], "reason": "no_remaining_risk_or_exposure_capacity"})
            continue
        planned_total += notional
        planned_by_group[row["group"]] = planned_by_group.get(row["group"], 0.0) + notional
        allocations.append({
            "symbol": row["symbol"], "strategy": row["strategy"],
            "correlation_group": row["group"],
            "risk_adjusted_score": round(row["score"], 8),
            "proposed_notional": notional,
            "equity_fraction": notional / equity,
            "risk_budget_fraction": notional * row["stop_distance"] / equity,
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
        "equity": equity, "available_cash": available_cash,
        "cash_buffer_fraction": cash_buffer,
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
            "Economic disposition is recomputed from the supplied research report; report authenticity and storage provenance must be verified by the calling workflow.",
            "A paper allocation proposal is not a model approval, trade instruction or live order.",
            "The risk-adjusted score is a transparent ranking heuristic, not a calibrated forecast.",
            "Correlation groups are supplied labels; pairwise covariance estimation is not implemented here.",
            "Caps may leave budget unused; the planner does not relax limits to fill the portfolio.",
        ],
    }
