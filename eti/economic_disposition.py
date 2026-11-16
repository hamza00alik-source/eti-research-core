"""Economic disposition for completed ETI research runs.

Training completion, probability calibration, and economic viability are separate
claims. This module labels outer-test evidence conservatively; it never approves
a model or enables execution.
"""
from __future__ import annotations

import math
from typing import Any


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def assess_economic_viability(report: dict[str, Any]) -> dict[str, Any]:
    """Classify model-training or rule-strategy evidence without granting authority."""
    if not isinstance(report, dict):
        raise ValueError("report must be an object")
    if report.get("kind") == "strategy_walk_forward_report":
        return assess_rule_strategy_viability(report)

    fold_count = report.get("fold_count")
    folds = report.get("folds")
    trading = report.get("trading_performance")
    summary = trading.get("summary") if isinstance(trading, dict) else None
    scenarios = (report.get("cost_sensitivity") or {}).get("scenarios")

    missing = []
    if report.get("training_performed") is not True:
        missing.append("training_performed is not true")
    if report.get("execution_enabled") is not False:
        missing.append("execution_enabled must remain false")
    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 1:
        missing.append("valid fold_count is missing")
    if not isinstance(folds, list) or not isinstance(fold_count, int) or len(folds) != fold_count:
        missing.append("fold evidence is missing or inconsistent")
    if not isinstance(summary, dict):
        missing.append("trading performance summary is missing")
    if not isinstance(scenarios, list):
        missing.append("cost sensitivity scenarios are missing")

    if missing:
        return {
            "status": "insufficient_evidence",
            "reasons": missing,
            "live_execution_eligible": False,
        }

    one_x = [
        item for item in scenarios
        if isinstance(item, dict) and item.get("cost_multiplier") == 1.0
    ]
    two_x = [
        item for item in scenarios
        if isinstance(item, dict) and item.get("cost_multiplier") == 2.0
    ]
    if (
        len(one_x) != 1 or one_x[0].get("fold_count") != fold_count
        or len(two_x) != 1 or two_x[0].get("fold_count") != fold_count
    ):
        return {
            "status": "insufficient_evidence",
            "reasons": ["unique 1x- and 2x-cost scenarios matching all folds are required"],
            "live_execution_eligible": False,
        }

    try:
        mean_net = _number(summary.get("mean_net_return"), "mean_net_return")
        folds_beating_no_trade = summary.get("folds_beating_no_trade")
        positive_folds = one_x[0].get("positive_return_folds")
        two_x_net = _number(two_x[0].get("mean_net_return"), "2x mean_net_return")
        two_x_positive = two_x[0].get("positive_return_folds")
    except ValueError as exc:
        return {
            "status": "insufficient_evidence",
            "reasons": [str(exc)],
            "live_execution_eligible": False,
        }

    integer_fields = {
        "folds_beating_no_trade": folds_beating_no_trade,
        "positive_return_folds": positive_folds,
        "two_x_positive_return_folds": two_x_positive,
    }
    invalid = [
        name for name, value in integer_fields.items()
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= fold_count
    ]
    if invalid:
        return {
            "status": "insufficient_evidence",
            "reasons": [f"invalid fold counts: {', '.join(invalid)}"],
            "live_execution_eligible": False,
        }

    reasons = []
    if mean_net <= 0:
        reasons.append("mean net return is not positive at base costs")
    if positive_folds != fold_count:
        reasons.append("not every outer-test fold has positive net return")
    if folds_beating_no_trade != fold_count:
        reasons.append("not every outer-test fold beats the no-trade baseline")
    if two_x_net <= 0:
        reasons.append("mean net return is not positive at 2x costs")
    if two_x_positive != fold_count:
        reasons.append("not every outer-test fold remains positive at 2x costs")

    return {
        "status": "no_go" if reasons else "human_review_required",
        "fold_count": fold_count,
        "mean_net_return": mean_net,
        "folds_beating_no_trade": folds_beating_no_trade,
        "base_cost_positive_return_folds": positive_folds,
        "two_x_mean_net_return": two_x_net,
        "two_x_positive_return_folds": two_x_positive,
        "reasons": reasons or ["all conservative economic checks passed; independent human review is still required"],
        "live_execution_eligible": False,
    }


def assess_rule_strategy_viability(report: dict[str, Any]) -> dict[str, Any]:
    """Assess untouched outer-test folds from the frozen rule-strategy evaluator."""
    if not isinstance(report, dict) or report.get("kind") != "strategy_walk_forward_report":
        raise ValueError("report must be a strategy_walk_forward_report")
    if report.get("method") != "validation_selected_rule_based_walk_forward_v4_cost_stress":
        return {"status": "insufficient_evidence", "reasons": ["unsupported rule-strategy evaluation method"], "live_execution_eligible": False}
    if report.get("execution_enabled") is not False:
        return {"status": "insufficient_evidence", "reasons": ["execution_enabled must remain false"], "live_execution_eligible": False}
    fold_count, folds = report.get("fold_count"), report.get("folds")
    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 2 or not isinstance(folds, list) or len(folds) != fold_count:
        return {"status": "insufficient_evidence", "reasons": ["at least two consistent outer-test folds are required"], "live_execution_eligible": False}
    base, stressed, drawdowns = [], [], []
    for index, fold in enumerate(folds):
        test = fold.get("test") if isinstance(fold, dict) else None
        if not isinstance(test, dict):
            return {"status": "insufficient_evidence", "reasons": [f"outer-test evidence missing for fold {index}"], "live_execution_eligible": False}
        try:
            base.append(_number(test.get("net_return"), f"fold[{index}].net_return"))
            stressed.append(_number(test.get("two_x_cost_net_return"), f"fold[{index}].two_x_cost_net_return"))
            drawdown = _number(test.get("max_drawdown"), f"fold[{index}].max_drawdown")
        except ValueError as exc:
            return {"status": "insufficient_evidence", "reasons": [str(exc)], "live_execution_eligible": False}
        if not -1.0 <= drawdown <= 0.0:
            return {"status": "insufficient_evidence", "reasons": [f"fold[{index}].max_drawdown must be between -1 and 0"], "live_execution_eligible": False}
        drawdowns.append(abs(drawdown))
    import statistics
    mean_base, mean_stressed = statistics.fmean(base), statistics.fmean(stressed)
    base_positive, stressed_positive = sum(v > 0 for v in base), sum(v > 0 for v in stressed)
    reasons = []
    if mean_base <= 0 or base_positive != fold_count:
        reasons.append("every outer-test fold must be positive at base costs")
    if mean_stressed <= 0 or stressed_positive != fold_count:
        reasons.append("every outer-test fold must remain positive at 2x costs")
    return {
        "status": "no_go" if reasons else "human_review_required",
        "evidence_type": "rule_based_strategy_walk_forward",
        "fold_count": fold_count,
        "mean_net_return": mean_base,
        "two_x_mean_net_return": mean_stressed,
        "base_cost_positive_return_folds": base_positive,
        "two_x_cost_positive_return_folds": stressed_positive,
        "two_x_positive_return_folds": stressed_positive,
        "folds_beating_no_trade": base_positive,
        "mean_max_drawdown": statistics.fmean(drawdowns),
        "reasons": reasons or ["rule-strategy cost-stress checks passed; owner review remains mandatory"],
        "live_execution_eligible": False,
    }
