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
    """Classify economic evidence without granting deployment authority."""
    if not isinstance(report, dict):
        raise ValueError("report must be an object")

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
