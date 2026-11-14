"""Fail-closed execution-cost evidence diagnostics for Solana research.

This module never estimates missing costs and never reports realized net P&L.
A complete evidence set can produce only a cost-adjusted sampled-mark scenario.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any


REQUIRED_COST_COMPONENTS = (
    "spread",
    "pool_price_impact",
    "protocol_fee",
    "priority_fee",
    "latency_slippage",
)


def _aware_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _valid_component(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    amount = value.get("cost_bps")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        return False
    try:
        finite_nonnegative = math.isfinite(float(amount)) and float(amount) >= 0
    except (OverflowError, TypeError, ValueError):
        return False
    return (
        finite_nonnegative
        and value.get("unit") == "bps_of_entry_notional"
        and value.get("measurement") == "measured"
        and isinstance(value.get("source_id"), str)
        and bool(value["source_id"].strip())
        and _aware_timestamp(value.get("observed_at"))
    )


def evaluate_execution_cost_evidence(
    gross_sampled_return_pct: Any,
    evidence: Any,
) -> dict[str, Any]:
    """Validate measured cost/exit evidence without implying executable P&L.

    Each cost_bps must represent the complete round-trip cost contribution
    as basis points of entry notional. Estimates are intentionally rejected.
    """
    if isinstance(gross_sampled_return_pct, bool) or not isinstance(gross_sampled_return_pct, (int, float)):
        gross_valid = False
    else:
        try:
            gross_valid = math.isfinite(float(gross_sampled_return_pct)) and float(gross_sampled_return_pct) > -100
        except (OverflowError, TypeError, ValueError):
            gross_valid = False

    if not isinstance(evidence, dict):
        evidence = {}
    components = evidence.get("cost_components")
    if not isinstance(components, dict):
        components = {}

    missing: list[str] = []
    invalid: list[str] = []
    total_bps = 0.0
    for name in REQUIRED_COST_COMPONENTS:
        component = components.get(name)
        if component is None:
            missing.append(name)
        elif not _valid_component(component):
            invalid.append(name)
        else:
            total_bps += float(component["cost_bps"])

    exit_evidence = evidence.get("exit_evidence")
    exit_valid = (
        isinstance(exit_evidence, dict)
        and exit_evidence.get("status") == "exit_observed"
        and isinstance(exit_evidence.get("source_id"), str)
        and bool(exit_evidence["source_id"].strip())
        and isinstance(exit_evidence.get("transaction_id"), str)
        and bool(exit_evidence["transaction_id"].strip())
        and _aware_timestamp(exit_evidence.get("observed_at"))
    )
    exit_status = exit_evidence.get("status") if isinstance(exit_evidence, dict) else None
    if not exit_valid:
        if exit_status == "exit_failed":
            invalid.append("exit_evidence_failed")
        else:
            missing.append("successful_exit_evidence")

    if not gross_valid:
        invalid.append("gross_sampled_return_pct")

    complete = not missing and not invalid
    scenario_return = None
    if complete:
        scenario_return = float(gross_sampled_return_pct) - total_bps / 100.0

    return {
        "status": "COST_EVIDENCE_COMPLETE" if complete else "COST_EVIDENCE_INCOMPLETE",
        "cost_evidence_complete": complete,
        "required_components": list(REQUIRED_COST_COMPONENTS),
        "missing_components": missing,
        "invalid_components": invalid,
        "measured_round_trip_cost_bps": total_bps if complete else None,
        "cost_adjusted_sampled_mark_scenario_return_pct": scenario_return,
        "net_return_pct": None,
        "realized_pnl_available": False,
        "limitations": [
            "A cost-adjusted sampled-mark scenario is not an executable quote, fill, or realized net return.",
            "Every cost component must be a measured round-trip contribution in basis points of entry notional.",
            "Missing, estimated, stale-format, or unsourced costs are not replaced with zero.",
            "Observed exit evidence does not establish that the sampled baseline and target prices were executable.",
        ],
    }
