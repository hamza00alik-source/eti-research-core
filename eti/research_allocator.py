"""Budget-aware, non-predictive allocation of market research candidates.

This module ranks data readiness, not expected return. It creates a research
plan only: no model fitting, trading signals, or order execution.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

MAX_DAILY_USD = 10.0
MAX_MONTHLY_USD = 100.0


def _finite_nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return number


def _timeframe_seconds(value: Any) -> float:
    if not isinstance(value, str):
        raise ValueError("provenance.timeframe must be a fixed-duration string")
    match = re.fullmatch(r"(\\d+)([smhdw])", value.strip().lower())
    if not match:
        raise ValueError(f"unsupported fixed-duration timeframe: {value}")
    amount, unit = int(match.group(1)), match.group(2)
    return float(amount * {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit])


def _candidate_score(candidate: dict[str, Any], *, max_age_seconds: float) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if candidate.get("status") != "data_ready":
        return 0.0, ["data_not_ready"]

    quality = candidate.get("quality")
    provenance = candidate.get("provenance")
    if not isinstance(quality, dict) or not isinstance(provenance, dict):
        return 0.0, ["missing_quality_or_provenance"]

    try:
        rows = _finite_nonnegative("quality.rows", quality.get("rows"))
        requested = _finite_nonnegative("provenance.requested_rows", provenance.get("requested_rows"))
        age = _finite_nonnegative("quality.last_candle_age_seconds", quality.get("last_candle_age_seconds"))
        missing = _finite_nonnegative("quality.missing_interval_estimate", quality.get("missing_interval_estimate"))
        max_gap = _finite_nonnegative("quality.max_gap_seconds", quality.get("max_gap_seconds"))
        interval_value = candidate.get("timeframe_seconds")
        if interval_value is None:
            interval_value = _timeframe_seconds(provenance.get("timeframe"))
        interval = _finite_nonnegative("timeframe_seconds", interval_value)
    except ValueError as exc:
        return 0.0, [str(exc)]

    if rows <= 0 or requested <= 0 or interval <= 0:
        return 0.0, ["invalid_history_or_timeframe"]
    if quality.get("ohlcv_validated") is not True or quality.get("timestamp_order") != "ascending":
        return 0.0, ["ohlcv_quality_not_validated"]
    if age > max_age_seconds:
        reasons.append("stale_last_candle")
    asset_class = str(candidate.get("instrument", {}).get("asset_class", "")).lower()
    history_score = min(rows / requested, 1.0) * 40.0
    freshness_score = max(0.0, 30.0 * (1.0 - age / max_age_seconds)) if max_age_seconds else 0.0
    if asset_class == "crypto":
        if max_gap > interval * 3:
            reasons.append("excessive_max_gap")
        missing_fraction = min(missing / max(rows - 1.0, 1.0), 1.0)
        continuity_score = 30.0 * (1.0 - missing_fraction)
    else:
        # Session closures differ by venue and instrument. Without a verified
        # exchange calendar, raw gaps cannot safely be called missing candles.
        # Withhold continuity credit and block economic eligibility pending
        # calendar-aware validation rather than misclassifying weekends as gaps.
        continuity_score = 0.0
        reasons.append("session_calendar_not_applied")
    score = history_score + freshness_score + continuity_score
    if reasons:
        score = 0.0
    return round(score, 4), reasons


def allocate_research_budget(
    scan_report: dict[str, Any],
    *,
    max_model_fit_calls: int = 32,
    fits_per_candidate: int = 2,
    max_candidates: int = 8,
    max_age_seconds: float = 7200.0,
    estimated_run_cost_usd: float | None = None,
    daily_spend_to_date_usd: float | None = None,
    monthly_spend_to_date_usd: float | None = None,
    cost_model_status: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Rank data-ready candidates and reserve a bounded fit-call budget.

    USD authorization is deliberately fail-closed: all three cost/spend values
    must be supplied and remain under the hard $10/day and $100/month ceilings.
    Candidate transaction-cost models are independently marked; unknown costs
    never become zero-cost assumptions.
    """
    for name, value in (
        ("max_model_fit_calls", max_model_fit_calls),
        ("fits_per_candidate", fits_per_candidate),
        ("max_candidates", max_candidates),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    max_age = _finite_nonnegative("max_age_seconds", max_age_seconds)
    if max_age <= 0:
        raise ValueError("max_age_seconds must be positive")
    if not isinstance(scan_report, dict) or not isinstance(scan_report.get("candidates"), list):
        raise ValueError("scan_report must contain a candidates list")
    if scan_report.get("scan_only") is not True or scan_report.get("execution_enabled") is not False:
        raise ValueError("allocator requires a read-only scan-only report")

    usd_inputs = (estimated_run_cost_usd, daily_spend_to_date_usd, monthly_spend_to_date_usd)
    usd_known = all(value is not None for value in usd_inputs)
    if usd_known:
        estimated, daily, monthly = (
            _finite_nonnegative(name, value) for name, value in zip(
                ("estimated_run_cost_usd", "daily_spend_to_date_usd", "monthly_spend_to_date_usd"),
                usd_inputs,
            )
        )
        usd_authorized = (
            estimated <= MAX_DAILY_USD
            and daily + estimated <= MAX_DAILY_USD
            and monthly + estimated <= MAX_MONTHLY_USD
        )
        usd_reason = "within_hard_usd_caps" if usd_authorized else "usd_budget_cap_exceeded"
    else:
        estimated = daily = monthly = None
        usd_authorized = False
        usd_reason = "usd_cost_or_spend_inputs_missing"

    cost_status = cost_model_status or {}
    ranked: list[dict[str, Any]] = []
    for source in scan_report["candidates"]:
        if not isinstance(source, dict) or not isinstance(source.get("instrument"), dict):
            continue
        score, reasons = _candidate_score(source, max_age_seconds=max_age)
        instrument = source["instrument"]
        key = f"{instrument.get('venue', '')}:{instrument.get('symbol', '')}"
        transaction_cost_status = cost_status.get(key, "unknown")
        if transaction_cost_status != "validated":
            reasons.append("transaction_cost_model_not_validated")
        ranked.append({
            "instrument": instrument,
            "data_readiness_score": score,
            "data_status": source.get("status", "unknown"),
            "transaction_cost_model_status": transaction_cost_status,
            "economic_evaluation_eligible": (
                source.get("status") == "data_ready" and not reasons
                and transaction_cost_status == "validated"
            ),
            "eligibility_reasons": reasons,
            "provenance": source.get("provenance"),
            "quality": source.get("quality"),
        })

    # Stable ordering; ties are resolved by explicit instrument identity.
    ranked.sort(key=lambda row: (
        -row["data_readiness_score"],
        str(row["instrument"].get("asset_class", "")),
        str(row["instrument"].get("venue", "")),
        str(row["instrument"].get("symbol", "")),
    ))
    ready = [row for row in ranked if row["data_readiness_score"] > 0]
    capacity = max_model_fit_calls // fits_per_candidate
    selection_limit = min(max_candidates, capacity)
    selected = ready[:selection_limit]
    for index, row in enumerate(selected):
        row["research_priority_rank"] = index + 1
        row["allocated_model_fit_calls"] = fits_per_candidate
        row["training_authorized"] = False
        row["training_block_reason"] = (
            "usd_budget_not_verified" if not usd_authorized
            else "research_plan_only_training_not_invoked"
        )
    selected_keys = {
        (row["instrument"].get("venue"), row["instrument"].get("symbol"))
        for row in selected
    }
    for row in ranked:
        if (row["instrument"].get("venue"), row["instrument"].get("symbol")) not in selected_keys:
            row["research_priority_rank"] = None
            row["allocated_model_fit_calls"] = 0
            row["training_authorized"] = False
            row["training_block_reason"] = "not_selected_within_research_budget"

    return {
        "schema_version": "eti.research_allocation.v1",
        "plan_only": True,
        "training_performed": False,
        "training_authorized": False,
        "execution_enabled": False,
        "ranking_basis": "validated history coverage, freshness and continuity; not expected return",
        "compute_budget": {
            "max_model_fit_calls": max_model_fit_calls,
            "fits_per_candidate": fits_per_candidate,
            "selected_candidate_count": len(selected),
            "allocated_model_fit_calls": len(selected) * fits_per_candidate,
            "unused_model_fit_calls": max_model_fit_calls - len(selected) * fits_per_candidate,
        },
        "usd_budget": {
            "hard_daily_cap_usd": MAX_DAILY_USD,
            "hard_monthly_cap_usd": MAX_MONTHLY_USD,
            "estimated_run_cost_usd": estimated,
            "daily_spend_to_date_usd": daily,
            "monthly_spend_to_date_usd": monthly,
            "authorized": usd_authorized,
            "reason": usd_reason,
        },
        "summary": {
            "scanned_count": len(ranked),
            "data_ready_ranked_count": len(ready),
            "selected_count": len(selected),
            "economic_evaluation_eligible_count": sum(
                row["economic_evaluation_eligible"] for row in selected
            ),
        },
        "candidates": ranked,
        "warning": (
            "Research allocation only. A high data-readiness score is not alpha, "
            "a forecast, or evidence of profitability. Transaction costs must be "
            "validated before economic evaluation. No training or order execution occurred."
        ),
    }


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Create a research-only candidate allocation from a universe scan JSON"
    )
    parser.add_argument("scan_report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-allocation.json"))
    parser.add_argument("--max-model-fit-calls", type=int, default=32)
    parser.add_argument("--fits-per-candidate", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-age-seconds", type=float, default=7200)
    parser.add_argument("--estimated-run-cost-usd", type=float)
    parser.add_argument("--daily-spend-to-date-usd", type=float)
    parser.add_argument("--monthly-spend-to-date-usd", type=float)
    parser.add_argument(
        "--cost-models", type=Path,
        help="JSON mapping 'venue:symbol' to 'validated'; omitted costs remain unknown",
    )
    args = parser.parse_args()
    scan = json.loads(args.scan_report.read_text(encoding="utf-8"))
    cost_models = (
        json.loads(args.cost_models.read_text(encoding="utf-8"))
        if args.cost_models else None
    )
    report = allocate_research_budget(
        scan,
        max_model_fit_calls=args.max_model_fit_calls,
        fits_per_candidate=args.fits_per_candidate,
        max_candidates=args.max_candidates,
        max_age_seconds=args.max_age_seconds,
        estimated_run_cost_usd=args.estimated_run_cost_usd,
        daily_spend_to_date_usd=args.daily_spend_to_date_usd,
        monthly_spend_to_date_usd=args.monthly_spend_to_date_usd,
        cost_model_status=cost_models,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
