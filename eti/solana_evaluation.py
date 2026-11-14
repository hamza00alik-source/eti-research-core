"""Non-fitting benchmark diagnostics for chronological Solana research folds.

All results use sampled mark returns, not executable fills or net P&L.
The fixed buy-pressure rule is a transparent comparator, not a fitted strategy.
"""
from __future__ import annotations

import math
from typing import Any, Iterable


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _summarize(returns: list[float]) -> dict[str, Any]:
    if not returns:
        return {
            "trade_count": 0, "mean_return_pct": None, "median_return_pct": None,
            "win_rate": None, "cumulative_compounded_return_pct": None,
            "max_drawdown_pct": None,
        }
    ordered = sorted(returns)
    equity = peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        equity *= 1.0 + value / 100.0
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak * 100.0)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    compounded = (math.prod(1.0 + x / 100.0 for x in returns) - 1.0) * 100.0
    return {
        "trade_count": len(returns),
        "mean_return_pct": sum(returns) / len(returns),
        "median_return_pct": median,
        "win_rate": sum(x > 0 for x in returns) / len(returns),
        "cumulative_compounded_return_pct": compounded,
        "max_drawdown_pct": max_drawdown,
    }


def evaluate_solana_research_folds(
    folds: Iterable[dict[str, Any]],
    *,
    buy_pressure_threshold: float = 0.55,
) -> dict[str, Any]:
    """Compare fixed, non-fitted baselines on each fold's validation rows only."""
    if isinstance(buy_pressure_threshold, bool) or not isinstance(buy_pressure_threshold, (int, float)) or not 0.5 < buy_pressure_threshold <= 1:
        raise ValueError("buy_pressure_threshold must be greater than 0.5 and at most 1")
    fold_reports: list[dict[str, Any]] = []
    all_returns: dict[str, list[float]] = {
        "no_trade": [], "always_long": [], "buy_pressure": [],
    }
    excluded_rows = 0
    for fold_number, fold in enumerate(folds, start=1):
        if not isinstance(fold, dict) or not isinstance(fold.get("validation"), list):
            raise ValueError("each fold must contain a validation list")
        valid: list[tuple[float, dict[str, Any]]] = []
        for row in fold["validation"]:
            if not isinstance(row, dict) or not _finite(row.get("target_sampled_price_return_pct")):
                excluded_rows += 1
                continue
            features = row.get("features")
            if not isinstance(features, dict):
                excluded_rows += 1
                continue
            buys, sells = features.get("txns_h1_buys"), features.get("txns_h1_sells")
            if not _finite(buys) or not _finite(sells) or buys < 0 or sells < 0:
                excluded_rows += 1
                continue
            valid.append((float(row["target_sampled_price_return_pct"]), row))
        valid.sort(key=lambda pair: (
            str(pair[1].get("baseline_at", "")),
            str(pair[1].get("mint", "")),
            str(pair[1].get("pool_address", "")),
        ))
        always = [outcome for outcome, _ in valid]
        pressure = []
        for outcome, row in valid:
            features = row["features"]
            total = features["txns_h1_buys"] + features["txns_h1_sells"]
            if total > 0 and features["txns_h1_buys"] / total >= buy_pressure_threshold:
                pressure.append(outcome)
        all_returns["always_long"].extend(always)
        all_returns["buy_pressure"].extend(pressure)
        fold_reports.append({
            "fold": fold.get("fold", fold_number),
            "validation_rows": len(valid),
            "baselines": {
                "no_trade": _summarize([]),
                "always_long": _summarize(always),
                "fixed_buy_pressure": _summarize(pressure),
            },
        })
    return {
        "status": "BENCHMARK_DIAGNOSTICS_ONLY",
        "model_fitting_performed": False,
        "training_authorized": False,
        "outcome_basis": "gross_sampled_price_return_pct",
        "buy_pressure_rule": {
            "definition": "txns_h1_buys / (txns_h1_buys + txns_h1_sells) >= threshold",
            "threshold": float(buy_pressure_threshold),
            "fit_on_validation_data": False,
        },
        "folds": fold_reports,
        "aggregate": {
            "no_trade": _summarize([]),
            "always_long": _summarize(all_returns["always_long"]),
            "fixed_buy_pressure": _summarize(all_returns["buy_pressure"]),
        },
        "excluded_validation_rows": excluded_rows,
        "limitations": [
            "Returns are sampled mark outcomes, not executable fills or net P&L.",
            "No fees, spread, price impact, priority fees, latency, failed exits or capital constraints are modeled.",
            "The fixed buy-pressure comparator is descriptive and is not a validated trading strategy.",
            "Aggregate metrics pool validation observations across folds and are not an independent final holdout.",
            "No threshold selection, model fitting, calibration or training approval occurs.",
        ],
    }
