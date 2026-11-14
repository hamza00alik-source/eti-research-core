"""Non-fitting benchmark diagnostics for chronological Solana research folds.

All results use sampled mark returns, not executable fills or net P&L.
The fixed buy-pressure rule is a transparent comparator, not a fitted strategy.
"""
from __future__ import annotations

import math
from datetime import datetime
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
            if not isinstance(row, dict) or not _finite(row.get("target_sampled_price_return_pct")) or row.get("target_sampled_price_return_pct") <= -100:
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
            if math.isfinite(total) and total > 0 and features["txns_h1_buys"] / total >= buy_pressure_threshold:
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



def evaluate_prepared_solana_walk_forward(
    exported_labels: list[dict[str, Any]],
    prepared_dataset: dict[str, Any],
    *,
    n_splits: int = 3,
    min_train_groups: int = 3,
    embargo_seconds: int = 86400,
    buy_pressure_threshold: float = 0.55,
) -> dict[str, Any]:
    """Join prepared rows to fixed chronological validation folds and benchmark.

    Fold membership is derived from exported label timestamps, never shuffled.
    Training-fold rows are counted for diagnostics only; their outcomes are not
    passed to the benchmark. Missing prepared validation rows remain excluded.
    """
    from eti.solana_labels import chronological_walk_forward_splits

    if not isinstance(exported_labels, list) or not isinstance(prepared_dataset, dict):
        raise ValueError("exported_labels and prepared_dataset have invalid shapes")
    if prepared_dataset.get("status") != "PREPARATION_ONLY" or prepared_dataset.get("training_authorized") is not False:
        raise ValueError("prepared dataset is not a fail-closed preparation artifact")
    prepared_rows = prepared_dataset.get("rows")
    if not isinstance(prepared_rows, list):
        raise ValueError("prepared dataset rows must be a list")

    rows_by_label_id: dict[Any, dict[str, Any]] = {}
    for row in prepared_rows:
        if not isinstance(row, dict) or not isinstance(row.get("label_id"), (str, int)) or isinstance(row.get("label_id"), bool):
            raise ValueError("every prepared row must have a string or integer label_id")
        label_id = row["label_id"]
        if label_id in rows_by_label_id:
            raise ValueError("prepared dataset contains duplicate label_id values")
        rows_by_label_id[label_id] = row

    split_labels: list[dict[str, Any]] = []
    label_ids: set[Any] = set()
    for exported in exported_labels:
        if not isinstance(exported, dict) or not isinstance(exported.get("id"), (str, int)) or isinstance(exported.get("id"), bool):
            raise ValueError("every exported label must have a string or integer id")
        label_id = exported["id"]
        if label_id in label_ids:
            raise ValueError("export contains duplicate label id values")
        label_ids.add(label_id)
        label_body = exported.get("label")
        if not isinstance(label_body, dict) or exported.get("label_parse_error"):
            continue
        split_row = dict(label_body)
        for key in ("id", "mint", "pool_address", "baseline_snapshot_id", "target_snapshot_id", "horizon_seconds"):
            if key in exported:
                split_row[key] = exported[key]
        split_labels.append(split_row)

    folds = chronological_walk_forward_splits(
        split_labels,
        n_splits=n_splits,
        min_train_groups=min_train_groups,
        embargo_seconds=embargo_seconds,
    )
    benchmark_folds: list[dict[str, Any]] = []
    fold_metadata: list[dict[str, Any]] = []
    horizon_folds: dict[int, list[dict[str, Any]]] = {}
    validation_intervals: list[tuple[str, datetime, datetime]] = []
    for index, fold in enumerate(folds, start=1):
        validation_rows: list[dict[str, Any]] = []
        missing_prepared = 0
        for label in fold["validation"]:
            prepared = rows_by_label_id.get(label.get("id"))
            if prepared is None:
                missing_prepared += 1
                continue
            validation_rows.append(prepared)
            horizon = prepared.get("horizon_seconds")
            if isinstance(horizon, int) and not isinstance(horizon, bool) and horizon > 0:
                horizon_folds.setdefault(horizon, []).append({"fold": index, "row": prepared})
            try:
                start = datetime.fromisoformat(str(label["baseline_at"]).replace("Z", "+00:00"))
                end = datetime.fromisoformat(str(label["target_at"]).replace("Z", "+00:00"))
                mint = label.get("mint")
                if isinstance(mint, str) and start.tzinfo is not None and end.tzinfo is not None and end > start:
                    validation_intervals.append((mint, start, end))
            except (KeyError, TypeError, ValueError):
                pass
        benchmark_folds.append({"fold": index, "validation": validation_rows})
        fold_metadata.append({
            "fold": index,
            "train_label_count": len(fold["train"]),
            "validation_label_count": len(fold["validation"]),
            "prepared_validation_count": len(validation_rows),
            "unprepared_validation_count": missing_prepared,
            "train_mint_count": fold["train_group_count"],
            "validation_mint_count": fold["validation_group_count"],
            "purged_mint_group_count": fold["purged_group_count"],
            "boundary_at": fold["boundary_at"],
            "validation_start_at": fold["validation_start_at"],
            "embargo_seconds": fold["embargo_seconds"],
        })
    report = evaluate_solana_research_folds(
        benchmark_folds,
        buy_pressure_threshold=buy_pressure_threshold,
    )
    by_horizon: dict[str, Any] = {}
    for horizon, entries in sorted(horizon_folds.items()):
        grouped_folds: dict[int, list[dict[str, Any]]] = {}
        for entry in entries:
            grouped_folds.setdefault(entry["fold"], []).append(entry["row"])
        horizon_report = evaluate_solana_research_folds(
            [{"fold": fold_number, "validation": rows}
             for fold_number, rows in sorted(grouped_folds.items())],
            buy_pressure_threshold=buy_pressure_threshold,
        )
        by_horizon[str(horizon)] = {
            "folds": horizon_report["folds"],
            "aggregate": horizon_report["aggregate"],
            "excluded_validation_rows": horizon_report["excluded_validation_rows"],
        }

    overlap_comparisons = 0
    overlapping_pairs = 0
    intervals_by_mint: dict[str, list[tuple[datetime, datetime]]] = {}
    for mint, start, end in validation_intervals:
        intervals_by_mint.setdefault(mint, []).append((start, end))
    for intervals in intervals_by_mint.values():
        intervals.sort(key=lambda pair: pair[0])
        active: list[tuple[datetime, datetime]] = []
        for start, end in intervals:
            active = [(a, b) for a, b in active if b > start]
            overlap_comparisons += len(active)
            overlapping_pairs += sum(1 for a, b in active if a < end and start < b)
            active.append((start, end))

    report["by_horizon_seconds"] = by_horizon
    report["validation_dependence"] = {
        "validation_label_count": len(validation_intervals),
        "distinct_validation_mint_count": len(intervals_by_mint),
        "repeated_mint_label_count": len(validation_intervals) - len(intervals_by_mint),
        "overlapping_interval_pair_count": overlapping_pairs,
        "overlap_comparison_count": overlap_comparisons,
        "note": "Overlapping labels and repeated mints are dependent observations; aggregate metrics must not be interpreted as independent trades or independent samples.",
    }
    report["walk_forward"] = {
        "method": "chronological_expanding_mint_grouped_purged_embargo_v1",
        "fold_metadata": fold_metadata,
        "unmatched_prepared_row_count": sum(
            1 for label_id in rows_by_label_id if label_id not in label_ids
        ),
        "training_rows_used_for_metrics": False,
    }
    report["limitations"].append(
        "Prepared rows missing from a validation fold are excluded; the runner never substitutes another fold's row."
    )
    return report
