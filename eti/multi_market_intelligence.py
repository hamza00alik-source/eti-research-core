"""Independent multi-market validation for market-intelligence specialists.

Each instrument is trained and evaluated in isolation. No cross-market row
pooling, target mixing, or fold-level pseudo-replication is performed.
"""
from __future__ import annotations

import hashlib
import json
from statistics import median
from typing import Iterable

from .evaluation import walk_forward_folds
from .market_data_hub import MarketDataSnapshot
from .market_intelligence_training import (
    _prepare_dataset,
    train_market_intelligence_specialists,
)


def train_multi_market_intelligence_specialists(
    snapshots: Iterable[dict],
    *,
    independence_group_key: str = "independence_group",
    minimum_independent_markets: int = 3,
    horizon: int = 6,
    min_train: int = 500,
    validation_size: int = 100,
    test_size: int = 100,
    step_size: int = 100,
    max_model_fits: int = 96,
) -> dict:
    """Run isolated specialist research for several independently identified markets.

    Each item is {"snapshot": MarketDataSnapshot, "independence_group": str}.
    The group is an explicit economic-underlying identifier supplied by the
    caller (e.g. BTC, EURUSD, AAPL); duplicate groups are not counted as
    independent replications even when venues differ.

    All inputs and the aggregate fit budget are preflighted before any estimator
    is fitted. Aggregates are descriptive medians of per-market summaries only;
    overlapping folds are never treated as independent observations.
    """
    from .market_intelligence_training import _positive_int

    horizon = _positive_int(horizon, "horizon")
    min_train = _positive_int(min_train, "min_train", 20)
    validation_size = _positive_int(validation_size, "validation_size", 10)
    test_size = _positive_int(test_size, "test_size", 10)
    step_size = _positive_int(step_size, "step_size")
    max_model_fits = _positive_int(max_model_fits, "max_model_fits")
    minimum_independent_markets = _positive_int(
        minimum_independent_markets, "minimum_independent_markets", 2
    )
    if not isinstance(independence_group_key, str) or not independence_group_key.strip():
        raise ValueError("independence_group_key must be a non-empty string")

    items = list(snapshots)
    if not items:
        raise ValueError("At least one market snapshot is required")

    prepared = []
    identities = set()
    groups = set()
    timeframes = set()
    estimated_fits = 0
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"snapshots[{index}] must be an object")
        snapshot = item.get("snapshot")
        if not isinstance(snapshot, MarketDataSnapshot):
            raise TypeError(f"snapshots[{index}].snapshot must be a MarketDataSnapshot")
        group = item.get(independence_group_key)
        if not isinstance(group, str) or not group.strip():
            raise ValueError(
                f"snapshots[{index}] requires explicit {independence_group_key}"
            )
        instrument = snapshot.instrument
        identity = (instrument.venue, instrument.market_type, instrument.symbol)
        if identity in identities:
            raise ValueError(f"Duplicate market identity: {identity}")
        identities.add(identity)
        groups.add(group.strip())
        timeframes.add(snapshot.timeframe)

        dataset, features = _prepare_dataset(snapshot.frame, horizon)
        folds = walk_forward_folds(
            len(dataset), min_train=min_train,
            validation_size=validation_size, test_size=test_size,
            step_size=step_size, purge_bars=horizon,
        )
        for fold_number, fold in enumerate(folds):
            if dataset.iloc[:fold.train_end]["future_regime"].nunique() < 2:
                raise ValueError(
                    f"Market {identity} fold {fold_number} training has fewer than two regime classes"
                )
        estimated_fits += len(folds) * 4
        prepared.append({
            "snapshot": snapshot, "group": group.strip(),
            "identity": identity, "dataset_rows": len(dataset),
            "features": features, "fold_count": len(folds),
        })

    if len(timeframes) != 1:
        raise ValueError("All market snapshots must use the same timeframe")
    if estimated_fits > max_model_fits:
        raise ValueError(
            f"Multi-market training stopped before fitting: estimated "
            f"{estimated_fits} model fits exceed cap {max_model_fits}"
        )

    market_reports = []
    for item in prepared:
        snapshot = item["snapshot"]
        report = train_market_intelligence_specialists(
            snapshot.frame, horizon=horizon, min_train=min_train,
            validation_size=validation_size, test_size=test_size,
            step_size=step_size, max_model_fits=item["fold_count"] * 4,
        )
        regime = report["roles"]["regime_classifier"]
        volatility = report["roles"]["volatility_estimator"]
        baseline_mae = volatility["mean_baseline_mae"]
        model_mae = volatility["mean_outer_test_mae"]
        market_reports.append({
            "independence_group": item["group"],
            "instrument": {
                "venue": snapshot.instrument.venue,
                "market_type": snapshot.instrument.market_type,
                "symbol": snapshot.instrument.symbol,
                "asset_class": snapshot.instrument.asset_class,
            },
            "timeframe": snapshot.timeframe,
            "rows": report["rows"],
            "fold_count": report["fold_count"],
            "lineage": {
                **report["lineage"],
                "source_provenance": snapshot.provenance,
                "data_quality": snapshot.quality,
            },
            "roles": {
                "regime_classifier": {
                    "mean_outer_test_balanced_accuracy":
                        regime["mean_outer_test_balanced_accuracy"],
                    "mean_baseline_balanced_accuracy":
                        regime["mean_baseline_balanced_accuracy"],
                    "balanced_accuracy_evaluated_folds":
                        regime["balanced_accuracy_evaluated_folds"],
                    "folds_with_insufficient_class_coverage":
                        regime["folds_with_insufficient_class_coverage"],
                    "fold_class_coverage": [
                        fold["model"]["class_coverage"]
                        for fold in regime["outer_test_folds"]
                    ],
                    "folds": regime["outer_test_folds"],
                },
                "volatility_estimator": {
                    "mean_outer_test_mae": model_mae,
                    "mean_baseline_mae": baseline_mae,
                    "mae_improvement_over_baseline": baseline_mae - model_mae,
                    "folds": volatility["outer_test_folds"],
                },
            },
            "status": "research_candidate_only",
        })

    regime_values = [
        row["roles"]["regime_classifier"]["mean_outer_test_balanced_accuracy"]
        for row in market_reports
        if row["roles"]["regime_classifier"]["mean_outer_test_balanced_accuracy"] is not None
    ]
    volatility_improvements = [
        row["roles"]["volatility_estimator"]["mae_improvement_over_baseline"]
        for row in market_reports
    ]
    independent_count = len(groups)
    sufficient = independent_count >= minimum_independent_markets
    lineage_payload = [
        {
            "identity": row["instrument"],
            "group": row["independence_group"],
            "timeframe": row["timeframe"],
            "dataset_sha256": row["lineage"]["dataset_sha256"],
            "code_sha256": row["lineage"]["code_sha256"],
            "source_provenance": row["lineage"]["source_provenance"],
        }
        for row in market_reports
    ]
    lineage_hash = hashlib.sha256(
        json.dumps(lineage_payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()

    return {
        "schema": "eti.multi_market_intelligence_validation.v1",
        "department": "market_intelligence",
        "timeframe": next(iter(timeframes)),
        "horizon_bars": horizon,
        "market_count": len(market_reports),
        "independent_market_count": independent_count,
        "independence_groups": sorted(groups),
        "minimum_independent_markets": minimum_independent_markets,
        "summary": {
            "status": "cross_market_evidence_available" if sufficient
                      else "insufficient_independent_markets",
            "median_market_regime_balanced_accuracy":
                median(regime_values) if regime_values else None,
            "markets_with_defined_regime_balanced_accuracy": len(regime_values),
            "median_market_volatility_mae_improvement":
                median(volatility_improvements) if volatility_improvements else None,
            "markets_beating_volatility_baseline":
                sum(value > 0 for value in volatility_improvements),
            "interpretation": (
                "Descriptive cross-market medians only; not a confidence interval, "
                "profitability result, or model-promotion decision."
            ),
        },
        "markets": market_reports,
        "compute_budget": {
            "estimated_model_fit_calls": estimated_fits,
            "actual_model_fit_calls": estimated_fits,
            "max_model_fit_calls": max_model_fits,
            "preflight_completed_before_any_fit": True,
        },
        "lineage": {"cross_market_sha256": lineage_hash},
        "promotion": {
            "eligible": False,
            "reason": "Multi-market research report only; independent calibration, economic validation and human approval remain mandatory.",
        },
        "execution_enabled": False,
        "warning": (
            "Every market was fitted independently. No rows or targets were pooled. "
            "Folds within a market overlap and are not independent replications; "
            "the independent evidence unit is the caller-declared underlying group. "
            "The cross-market summary is descriptive only."
        ),
    }
