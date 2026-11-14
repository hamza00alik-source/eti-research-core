"""Non-fitting benchmark diagnostics for chronological Solana research folds.

All results use sampled mark returns, not executable fills or net P&L.
The fixed buy-pressure rule is a transparent comparator, not a fitted strategy.
"""
from __future__ import annotations

import math
import random
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
            "compounding_interpretation": "NOT_APPLICABLE_NO_VALID_OBSERVATIONS",
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
        "compounding_interpretation": (
            "POOLED_SAMPLED_OBSERVATION_DIAGNOSTIC_NOT_PORTFOLIO_PNL"
        ),
    }


def _cluster_bootstrap_mean_interval(
    cluster_returns: dict[str, list[float]],
    *,
    replicates: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 1729,
) -> dict[str, Any]:
    """Exploratory percentile interval resampling whole mint clusters.

    The estimand is the row-weighted mean among selected observations. Sampling
    whole mints preserves within-mint dependence, but does not correct selection,
    regime drift, overlapping cross-mint market shocks, or provider bias.
    """
    clusters = [values for _, values in sorted(cluster_returns.items()) if values]
    cluster_count = len(clusters)
    result: dict[str, Any] = {
        "method": "MINT_CLUSTER_BOOTSTRAP_PERCENTILE",
        "confidence_level": confidence_level,
        "replicates": replicates,
        "seed": seed,
        "cluster_count": cluster_count,
        "mean_return_pct_interval": None,
        "status": "INSUFFICIENT_MINT_CLUSTERS",
        "minimum_clusters_for_interval": 20,
        "interpretation": (
            "Exploratory cluster-resampling interval for the row-weighted mean; "
            "not a significance test, predictive guarantee, portfolio return, "
            "or correction for shared market shocks."
        ),
    }
    if cluster_count < 20:
        return result
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled = [clusters[rng.randrange(cluster_count)] for _ in range(cluster_count)]
        values = [value for cluster in sampled for value in cluster]
        if values:
            estimates.append(sum(values) / len(values))
    if not estimates:
        result["status"] = "NO_BOOTSTRAP_ESTIMATES"
        return result
    estimates.sort()
    tail = (1.0 - confidence_level) / 2.0

    def percentile(q: float) -> float:
        position = (len(estimates) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return estimates[lower]
        return estimates[lower] + (estimates[upper] - estimates[lower]) * (position - lower)

    result["mean_return_pct_interval"] = [percentile(tail), percentile(1.0 - tail)]
    result["status"] = "EXPLORATORY_INTERVAL_AVAILABLE"
    return result


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
    cluster_returns: dict[str, dict[str, list[float]]] = {
        "always_long": {}, "buy_pressure": {},
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
        for outcome, row in valid:
            mint = row.get("mint")
            cluster_id = mint if isinstance(mint, str) and mint.strip() else "__missing_mint_cluster__"
            cluster_returns["always_long"].setdefault(cluster_id, []).append(outcome)
            features = row["features"]
            total = features["txns_h1_buys"] + features["txns_h1_sells"]
            if math.isfinite(total) and total > 0 and features["txns_h1_buys"] / total >= buy_pressure_threshold:
                cluster_returns["buy_pressure"].setdefault(cluster_id, []).append(outcome)
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
        "aggregate_metric_interpretation": (
            "Pooled compounded returns and sequential drawdown are ordering-dependent "
            "sampled-observation diagnostics, not a portfolio equity curve, executable "
            "trade sequence, or realized P&L. Validation observations may overlap, "
            "repeat mints, and span different horizons."
        ),
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
        "cluster_uncertainty": {
            "always_long": _cluster_bootstrap_mean_interval(cluster_returns["always_long"]),
            "fixed_buy_pressure": _cluster_bootstrap_mean_interval(cluster_returns["buy_pressure"]),
            "no_trade": {
                "status": "NOT_APPLICABLE_NO_TRADE_OBSERVATIONS",
                "mean_return_pct_interval": None,
                "cluster_count": 0,
            },
            "note": (
                "Intervals are withheld below 20 distinct mint clusters. Even when "
                "available, this exploratory bootstrap does not account for common "
                "market shocks across mints or establish strategy significance."
            ),
        },
        "excluded_validation_rows": excluded_rows,
        "limitations": [
            "Returns are sampled mark outcomes, not executable fills or net P&L.",
            "No fees, spread, price impact, priority fees, latency, failed exits or capital constraints are modeled.",
            "The fixed buy-pressure comparator is descriptive and is not a validated trading strategy.",
            "Aggregate metrics pool validation observations across folds and are not an independent final holdout.",
            "Compounded return and sequential drawdown are pooled sampled-observation diagnostics, not portfolio performance or realized P&L.",
            "No threshold selection, model fitting, calibration or training approval occurs.",
            "Mint-cluster bootstrap intervals are exploratory; intervals are withheld below 20 clusters and do not account for shared market shocks across mints.",
        ],
    }



def evaluate_solana_regime_diagnostics(
    validation_rows: list[dict[str, Any]],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Stratify validation outcomes by predeclared baseline flow/activity proxies.

    These are not trend or volatility regimes. Snapshot-only inputs cannot
    establish realized volatility, candle structure, or a persistent trend.
    Thresholds must be supplied before inspecting validation outcomes.
    """
    if thresholds is None:
        return {
            "status": "NOT_CONFIGURED",
            "classified_row_count": 0,
            "regimes": {},
            "note": "No thresholds supplied; no regime labels were inferred.",
        }
    required = {
        "buy_pressure_high", "buy_pressure_low",
        "turnover_liquidity_ratio_high", "turnover_liquidity_ratio_low",
    }
    if not isinstance(thresholds, dict) or set(thresholds) != required:
        raise ValueError("regime thresholds must contain exactly: " + ", ".join(sorted(required)))
    values = {key: thresholds[key] for key in required}
    if any(not _finite(value) for value in values.values()):
        raise ValueError("regime thresholds must be finite numbers")
    if not (0 <= values["buy_pressure_low"] < values["buy_pressure_high"] <= 1):
        raise ValueError("buy-pressure thresholds must satisfy 0 <= low < high <= 1")
    if not (0 <= values["turnover_liquidity_ratio_low"] < values["turnover_liquidity_ratio_high"]):
        raise ValueError("turnover/liquidity thresholds must satisfy 0 <= low < high")
    groups: dict[str, list[float]] = {}
    excluded = 0
    for row in validation_rows:
        if not isinstance(row, dict) or not _finite(row.get("target_sampled_price_return_pct")):
            excluded += 1
            continue
        features = row.get("features")
        if not isinstance(features, dict):
            excluded += 1
            continue
        buys, sells = features.get("txns_h1_buys"), features.get("txns_h1_sells")
        volume, liquidity = features.get("volume_h24_usd"), features.get("liquidity_usd")
        if (not all(_finite(x) for x in (buys, sells, volume, liquidity))
                or buys < 0 or sells < 0 or volume < 0 or liquidity <= 0
                or buys + sells <= 0):
            excluded += 1
            continue
        pressure = buys / (buys + sells)
        turnover = volume / liquidity
        if pressure >= values["buy_pressure_high"]:
            flow = "BUY_DOMINANT"
        elif pressure <= values["buy_pressure_low"]:
            flow = "SELL_DOMINANT"
        else:
            flow = "BALANCED_FLOW"
        if turnover >= values["turnover_liquidity_ratio_high"]:
            activity = "HIGH_TURNOVER_PROXY"
        elif turnover <= values["turnover_liquidity_ratio_low"]:
            activity = "LOW_TURNOVER_PROXY"
        else:
            activity = "MID_TURNOVER_PROXY"
        groups.setdefault(flow + "__" + activity, []).append(
            float(row["target_sampled_price_return_pct"])
        )
    return {
        "status": "BASELINE_PROXY_REGIME_DIAGNOSTICS_ONLY",
        "thresholds": values,
        "classified_row_count": sum(len(items) for items in groups.values()),
        "excluded_row_count": excluded,
        "regimes": {name: _summarize(returns) for name, returns in sorted(groups.items())},
        "interpretation": (
            "Groups use only baseline 1h buy/sell flow and h24 volume-to-liquidity "
            "snapshot proxies. They are not trend, realized-volatility, candle, "
            "or market-wide regimes. Thresholds are caller-predeclared; results "
            "are validation diagnostics, not strategy selection or approval."
        ),
    }



def evaluate_solana_fold_regime_stability(
    folds: list[dict[str, Any]],
    thresholds: dict[str, float] | None,
    *,
    buy_pressure_threshold: float = 0.55,
    minimum_rows_per_fold_regime: int = 5,
    minimum_mints_per_fold_regime: int = 3,
) -> dict[str, Any]:
    """Report chronological fold stability for fixed baseline proxy regimes.

    Fold-level evidence is withheld from stability aggregation when either
    the row or distinct-mint minimum is not met. This is descriptive only.
    """
    if (isinstance(minimum_rows_per_fold_regime, bool)
            or not isinstance(minimum_rows_per_fold_regime, int)
            or minimum_rows_per_fold_regime < 1):
        raise ValueError("minimum_rows_per_fold_regime must be a positive integer")
    if (isinstance(minimum_mints_per_fold_regime, bool)
            or not isinstance(minimum_mints_per_fold_regime, int)
            or minimum_mints_per_fold_regime < 1):
        raise ValueError("minimum_mints_per_fold_regime must be a positive integer")
    if thresholds is None:
        return {
            "status": "NOT_CONFIGURED",
            "folds": [],
            "regimes": {},
            "minimum_rows_per_fold_regime": minimum_rows_per_fold_regime,
            "minimum_mints_per_fold_regime": minimum_mints_per_fold_regime,
            "note": "No predeclared regime thresholds supplied; no fold stability inferred.",
        }
    fold_groups: list[dict[str, list[dict[str, Any]]]] = []
    all_regimes: set[str] = set()
    fold_reports: list[dict[str, Any]] = []
    required = {
        "buy_pressure_high", "buy_pressure_low",
        "turnover_liquidity_ratio_high", "turnover_liquidity_ratio_low",
    }
    if not isinstance(thresholds, dict) or set(thresholds) != required:
        raise ValueError("regime thresholds must contain exactly: " + ", ".join(sorted(required)))
    values = {key: thresholds[key] for key in required}
    if any(not _finite(value) for value in values.values()):
        raise ValueError("regime thresholds must be finite numbers")
    if not (0 <= values["buy_pressure_low"] < values["buy_pressure_high"] <= 1):
        raise ValueError("buy-pressure thresholds must satisfy 0 <= low < high <= 1")
    if not (0 <= values["turnover_liquidity_ratio_low"] < values["turnover_liquidity_ratio_high"]):
        raise ValueError("turnover/liquidity thresholds must satisfy 0 <= low < high")
    for fold_number, fold in enumerate(folds, start=1):
        if not isinstance(fold, dict) or not isinstance(fold.get("validation"), list):
            raise ValueError("each fold must contain a validation list")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in fold["validation"]:
            if not isinstance(row, dict) or not _finite(row.get("target_sampled_price_return_pct")):
                continue
            features = row.get("features")
            if not isinstance(features, dict):
                continue
            buys, sells = features.get("txns_h1_buys"), features.get("txns_h1_sells")
            volume, liquidity = features.get("volume_h24_usd"), features.get("liquidity_usd")
            if (not all(_finite(x) for x in (buys, sells, volume, liquidity))
                    or buys < 0 or sells < 0 or volume < 0 or liquidity <= 0
                    or buys + sells <= 0):
                continue
            pressure = buys / (buys + sells)
            turnover = volume / liquidity
            flow = ("BUY_DOMINANT" if pressure >= values["buy_pressure_high"]
                    else "SELL_DOMINANT" if pressure <= values["buy_pressure_low"]
                    else "BALANCED_FLOW")
            activity = ("HIGH_TURNOVER_PROXY" if turnover >= values["turnover_liquidity_ratio_high"]
                        else "LOW_TURNOVER_PROXY" if turnover <= values["turnover_liquidity_ratio_low"]
                        else "MID_TURNOVER_PROXY")
            grouped.setdefault(flow + "__" + activity, []).append(row)
        fold_groups.append(grouped)
        all_regimes.update(grouped)
        strata = {}
        for regime, rows in sorted(grouped.items()):
            mint_counts: dict[str, int] = {}
            for row in rows:
                mint = row.get("mint")
                key = mint.strip() if isinstance(mint, str) and mint.strip() else "__missing_mint__"
                mint_counts[key] = mint_counts.get(key, 0) + 1
            distinct_mints = len(mint_counts)
            row_count = len(rows)
            eligible = (row_count >= minimum_rows_per_fold_regime
                        and distinct_mints >= minimum_mints_per_fold_regime
                        and "__missing_mint__" not in mint_counts)
            benchmark = evaluate_solana_research_folds(
                [{"fold": fold.get("fold", fold_number), "validation": rows}],
                buy_pressure_threshold=buy_pressure_threshold,
            )
            strata[regime] = {
                "row_count": row_count,
                "distinct_mint_count": distinct_mints,
                "largest_mint_row_share": max(mint_counts.values()) / row_count if row_count else None,
                "status": "FOLD_SAMPLE_THRESHOLD_MET" if eligible else "INSUFFICIENT_OR_MISSING_MINT_DIVERSITY",
                "stability_eligible": eligible,
                "baselines": benchmark["aggregate"],
            }
        fold_reports.append({
            "fold": fold.get("fold", fold_number),
            "classified_row_count": sum(len(rows) for rows in grouped.values()),
            "regimes": strata,
        })
    regime_summary: dict[str, Any] = {}
    for regime in sorted(all_regimes):
        per_baseline: dict[str, Any] = {}
        for baseline in ("always_long", "fixed_buy_pressure"):
            means = []
            positive = 0
            negative = 0
            eligible_folds = 0
            for grouped in fold_groups:
                rows = grouped.get(regime, [])
                mints = {row.get("mint") for row in rows
                         if isinstance(row.get("mint"), str) and row.get("mint").strip()}
                if (len(rows) < minimum_rows_per_fold_regime
                        or len(mints) < minimum_mints_per_fold_regime
                        or any(not isinstance(row.get("mint"), str) or not row.get("mint").strip() for row in rows)):
                    continue
                summary = evaluate_solana_research_folds(
                    [{"validation": rows}], buy_pressure_threshold=buy_pressure_threshold,
                )["aggregate"][baseline]
                mean = summary.get("mean_return_pct")
                if not _finite(mean):
                    continue
                eligible_folds += 1
                means.append(float(mean))
                positive += mean > 0
                negative += mean < 0
            per_baseline[baseline] = {
                "eligible_fold_count": eligible_folds,
                "positive_mean_fold_count": positive,
                "negative_mean_fold_count": negative,
                "mean_return_sign_consistency": (
                    max(positive, negative) / eligible_folds if eligible_folds else None
                ),
                "fold_mean_returns_pct": means,
                "status": "DESCRIPTIVE_FOLD_STABILITY_AVAILABLE" if eligible_folds else "INSUFFICIENT_ELIGIBLE_FOLDS",
            }
        regime_summary[regime] = per_baseline
    return {
        "status": "CHRONOLOGICAL_FOLD_REGIME_STABILITY_DIAGNOSTICS_ONLY",
        "thresholds": values,
        "minimum_rows_per_fold_regime": minimum_rows_per_fold_regime,
        "minimum_mints_per_fold_regime": minimum_mints_per_fold_regime,
        "folds": fold_reports,
        "regimes": regime_summary,
        "model_fitting_performed": False,
        "training_authorized": False,
        "note": "Fold means are gross sampled-mark diagnostics. Eligibility excludes missing mint identities and requires both row and mint minima; sign consistency is not significance, edge validation, or independent evidence when labels overlap.",
    }


def evaluate_solana_time_series_regimes(
    history_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify validation rows from point-in-time same-mint/pool history.

    Volatility is the sample standard deviation of interval-normalized log
    returns, expressed as percent per square-root-hour. This is a sampling-
    adjusted proxy, not candle-based realized volatility.
    """
    if config is None:
        return {
            "status": "NOT_CONFIGURED",
            "classified_row_count": 0,
            "unclassified_row_count": len(validation_rows),
            "regimes": {},
            "note": "Explicit lookback, sampling controls and thresholds are required.",
        }
    required = {
        "lookback_observations", "minimum_observations",
        "trend_abs_return_pct", "volatility_high_pct", "volatility_low_pct",
        "max_gap_seconds", "minimum_history_span_seconds",
        "minimum_rows_per_regime_horizon", "buy_pressure_threshold",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("time-series regime config must contain exactly: " + ", ".join(sorted(required)))
    lookback = config["lookback_observations"]
    minimum = config["minimum_observations"]
    for key in ("lookback_observations", "minimum_observations",
                "max_gap_seconds", "minimum_history_span_seconds", "minimum_rows_per_regime_horizon"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(key + " must be a positive integer")
    if minimum < 3 or lookback < minimum:
        raise ValueError("lookback_observations must be >= minimum_observations >= 3")
    for key in ("trend_abs_return_pct", "volatility_high_pct", "volatility_low_pct"):
        if not _finite(config[key]) or config[key] < 0:
            raise ValueError(key + " must be a finite non-negative number")
    if config["volatility_low_pct"] >= config["volatility_high_pct"]:
        raise ValueError("volatility thresholds must satisfy low < high")
    threshold = config["buy_pressure_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0.5 < threshold <= 1:
        raise ValueError("buy_pressure_threshold must be greater than 0.5 and at most 1")

    def parse_stamp(row: dict[str, Any]) -> datetime | None:
        raw = row.get("baseline_at")
        if not isinstance(raw, str):
            return None
        try:
            stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        return stamp if stamp.tzinfo is not None and stamp.utcoffset() is not None else None

    histories: dict[tuple[str, str], list[tuple[datetime, float]]] = {}
    for row in history_rows:
        if not isinstance(row, dict):
            continue
        mint, pool = row.get("mint"), row.get("pool_address")
        features = row.get("features")
        stamp = parse_stamp(row)
        price = features.get("price_usd") if isinstance(features, dict) else None
        if (not isinstance(mint, str) or not mint.strip()
                or not isinstance(pool, str) or not pool.strip()
                or stamp is None or not _finite(price) or price <= 0):
            continue
        histories.setdefault((mint, pool), []).append((stamp, float(price)))

    groups: dict[str, list[float]] = {}
    horizon_groups: dict[str, dict[str, list[float]]] = {}
    horizon_cluster_groups: dict[str, dict[str, dict[str, list[float]]]] = {}
    horizon_regime_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    classified = 0
    unclassified = 0
    reason_counts: dict[str, int] = {}
    sampling = {
        "classified_history_span_seconds": [],
        "classified_median_gap_seconds": [],
        "classified_max_gap_seconds": [],
    }

    def reject(reason: str) -> None:
        nonlocal unclassified
        unclassified += 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    for row in validation_rows:
        if not isinstance(row, dict):
            reject("malformed_validation_row")
            continue
        mint, pool = row.get("mint"), row.get("pool_address")
        stamp = parse_stamp(row)
        outcome = row.get("target_sampled_price_return_pct")
        if (not isinstance(mint, str) or not mint.strip()
                or not isinstance(pool, str) or not pool.strip()
                or stamp is None or not _finite(outcome)):
            reject("invalid_validation_identity_time_or_outcome")
            continue
        eligible = sorted(
            (time, price) for time, price in histories.get((mint, pool), [])
            if time <= stamp
        )
        by_time: dict[datetime, set[float]] = {}
        for time, price in eligible:
            by_time.setdefault(time, set()).add(price)
        if any(len(prices) != 1 for prices in by_time.values()):
            reject("conflicting_prices_at_same_timestamp")
            continue
        points = [(time, next(iter(prices))) for time, prices in sorted(by_time.items())]
        points = points[-lookback:]
        if len(points) < minimum:
            reject("insufficient_prior_baseline_observations")
            continue
        gaps = [
            (points[index][0] - points[index - 1][0]).total_seconds()
            for index in range(1, len(points))
        ]
        span = (points[-1][0] - points[0][0]).total_seconds()
        if not gaps or any(gap <= 0 for gap in gaps):
            reject("invalid_sampling_intervals")
            continue
        if span < config["minimum_history_span_seconds"]:
            reject("insufficient_history_time_span")
            continue
        if max(gaps) > config["max_gap_seconds"]:
            reject("sampling_gap_exceeds_configured_maximum")
            continue

        # Normalize each log return by sqrt(elapsed hours). Thresholds therefore
        # use percent per square-root-hour, not percent per observation.
        normalized_returns = [
            math.log(points[index][1] / points[index - 1][1]) * 100.0
            / math.sqrt(gaps[index - 1] / 3600.0)
            for index in range(1, len(points))
        ]
        if not normalized_returns or not all(math.isfinite(value) for value in normalized_returns):
            reject("invalid_sampled_price_path")
            continue
        trend_return = (points[-1][1] / points[0][1] - 1.0) * 100.0
        mean_return = sum(normalized_returns) / len(normalized_returns)
        variance = sum((value - mean_return) ** 2 for value in normalized_returns) / (len(normalized_returns) - 1)
        volatility = math.sqrt(variance)
        if volatility >= config["volatility_high_pct"]:
            vol_regime = "HIGH_TIME_NORMALIZED_VOLATILITY_PROXY"
        elif volatility <= config["volatility_low_pct"]:
            vol_regime = "LOW_TIME_NORMALIZED_VOLATILITY_PROXY"
        else:
            vol_regime = "MID_TIME_NORMALIZED_VOLATILITY_PROXY"
        if trend_return >= config["trend_abs_return_pct"]:
            trend_regime = "UPWARD_TREND_PROXY"
        elif trend_return <= -config["trend_abs_return_pct"]:
            trend_regime = "DOWNWARD_TREND_PROXY"
        else:
            trend_regime = "RANGE_BOUND_RETURN_PROXY"
        regime = trend_regime + "__" + vol_regime
        value = float(outcome)
        groups.setdefault(regime, []).append(value)
        horizon = row.get("horizon_seconds")
        horizon_key = str(horizon) if isinstance(horizon, int) and not isinstance(horizon, bool) and horizon > 0 else "UNKNOWN"
        horizon_groups.setdefault(horizon_key, {}).setdefault(regime, []).append(value)
        horizon_cluster_groups.setdefault(horizon_key, {}).setdefault(regime, {}).setdefault(mint, []).append(value)
        horizon_regime_rows.setdefault(horizon_key, {}).setdefault(regime, []).append(row)
        sampling["classified_history_span_seconds"].append(span)
        sampling["classified_median_gap_seconds"].append(sorted(gaps)[len(gaps) // 2] if len(gaps) % 2 else (sorted(gaps)[len(gaps)//2-1] + sorted(gaps)[len(gaps)//2]) / 2)
        sampling["classified_max_gap_seconds"].append(max(gaps))
        classified += 1

    def summarize_groups(source: dict[str, list[float]]) -> dict[str, Any]:
        return {name: _summarize(values) for name, values in sorted(source.items())}

    return {
        "status": "TIME_SERIES_PROXY_REGIME_DIAGNOSTICS_ONLY",
        "config": dict(config),
        "classified_row_count": classified,
        "unclassified_row_count": unclassified,
        "unclassified_reasons": reason_counts,
        "regimes": summarize_groups(groups),
        "regimes_by_horizon_seconds": {
            horizon: {
                regime: {
                    "row_count": len(values),
                    "distinct_mint_count": len(horizon_cluster_groups[horizon][regime]),
                    "minimum_rows_required": config["minimum_rows_per_regime_horizon"],
                    "status": (
                        "DESCRIPTIVE_SAMPLE_THRESHOLD_MET"
                        if len(values) >= config["minimum_rows_per_regime_horizon"]
                        else "INSUFFICIENT_REGIME_HORIZON_ROWS"
                    ),
                    "summary": (
                        _summarize(values)
                        if len(values) >= config["minimum_rows_per_regime_horizon"]
                        else None
                    ),
                    "mint_cluster_bootstrap": _cluster_bootstrap_mean_interval(
                        horizon_cluster_groups[horizon][regime]
                    ),
                    "fixed_benchmark_comparison": evaluate_solana_research_folds(
                        [{"fold": 1, "validation": horizon_regime_rows[horizon][regime]}],
                        buy_pressure_threshold=float(config["buy_pressure_threshold"]),
                    ),
                }
                for regime, values in sorted(regimes.items())
            }
            for horizon, regimes in sorted(horizon_groups.items())
        },
        "sampling_quality": {
            "maximum_allowed_gap_seconds": config["max_gap_seconds"],
            "minimum_required_history_span_seconds": config["minimum_history_span_seconds"],
            "classified_rows": classified,
            "mean_history_span_seconds": sum(sampling["classified_history_span_seconds"]) / classified if classified else None,
            "mean_median_gap_seconds": sum(sampling["classified_median_gap_seconds"]) / classified if classified else None,
            "mean_max_gap_seconds": sum(sampling["classified_max_gap_seconds"]) / classified if classified else None,
        },
        "method": {
            "history_scope": "same_mint_and_pool_only",
            "future_baseline_observations_excluded": True,
            "same_timestamp_conflicts": "excluded_not_averaged",
            "trend_proxy": "first_to_last_percent_change_over_sampled_baseline_prices",
            "volatility_proxy": "sample_standard_deviation_of_log_return_pct_divided_by_sqrt_elapsed_hours",
            "volatility_unit": "percent_per_sqrt_hour",
            "time_normalized": True,
            "maximum_gap_enforced": True,
            "minimum_history_span_enforced": True,
            "ohlcv_candles_required": False,
            "minimum_rows_per_regime_horizon_enforced": True,
            "fixed_benchmarks": ["no_trade", "always_long", "fixed_buy_pressure"],
            "buy_pressure_threshold": float(config["buy_pressure_threshold"]),
            "benchmark_threshold_fit_on_validation_data": False,
            "uncertainty_method": "mint_cluster_bootstrap_percentile_when_at_least_20_distinct_mints",
        },
        "limitations": [
            "These are sampled-price path proxies, not exchange-candle trend or realized volatility.",
            "Interval normalization reduces cadence dependence but does not recover intraperiod paths or make irregular provider sampling unbiased.",
            "Rows with gaps above the predeclared maximum or insufficient time span are withheld.",
            "Sparse snapshots can miss intraperiod jumps, wicks, liquidity events and market-wide shocks.",
            "Thresholds are percent per square-root-hour and must be predeclared independently of validation outcomes.",
            "Regime-stratified outcomes remain sampled-mark diagnostics, not executable returns, net P&L, or training authorization.",
        ],
    }


def evaluate_solana_fold_time_series_regime_stability(
    history_rows: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    config: dict[str, Any] | None,
    *,
    minimum_mints_per_fold_regime: int = 3,
) -> dict[str, Any]:
    """Evaluate trend/volatility proxy stability independently in each fold.

    Every fold is classified against its own validation timestamps using only
    same-mint/pool history at or before each row's baseline timestamp.
    """
    if (isinstance(minimum_mints_per_fold_regime, bool)
            or not isinstance(minimum_mints_per_fold_regime, int)
            or minimum_mints_per_fold_regime < 1):
        raise ValueError("minimum_mints_per_fold_regime must be a positive integer")
    if config is None:
        return {
            "status": "NOT_CONFIGURED",
            "folds": [],
            "regimes_by_horizon_seconds": {},
            "minimum_mints_per_fold_regime": minimum_mints_per_fold_regime,
            "note": "No explicit time-series regime config supplied.",
        }
    fold_reports: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, list[dict[str, Any]]]] = {}
    minimum_rows = config.get("minimum_rows_per_regime_horizon")
    for fold_number, fold in enumerate(folds, start=1):
        if not isinstance(fold, dict) or not isinstance(fold.get("validation"), list):
            raise ValueError("each fold must contain a validation list")
        report = evaluate_solana_time_series_regimes(
            history_rows, fold["validation"], config
        )
        fold_strata: dict[str, Any] = {}
        for horizon, regimes in report.get("regimes_by_horizon_seconds", {}).items():
            for regime, stratum in regimes.items():
                key = horizon + "::" + regime
                aggregate.setdefault(key, {}).setdefault("folds", []).append({
                    "fold": fold.get("fold", fold_number),
                    "stratum": stratum,
                })
                distinct_mints = stratum.get("distinct_mint_count", 0)
                row_count = stratum.get("row_count", 0)
                eligible = (
                    isinstance(row_count, int) and row_count >= minimum_rows
                    and isinstance(distinct_mints, int)
                    and distinct_mints >= minimum_mints_per_fold_regime
                )
                benchmark = stratum.get("fixed_benchmark_comparison", {}).get("aggregate", {})
                fold_strata.setdefault(horizon, {})[regime] = {
                    "row_count": row_count,
                    "distinct_mint_count": distinct_mints,
                    "status": "FOLD_SAMPLE_THRESHOLD_MET" if eligible
                        else "INSUFFICIENT_FOLD_ROWS_OR_MINT_DIVERSITY",
                    "stability_eligible": eligible,
                    "baselines": benchmark,
                }
        fold_reports.append({
            "fold": fold.get("fold", fold_number),
            "classified_row_count": report.get("classified_row_count", 0),
            "unclassified_row_count": report.get("unclassified_row_count", 0),
            "regimes_by_horizon_seconds": fold_strata,
        })

    stability: dict[str, Any] = {}
    for key, data in sorted(aggregate.items()):
        horizon, regime = key.split("::", 1)
        baseline_results: dict[str, Any] = {}
        for baseline in ("always_long", "fixed_buy_pressure"):
            means: list[float] = []
            positive = negative = 0
            for item in data["folds"]:
                stratum = item["stratum"]
                if stratum.get("row_count", 0) < minimum_rows:
                    continue
                if stratum.get("distinct_mint_count", 0) < minimum_mints_per_fold_regime:
                    continue
                summary = stratum.get("fixed_benchmark_comparison", {}).get("aggregate", {}).get(baseline, {})
                mean = summary.get("mean_return_pct")
                if not _finite(mean):
                    continue
                means.append(float(mean))
                positive += mean > 0
                negative += mean < 0
            count = len(means)
            baseline_results[baseline] = {
                "eligible_fold_count": count,
                "positive_mean_fold_count": positive,
                "negative_mean_fold_count": negative,
                "mean_return_sign_consistency": max(positive, negative) / count if count else None,
                "fold_mean_returns_pct": means,
                "status": "DESCRIPTIVE_FOLD_STABILITY_AVAILABLE" if count
                    else "INSUFFICIENT_ELIGIBLE_FOLDS",
            }
        stability.setdefault(horizon, {})[regime] = baseline_results
    return {
        "status": "CHRONOLOGICAL_TIME_SERIES_REGIME_STABILITY_DIAGNOSTICS_ONLY",
        "minimum_rows_per_fold_regime": minimum_rows,
        "minimum_mints_per_fold_regime": minimum_mints_per_fold_regime,
        "folds": fold_reports,
        "regimes_by_horizon_seconds": stability,
        "future_history_excluded_per_validation_row": True,
        "model_fitting_performed": False,
        "training_authorized": False,
        "note": "Each fold is classified independently using same-mint/pool history at or before each baseline timestamp. Fold means are sampled-mark diagnostics; sign consistency is not significance or proof of edge, and overlapping labels may remain dependent.",
    }


def evaluate_prepared_solana_walk_forward(
    exported_labels: list[dict[str, Any]],
    prepared_dataset: dict[str, Any],
    *,
    n_splits: int = 3,
    min_train_groups: int = 3,
    embargo_seconds: int = 86400,
    buy_pressure_threshold: float = 0.55,
    execution_cost_evidence_by_label_id: dict[Any, dict[str, Any]] | None = None,
    transaction_attestations_by_label_id: dict[Any, list[dict[str, Any]]] | None = None,
    regime_thresholds: dict[str, float] | None = None,
    time_series_regime_config: dict[str, Any] | None = None,
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

    if execution_cost_evidence_by_label_id is not None and not isinstance(execution_cost_evidence_by_label_id, dict):
        raise ValueError("execution_cost_evidence_by_label_id must be a dictionary")
    if execution_cost_evidence_by_label_id is not None:
        for evidence_label_id, evidence_value in execution_cost_evidence_by_label_id.items():
            if not isinstance(evidence_label_id, (str, int)) or isinstance(evidence_label_id, bool):
                raise ValueError("execution cost evidence keys must be string or integer label IDs")
            if not isinstance(evidence_value, dict):
                raise ValueError("each execution cost evidence value must be a dictionary")

    if transaction_attestations_by_label_id is not None:
        if not isinstance(transaction_attestations_by_label_id, dict):
            raise ValueError("transaction_attestations_by_label_id must be a dictionary")
        for attestation_label_id, records in transaction_attestations_by_label_id.items():
            if not isinstance(attestation_label_id, (str, int)) or isinstance(attestation_label_id, bool):
                raise ValueError("transaction attestation keys must be string or integer label IDs")
            if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
                raise ValueError("transaction attestation values must be lists of records")

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
    validation_intervals: list[tuple[str, datetime, datetime, int]] = []
    validation_cost_evidence: list[dict[str, Any]] = []
    validation_transaction_attestations: list[dict[str, Any]] = []
    for index, fold in enumerate(folds, start=1):
        validation_rows: list[dict[str, Any]] = []
        missing_prepared = 0
        for label in fold["validation"]:
            prepared = rows_by_label_id.get(label.get("id"))
            if prepared is None:
                missing_prepared += 1
                continue
            if (
                prepared.get("mint") != label.get("mint")
                or prepared.get("pool_address") != label.get("pool_address")
            ):
                raise ValueError("prepared row identity does not match its exported label")
            validation_rows.append(prepared)
            if transaction_attestations_by_label_id is not None:
                import hashlib
                import json
                expected_identity = {
                    "label_id": label.get("id"),
                    "mint": label.get("mint"),
                    "pool_address": label.get("pool_address"),
                }
                supplied_records = transaction_attestations_by_label_id.get(label.get("id"), [])
                integrity_valid_count = 0
                receipt_observation_count = 0
                rejected_record_count = 0
                for record in supplied_records:
                    attestation = record.get("attestation")
                    digest = record.get("attestation_sha256")
                    try:
                        encoded = json.dumps(
                            attestation, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                        )
                        digest_valid = hashlib.sha256(encoded.encode("utf-8")).hexdigest() == digest
                    except (TypeError, ValueError):
                        digest_valid = False
                    record_identity = {
                        "label_id": record.get("label_id"),
                        "mint": record.get("mint"),
                        "pool_address": record.get("pool_address"),
                    }
                    if record.get("integrity_valid") is True and digest_valid and record_identity == expected_identity:
                        integrity_valid_count += 1
                        if (
                            isinstance(attestation, dict)
                            and attestation.get("transaction_receipt_verified") is True
                            and attestation.get("verified") is False
                            and attestation.get("status") == "FINALIZED_MINT_BALANCE_DEBIT_OBSERVED"
                            and attestation.get("signature") == record.get("signature")
                            and attestation.get("mint") == expected_identity["mint"]
                            and attestation.get("expected_pool_address") == expected_identity["pool_address"]
                            and attestation.get("wallet_address") == record.get("wallet_address")
                        ):
                            receipt_observation_count += 1
                        else:
                            rejected_record_count += 1
                    else:
                        rejected_record_count += 1
                validation_transaction_attestations.append({
                    **expected_identity,
                    "horizon_seconds": prepared.get("horizon_seconds"),
                    "attestation_record_count": len(supplied_records),
                    "integrity_valid_record_count": integrity_valid_count,
                    "finalized_receipt_observation_count": receipt_observation_count,
                    "rejected_record_count": rejected_record_count,
                })
            if execution_cost_evidence_by_label_id is not None:
                from eti.solana_costs import evaluate_execution_cost_evidence
                evidence = execution_cost_evidence_by_label_id.get(label.get("id"))
                cost_result = evaluate_execution_cost_evidence(
                    prepared.get("target_sampled_price_return_pct"),
                    evidence,
                )
                expected_identity = {
                    "label_id": label.get("id"),
                    "mint": label.get("mint"),
                    "pool_address": label.get("pool_address"),
                }
                supplied_identity = {
                    key: evidence.get(key) for key in expected_identity
                } if isinstance(evidence, dict) else {}
                if supplied_identity != expected_identity:
                    cost_result["status"] = "COST_EVIDENCE_INCOMPLETE"
                    cost_result["cost_evidence_complete"] = False
                    cost_result["cost_adjusted_sampled_mark_scenario_return_pct"] = None
                    cost_result["measured_round_trip_cost_bps"] = None
                    cost_result["invalid_components"].append(
                        "evidence_identity_mismatch" if supplied_identity else "evidence_identity_missing"
                    )
                validation_cost_evidence.append({
                    **expected_identity,
                    "horizon_seconds": prepared.get("horizon_seconds"),
                    **cost_result,
                })
            horizon = prepared.get("horizon_seconds")
            if isinstance(horizon, int) and not isinstance(horizon, bool) and horizon > 0:
                horizon_folds.setdefault(horizon, []).append({"fold": index, "row": prepared})
            try:
                start = datetime.fromisoformat(str(label["baseline_at"]).replace("Z", "+00:00"))
                end = datetime.fromisoformat(str(label["target_at"]).replace("Z", "+00:00"))
                mint = label.get("mint")
                if isinstance(mint, str) and start.tzinfo is not None and end.tzinfo is not None and end > start:
                    validation_intervals.append((mint, start, end, index))
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
    validation_rows_for_regimes = [
        row for fold in benchmark_folds for row in fold["validation"]
    ]
    report["regime_diagnostics"] = evaluate_solana_regime_diagnostics(
        validation_rows_for_regimes, regime_thresholds
    )
    report["fold_regime_stability"] = evaluate_solana_fold_regime_stability(
        benchmark_folds, regime_thresholds,
        buy_pressure_threshold=buy_pressure_threshold,
    )
    report["time_series_regime_diagnostics"] = evaluate_solana_time_series_regimes(
        prepared_rows, validation_rows_for_regimes, time_series_regime_config
    )
    report["fold_time_series_regime_stability"] = evaluate_solana_fold_time_series_regime_stability(
        prepared_rows, benchmark_folds, time_series_regime_config,
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
    fold_mints: dict[int, set[str]] = {}
    for mint, start, end, fold_number in validation_intervals:
        intervals_by_mint.setdefault(mint, []).append((start, end))
        fold_mints.setdefault(fold_number, set()).add(mint)
    for intervals in intervals_by_mint.values():
        intervals.sort(key=lambda pair: pair[0])
        active: list[tuple[datetime, datetime]] = []
        for start, end in intervals:
            active = [(a, b) for a, b in active if b > start]
            overlap_comparisons += len(active)
            overlapping_pairs += sum(1 for a, b in active if a < end and start < b)
            active.append((start, end))

    # Cross-mint interval overlap is a separate dependence channel: distinct
    # token identities can still share the same market-wide shock window.
    all_intervals = sorted(validation_intervals, key=lambda item: item[1])
    active_global: list[tuple[str, datetime, datetime]] = []
    cross_mint_overlapping_pairs = 0
    cross_mint_overlap_comparisons = 0
    for mint, start, end, _fold_number in all_intervals:
        active_global = [(other, a, b) for other, a, b in active_global if b > start]
        cross_mint_overlap_comparisons += sum(1 for other, _a, _b in active_global if other != mint)
        cross_mint_overlapping_pairs += sum(
            1 for other, a, b in active_global
            if other != mint and a < end and start < b
        )
        active_global.append((mint, start, end))
    mint_fold_counts: dict[str, int] = {}
    for mint_set in fold_mints.values():
        for mint in mint_set:
            mint_fold_counts[mint] = mint_fold_counts.get(mint, 0) + 1
    repeated_mint_across_fold_count = sum(1 for count in mint_fold_counts.values() if count > 1)

    report["by_horizon_seconds"] = by_horizon
    if transaction_attestations_by_label_id is None:
        report["transaction_receipt_attestations"] = {
            "status": "NOT_SUPPLIED",
            "validation_rows_checked": 0,
            "attestation_record_count": 0,
            "integrity_valid_record_count": 0,
            "finalized_receipt_observation_count": 0,
            "rejected_record_count": 0,
            "verified_swap_count": 0,
            "net_return_pct": None,
            "realized_pnl_available": False,
            "note": "No receipt attestations were supplied. Receipt observations are not verified swaps, exits, fills, or P&L.",
        }
    else:
        report["transaction_receipt_attestations"] = {
            "status": "RECEIPT_OBSERVATION_DIAGNOSTICS_ONLY",
            "validation_rows_checked": len(validation_transaction_attestations),
            "attestation_record_count": sum(row["attestation_record_count"] for row in validation_transaction_attestations),
            "integrity_valid_record_count": sum(row["integrity_valid_record_count"] for row in validation_transaction_attestations),
            "finalized_receipt_observation_count": sum(row["finalized_receipt_observation_count"] for row in validation_transaction_attestations),
            "rejected_record_count": sum(row["rejected_record_count"] for row in validation_transaction_attestations),
            "verified_swap_count": 0,
            "net_return_pct": None,
            "realized_pnl_available": False,
            "rows": validation_transaction_attestations,
            "note": "Digest and label identity checks support record-integrity diagnostics only. Finalized receipt observations do not prove swap causality, fill price, successful exit, transaction costs, or realized P&L.",
        }
    if execution_cost_evidence_by_label_id is None:
        report["execution_cost_evidence"] = {
            "status": "NOT_SUPPLIED",
            "validation_rows_checked": 0,
            "complete_evidence_count": 0,
            "incomplete_evidence_count": 0,
            "net_return_pct": None,
            "realized_pnl_available": False,
            "note": "No execution-cost evidence was supplied; no cost-adjusted scenario is calculated.",
        }
    else:
        complete_cost_rows = [row for row in validation_cost_evidence if row["cost_evidence_complete"] and _finite(row["cost_adjusted_sampled_mark_scenario_return_pct"])]
        scenario_returns = [float(row["cost_adjusted_sampled_mark_scenario_return_pct"]) for row in complete_cost_rows]
        ordered_scenarios = sorted(scenario_returns)
        median_scenario = None
        if ordered_scenarios:
            middle = len(ordered_scenarios) // 2
            median_scenario = ordered_scenarios[middle] if len(ordered_scenarios) % 2 else (ordered_scenarios[middle - 1] + ordered_scenarios[middle]) / 2
        report["execution_cost_evidence"] = {
            "status": "COST_EVIDENCE_DIAGNOSTICS_ONLY",
            "validation_rows_checked": len(validation_cost_evidence),
            "complete_evidence_count": len(complete_cost_rows),
            "incomplete_evidence_count": len(validation_cost_evidence) - len(complete_cost_rows),
            "cost_adjusted_sampled_mark_scenario_mean_pct": sum(scenario_returns) / len(scenario_returns) if scenario_returns else None,
            "cost_adjusted_sampled_mark_scenario_median_pct": median_scenario,
            "net_return_pct": None,
            "realized_pnl_available": False,
            "rows": validation_cost_evidence,
            "note": "Only prepared validation labels are checked. These are cost-adjusted sampled-mark scenarios, not executable fills, net returns, or realized P&L.",
        }
    fold_validation_label_count = sum(item["validation_label_count"] for item in fold_metadata)
    prepared_validation_interval_count = len(validation_intervals)
    report["validation_dependence"] = {
        "fold_validation_label_count": fold_validation_label_count,
        "unprepared_validation_label_count": sum(item["unprepared_validation_count"] for item in fold_metadata),
        "prepared_validation_interval_count": prepared_validation_interval_count,
        "validation_label_count": prepared_validation_interval_count,
        "distinct_validation_mint_count": len(intervals_by_mint),
        "repeated_mint_label_count": len(validation_intervals) - len(intervals_by_mint),
        "dependence_scope": "PREPARED_VALIDATION_ROWS_WITH_VALID_INTERVALS_ONLY",
        "overlapping_interval_pair_count": overlapping_pairs,
        "overlap_comparison_count": overlap_comparisons,
        "cross_mint_overlapping_interval_pair_count": cross_mint_overlapping_pairs,
        "cross_mint_overlap_comparison_count": cross_mint_overlap_comparisons,
        "distinct_mints_repeated_across_validation_folds": repeated_mint_across_fold_count,
        "validation_fold_count_with_prepared_intervals": len(fold_mints),
        "note": "Dependence counts cover prepared validation rows with parseable, timezone-aware, positive-duration intervals; fold_validation_label_count also includes unprepared labels. Same-mint and cross-mint interval overlap are separate descriptive dependence channels. Repeated mints across folds are not independent replications.",
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
