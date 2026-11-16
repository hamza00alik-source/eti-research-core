"""Synchronized, research-only portfolio evaluation across outer-test folds.

This module combines timestamped per-instrument net return streams. It does not
fit models, allocate live capital, promote models, or authorize execution.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _fold_stream(market: dict[str, Any], market_key: str, fold_id: int) -> tuple[pd.Series, pd.Series]:
    report = market.get("report")
    performance = report.get("trading_performance") if isinstance(report, dict) else None
    folds = performance.get("folds") if isinstance(performance, dict) else None
    if not isinstance(folds, list):
        raise ValueError(f"{market_key} is missing trading-performance folds")
    matches = [row for row in folds if isinstance(row, dict) and row.get("fold") == fold_id]
    if len(matches) != 1:
        raise ValueError(f"{market_key} must contain exactly one fold {fold_id}")
    fold = matches[0]
    trading = fold.get("trading")
    model = trading.get("model") if isinstance(trading, dict) else None
    timestamps = fold.get("timestamps")
    returns = model.get("bar_returns") if isinstance(model, dict) else None
    positions = model.get("positions") if isinstance(model, dict) else None
    if not isinstance(timestamps, list) or not isinstance(returns, list) or not isinstance(positions, list):
        raise ValueError(f"{market_key} fold {fold_id} lacks timestamped return/position streams")
    if len(timestamps) < 2 or len(timestamps) != len(returns) or len(timestamps) != len(positions):
        raise ValueError(f"{market_key} fold {fold_id} stream lengths are invalid")
    index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="coerce"))
    if index.isna().any() or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError(f"{market_key} fold {fold_id} timestamps must be unique, valid and chronological")
    return_values = np.asarray([_number(v, f"{market_key} bar return") for v in returns], dtype=float)
    position_values = np.asarray([_number(v, f"{market_key} position") for v in positions], dtype=float)
    if (return_values <= -1).any() or (np.abs(position_values) > 1).any():
        raise ValueError(f"{market_key} fold {fold_id} contains invalid returns or positions")
    return (
        pd.Series(return_values, index=index, name=market_key),
        pd.Series(position_values, index=index, name=market_key),
    )


def _fold_portfolio(markets: dict[str, Any], weights: dict[str, float],
                    fold_id: int, bars_per_year: float) -> dict[str, Any]:
    streams = {}
    positions = {}
    for key in weights:
        streams[key], positions[key] = _fold_stream(markets[key], key, fold_id)

    # Union timestamps preserve each venue's observed bars. A market that is
    # closed/missing at a timestamp contributes cash (zero return/exposure);
    # weights are never silently re-normalized.
    weight_series = pd.Series(weights, dtype=float)
    weighted_returns = returns.mul(weight_series, axis=1)
    portfolio_returns = weighted_returns.sum(axis=1)
    gross_exposure = position_frame.abs().mul(weight_series, axis=1).sum(axis=1)
    net_exposure = position_frame.mul(weight_series, axis=1).sum(axis=1)
    equity = (1.0 + portfolio_returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    volatility = float(portfolio_returns.std(ddof=1))
    sharpe = (
        float(portfolio_returns.mean() / volatility * math.sqrt(bars_per_year))
        if volatility > 0 else 0.0
    )
    corr = raw_returns.corr(min_periods=2)
    correlation = {
        left: {
            right: (float(corr.loc[left, right]) if pd.notna(corr.loc[left, right]) else None)
            for right in weights
        }
        for left in weights
    }
    active_weights = position_frame.abs().mul(weight_series, axis=1)
    active_total = active_weights.sum(axis=1).replace(0, np.nan)
    shares = active_weights.div(active_total, axis=0)
    hhi = (shares.pow(2).sum(axis=1)).dropna()
    variance = float(portfolio_returns.var(ddof=1))
    risk_contribution = {}
    if variance > 0 and len(weights) > 1:
        covariance = returns.cov(min_periods=2).reindex(index=weights, columns=weights).fillna(0.0)
        covariance_values = covariance.to_numpy(dtype=float)
        w = weight_series.to_numpy(dtype=float)
        marginal = covariance_values @ w
        contributions = w * marginal / variance
        risk_contribution = {key: float(value) for key, value in zip(weights, contributions)}
    return {
        "fold": fold_id,
        "bar_count": int(len(portfolio_returns)),
        "start": portfolio_returns.index[0].isoformat(),
        "end": portfolio_returns.index[-1].isoformat(),
        "total_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "sharpe_annualized": sharpe,
        "mean_gross_exposure_fraction": float(gross_exposure.mean()),
        "max_gross_exposure_fraction": float(gross_exposure.max()),
        "mean_net_exposure_fraction": float(net_exposure.mean()),
        "mean_active_weight_hhi": float(hhi.mean()) if len(hhi) else 0.0,
        "maximum_weight_fraction": float(max(weights.values())),
        "correlation_matrix": correlation,
        "variance_risk_contribution": risk_contribution,
        "synchronized_asset_bar_coverage": {
            key: float(len(streams[key].index.intersection(portfolio_returns.index)) / len(portfolio_returns))
            for key in weights
        },
    }


def build_portfolio_quality_report(
    result: dict[str, Any],
    weights: dict[str, float],
    *,
    minimum_folds: int = 5,
    bars_per_year: float | None = None,
) -> dict[str, Any]:
    """Combine exact instrument keys with fixed capital weights (unallocated weight is cash)."""
    if not isinstance(result, dict) or result.get("schema_version") != "eti.multi_market_research_result.v1":
        raise ValueError("unsupported multi-market research result schema")
    if result.get("training_performed") is not True or result.get("execution_enabled") is not False:
        raise ValueError("source must confirm completed research and disabled execution")
    markets = result.get("markets")
    if not isinstance(markets, dict) or not markets:
        raise ValueError("source market results are required")
    if not isinstance(weights, dict) or len(weights) < 2:
        raise ValueError("portfolio weights must include at least two exact instrument keys")
    normalized = {}
    for key, raw_weight in weights.items():
        if not isinstance(key, str) or key not in markets:
            raise ValueError(f"portfolio weight references unknown instrument: {key}")
        market = markets[key]
        if not isinstance(market, dict) or market.get("status") != "evaluated":
            raise ValueError(f"portfolio instrument {key} has no completed evaluation")
        weight = _number(raw_weight, f"weight for {key}")
        if weight <= 0 or weight > 1:
            raise ValueError(f"weight for {key} must be in (0, 1]")
        normalized[key] = weight
    total_weight = sum(normalized.values())
    if total_weight > 1.0 + 1e-9:
        raise ValueError("portfolio weights cannot exceed 1.0; residual capital is cash")
    if isinstance(minimum_folds, bool) or not isinstance(minimum_folds, int) or minimum_folds < 1:
        raise ValueError("minimum_folds must be a positive integer")
    timeframes = set()
    for key in normalized:
        report = markets[key].get("report")
        timeframe = report.get("timeframe") if isinstance(report, dict) else None
        if not isinstance(timeframe, str) or not timeframe:
            raise ValueError(f"{key} must declare its research timeframe")
        timeframes.add(timeframe)
    if len(timeframes) != 1:
        raise ValueError("weighted instruments must use the same research timeframe")
    timeframe = next(iter(timeframes))
    match = re.fullmatch(r"(\\d+)([smhdw])", timeframe)
    if not match:
        raise ValueError(f"unsupported fixed-duration portfolio timeframe: {timeframe}")
    amount, unit = int(match.group(1)), match.group(2)
    timeframe_seconds = amount * {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
    if timeframe_seconds <= 0:
        raise ValueError("portfolio timeframe must be positive")
    if bars_per_year is None:
        bars_per_year = 365.25 * 24 * 60 * 60 / timeframe_seconds
    bars_per_year = _number(bars_per_year, "bars_per_year")
    if bars_per_year <= 0:
        raise ValueError("bars_per_year must be positive")

    fold_lists = {}
    for key in normalized:
        report = markets[key].get("report")
        performance = report.get("trading_performance") if isinstance(report, dict) else None
        rows = performance.get("folds") if isinstance(performance, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{key} has no outer-test folds")
        ids = [row.get("fold") for row in rows if isinstance(row, dict)]
        if len(ids) != len(rows) or any(isinstance(v, bool) or not isinstance(v, int) for v in ids):
            raise ValueError(f"{key} fold identifiers are invalid")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{key} has duplicate fold identifiers")
        fold_lists[key] = set(ids)
    fold_sets = list(fold_lists.values())
    if any(item != fold_sets[0] for item in fold_sets[1:]):
        raise ValueError("weighted instruments must have identical outer-test fold identifiers")
    common_folds = sorted(fold_sets[0])
    if not common_folds:
        raise ValueError("weighted instruments have no outer-test folds")

    folds = [_fold_portfolio(markets, normalized, fold_id, bars_per_year) for fold_id in common_folds]
    returns = [item["total_return"] for item in folds]
    drawdowns = [item["max_drawdown"] for item in folds]
    evidence_status = "human_review_required" if len(folds) >= minimum_folds else "insufficient_synchronized_folds"
    return {
        "schema_version": "eti.portfolio_quality_report.v1",
        "source_schema_version": result["schema_version"],
        "weights": normalized,
        "cash_weight_fraction": float(1.0 - total_weight),
        "fold_count": len(folds),
        "minimum_folds": minimum_folds,
        "mean_fold_return": float(np.mean(returns)),
        "median_fold_return": float(np.median(returns)),
        "positive_return_folds": int(sum(value > 0 for value in returns)),
        "mean_fold_max_drawdown": float(np.mean(drawdowns)),
        "worst_fold_max_drawdown": float(min(drawdowns)),
        "mean_gross_exposure_fraction": float(np.mean([x["mean_gross_exposure_fraction"] for x in folds])),
        "maximum_observed_gross_exposure_fraction": float(max(x["max_gross_exposure_fraction"] for x in folds)),
        "mean_active_weight_hhi": float(np.mean([x["mean_active_weight_hhi"] for x in folds])),
        "evidence_status": evidence_status,
        "folds": folds,
        "model_promotion_authority": False,
        "budget_authority": False,
        "execution_authority": False,
        "limitations": [
            "Fixed operator-supplied weights are research assumptions, not an optimized allocation.",
            "Unallocated weight remains cash; closed/missing instruments contribute zero return without weight renormalization.",
            "Each instrument's independently trained strategy is combined only on timestamped outer-test streams; this does not simulate shared capital constraints, order queueing, partial fills, market impact, margin, or cross-currency conversion.",
            "Correlation is descriptive and may be unstable with few overlapping observations; fold windows may overlap and are not independent trials.",
            "This report cannot establish future profitability or authorize model promotion, budget changes, or live execution.",
        ],
        "warning": "Portfolio research evidence only. Human review and independent execution validation remain mandatory.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synchronized ETI portfolio research evidence")
    parser.add_argument("input", type=Path, help="eti.multi_market_research_result.v1 JSON")
    parser.add_argument("--weights", required=True,
                        help="JSON object mapping exact venue:market_type:symbol keys to fractions")
    parser.add_argument("--output", type=Path, default=Path("artifacts/portfolio-quality-report.json"))
    parser.add_argument("--minimum-folds", type=int, default=5)
    parser.add_argument("--bars-per-year", type=float, default=None)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    weights = json.loads(args.weights)
    report = build_portfolio_quality_report(
        source, weights, minimum_folds=args.minimum_folds, bars_per_year=args.bars_per_year
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output), "fold_count": report["fold_count"],
        "evidence_status": report["evidence_status"], "execution_enabled": False,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
