"""Conservative summaries of nested out-of-sample research evidence.

This module consumes completed research artifacts. It does not train models,
rank live workers, approve candidates, allocate money, or authorize execution.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _scenario(scenarios: list[dict[str, Any]], multiplier: float) -> dict[str, Any]:
    matches = [
        item for item in scenarios
        if isinstance(item, dict)
        and _finite_number(item.get("cost_multiplier"), "cost_multiplier") == multiplier
    ]
    if len(matches) != 1:
        raise ValueError(f"exactly one {multiplier:g}x cost-sensitivity scenario is required")
    return matches[0]




def _probability_calibration_evidence(report: dict[str, Any], fold_count: int) -> dict[str, Any]:
    """Validate fold-level calibration diagnostics and their aggregate summary."""
    aggregate = report.get("probability_calibration")
    if aggregate is None:
        return {"status": "not_available", "method": None,
                "outer_test_assessed_folds": 0, "outer_test_confirmed_folds": 0}
    if not isinstance(aggregate, dict) or aggregate.get("method") != "fold_local_temperature_scaling":
        raise ValueError("unsupported probability-calibration evidence")
    declared_folds = aggregate.get("fold_count")
    if isinstance(declared_folds, bool) or not isinstance(declared_folds, int) or declared_folds != fold_count:
        raise ValueError("aggregate probability-calibration fold_count does not match trading folds")
    folds = report.get("folds")
    if not isinstance(folds, list) or len(folds) != fold_count:
        raise ValueError("fold-level probability-calibration records must match trading folds")

    raw_nll, raw_brier, calibrated_nll, calibrated_brier = [], [], [], []
    assessed = confirmed = attempted = 0
    valid_statuses = {
        "calibration_confirmed_on_outer_test",
        "insufficient_outer_test_class_coverage",
        "outer_test_nll_worse_than_raw",
        "outer_test_brier_worse_than_raw",
        "not_assessed",
    }

    def validate_metrics(value: Any, index: int, label: str, expected_count: int) -> tuple[float, float]:
        if not isinstance(value, dict):
            raise ValueError(f"fold {index} {label} probability metrics must be an object")
        sample_count = value.get("sample_count")
        counts = value.get("class_counts")
        if (isinstance(sample_count, bool) or not isinstance(sample_count, int)
                or sample_count != expected_count or not isinstance(counts, dict)):
            raise ValueError(f"fold {index} {label} probability sample/class counts are invalid")
        normalized_counts = {}
        for cls in ("0", "1", "2"):
            count = counts.get(cls)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"fold {index} {label} class counts are invalid")
            normalized_counts[cls] = count
        if sum(normalized_counts.values()) != sample_count:
            raise ValueError(f"fold {index} {label} class counts do not sum to sample_count")
        nll = _finite_number(value.get("negative_log_likelihood"), f"fold {index} {label} NLL")
        brier = _finite_number(value.get("brier_score"), f"fold {index} {label} Brier score")
        if nll < 0 or not 0 <= brier <= 2:
            raise ValueError(f"fold {index} {label} probability metrics are outside valid bounds")
        return nll, brier

    for index, fold in enumerate(folds):
        item = fold.get("probability_calibration") if isinstance(fold, dict) else None
        if not isinstance(item, dict):
            raise ValueError(f"fold {index} is missing probability-calibration evidence")
        fit = item.get("calibration_fit")
        if not isinstance(fit, dict):
            raise ValueError(f"fold {index} calibration-fit evidence is missing")
        fit_status = fit.get("status")
        if fit_status in ("fit", "not_calibrated"):
            attempted += 1
        if fit_status == "fit" and (fit.get("passed") is not True
                                    or fit.get("method") != "temperature_scaling"):
            raise ValueError(f"fold {index} claims a calibration fit without a passed temperature fit")
        raw = item.get("raw")
        scaled = item.get("temperature_scaled")
        actual = fold.get("test_actual")
        if not isinstance(actual, list) or not actual:
            raise ValueError(f"fold {index} outer-test labels are required for calibration evidence")
        expected_count = len(actual)
        if raw is not None:
            nll, brier = validate_metrics(raw, index, "raw", expected_count)
            raw_nll.append(nll)
            raw_brier.append(brier)
        if scaled is not None:
            if raw is None or fit_status != "fit" or fit.get("passed") is not True:
                raise ValueError(f"fold {index} calibrated metrics lack a successful fit and raw baseline")
            nll, brier = validate_metrics(scaled, index, "calibrated", expected_count)
            calibrated_nll.append(nll)
            calibrated_brier.append(brier)
            assessed += 1

        confirmation = item.get("outer_test_confirmation")
        if confirmation not in valid_statuses:
            raise ValueError(f"fold {index} outer-test calibration confirmation is invalid")
        item_status = item.get("status")
        if scaled is not None:
            if confirmation == "calibration_confirmed_on_outer_test":
                if (item_status != "calibrated_and_outer_test_confirmed"
                        or item.get("outer_test_class_coverage_passed") is not True
                        or item.get("outer_test_nll_not_worse") is not True
                        or item.get("outer_test_brier_not_worse") is not True):
                    raise ValueError(f"fold {index} calibration confirmation flags are inconsistent")
                confirmed += 1
            elif item_status != "calibration_not_confirmed_on_outer_test":
                raise ValueError(f"fold {index} unconfirmed calibration status is inconsistent")
        else:
            if confirmation != "not_assessed" or item_status not in ("uncalibrated", "unavailable"):
                raise ValueError(f"fold {index} unassessed calibration status is inconsistent")

    def mean(values: list[float]) -> float | None:
        return float(sum(values) / len(values)) if values else None

    supplied_attempted = aggregate.get("calibration_attempted_folds")
    if (isinstance(supplied_attempted, bool) or not isinstance(supplied_attempted, int)
            or supplied_attempted != attempted):
        raise ValueError("aggregate calibration_attempted_folds does not match fold evidence")
    expected = {
        "outer_test_assessed_folds": assessed,
        "outer_test_confirmed_folds": confirmed,
        "mean_raw_outer_test_nll": mean(raw_nll),
        "mean_raw_outer_test_brier": mean(raw_brier),
        "mean_calibrated_outer_test_nll": mean(calibrated_nll),
        "mean_calibrated_outer_test_brier": mean(calibrated_brier),
    }
    for key, actual_value in expected.items():
        supplied = aggregate.get(key)
        if isinstance(actual_value, int):
            if isinstance(supplied, bool) or not isinstance(supplied, int) or supplied != actual_value:
                raise ValueError(f"aggregate probability-calibration field {key} does not match fold evidence")
        elif actual_value is None:
            if supplied is not None:
                raise ValueError(f"aggregate probability-calibration field {key} must be null")
        else:
            value = _finite_number(supplied, f"aggregate {key}")
            if not math.isclose(value, actual_value, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"aggregate probability-calibration field {key} does not match fold evidence")

    if confirmed == fold_count and assessed == fold_count:
        status = "confirmed_all_folds"
    elif confirmed:
        status = "partial_confirmation"
    elif assessed:
        status = "not_confirmed"
    else:
        status = "no_independent_assessment"
    return {"status": status, "method": aggregate["method"], **expected}


def summarize_market_research(market_key: str, market_result: dict[str, Any],
                              *, minimum_folds: int = 5) -> dict[str, Any]:
    """Summarize strictly validated outer-test folds for one evaluated instrument."""
    if not isinstance(market_key, str) or not market_key.strip():
        raise ValueError("market_key is required")
    if not isinstance(market_result, dict) or market_result.get("status") != "evaluated":
        raise ValueError("market result must have status='evaluated'")
    report = market_result.get("report")
    if not isinstance(report, dict) or report.get("training_performed") is not True:
        raise ValueError("evaluated market must contain a completed training report")
    performance = report.get("trading_performance")
    folds = performance.get("folds") if isinstance(performance, dict) else None
    if not isinstance(folds, list) or not folds:
        raise ValueError("nested out-of-sample trading folds are required")
    if isinstance(minimum_folds, bool) or not isinstance(minimum_folds, int) or minimum_folds < 1:
        raise ValueError("minimum_folds must be a positive integer")

    net_returns, drawdowns, sharpes, accuracies = [], [], [], []
    beats_buy_hold = beats_no_trade = 0
    for index, fold in enumerate(folds):
        if not isinstance(fold, dict):
            raise ValueError(f"fold {index} must be an object")
        trading = fold.get("trading")
        if not isinstance(trading, dict):
            raise ValueError(f"fold {index} is missing trading metrics")
        model = trading.get("model")
        buy_hold = trading.get("buy_and_hold")
        no_trade = trading.get("no_trade")
        if not all(isinstance(item, dict) for item in (model, buy_hold, no_trade)):
            raise ValueError(f"fold {index} is missing a benchmark or model metric")
        net = _finite_number(model.get("total_return"), f"fold {index} total_return")
        drawdown = _finite_number(model.get("max_drawdown"), f"fold {index} max_drawdown")
        sharpe = _finite_number(model.get("sharpe_annualized"), f"fold {index} sharpe")
        buy_return = _finite_number(buy_hold.get("total_return"), f"fold {index} buy_hold return")
        flat_return = _finite_number(no_trade.get("total_return"), f"fold {index} no_trade return")
        if net <= -1 or drawdown < -1 or drawdown > 0:
            raise ValueError(f"fold {index} contains impossible return/drawdown values")
        classification = fold.get("classification")
        if not isinstance(classification, dict):
            raise ValueError(f"fold {index} is missing classification metrics")
        accuracy = _finite_number(
            classification.get("balanced_accuracy"),
            f"fold {index} balanced_accuracy",
        )
        if not 0 <= accuracy <= 1:
            raise ValueError(f"fold {index} balanced_accuracy must be within [0, 1]")
        net_returns.append(net)
        drawdowns.append(drawdown)
        sharpes.append(sharpe)
        accuracies.append(accuracy)
        beats_buy_hold += int(net > buy_return)
        beats_no_trade += int(net > flat_return)

    sensitivity = report.get("cost_sensitivity")
    scenarios = sensitivity.get("scenarios") if isinstance(sensitivity, dict) else None
    if not isinstance(scenarios, list):
        raise ValueError("cost-sensitivity scenarios are required")
    base = _scenario(scenarios, 1.0)
    stressed = _scenario(scenarios, 2.0)
    for name, scenario in (("base", base), ("2x", stressed)):
        scenario_folds = scenario.get("fold_count")
        if isinstance(scenario_folds, bool) or not isinstance(scenario_folds, int) or scenario_folds != len(folds):
            raise ValueError(f"{name} cost scenario fold_count does not match outer-test folds")
        _finite_number(scenario.get("mean_net_return"), f"{name} mean_net_return")
        positive = scenario.get("positive_return_folds")
        if isinstance(positive, bool) or not isinstance(positive, int) or not 0 <= positive <= len(folds):
            raise ValueError(f"{name} positive_return_folds is invalid")

    stressed_mean = float(stressed["mean_net_return"])
    if len(folds) < minimum_folds:
        evidence_status = "insufficient_folds"
    elif stressed_mean <= 0:
        evidence_status = "not_robust_at_2x_costs"
    else:
        evidence_status = "human_review_required"

    calibration_evidence = _probability_calibration_evidence(report, len(folds))

    return {
        "market_key": market_key,
        "probability_calibration": calibration_evidence,
        "outer_test_fold_count": len(folds),
        "mean_net_return": sum(net_returns) / len(net_returns),
        "median_net_return": median(net_returns),
        "positive_return_folds": sum(value > 0 for value in net_returns),
        "folds_beating_buy_and_hold": beats_buy_hold,
        "folds_beating_no_trade": beats_no_trade,
        "mean_max_drawdown": sum(drawdowns) / len(drawdowns),
        "worst_fold_max_drawdown": min(drawdowns),
        "mean_sharpe_annualized": sum(sharpes) / len(sharpes),
        "mean_balanced_accuracy": sum(accuracies) / len(accuracies),
        "two_x_cost_mean_net_return": stressed_mean,
        "two_x_cost_positive_return_folds": stressed["positive_return_folds"],
        "evidence_status": evidence_status,
        "limitations": [
            "Outer test folds may overlap and are not independent statistical trials.",
            "Positive historical net returns do not establish future profitability.",
            "Cost stress multiplies modeled fee and slippage assumptions; it does not model market impact beyond the supplied cost model.",
            "This report is not a portfolio backtest and does not authorize model promotion or live execution.",
        ],
    }


def build_research_quality_report(result: dict[str, Any], *,
                                  minimum_folds: int = 5) -> dict[str, Any]:
    """Build an instrument-level evidence summary from an ETI research artifact."""
    if not isinstance(result, dict) or result.get("schema_version") != "eti.multi_market_research_result.v1":
        raise ValueError("unsupported multi-market research result schema")
    if result.get("execution_enabled") is not False:
        raise ValueError("research result must explicitly keep execution disabled")
    if result.get("training_performed") is not True:
        raise ValueError("source artifact must explicitly confirm training_performed=true")
    markets = result.get("markets")
    if not isinstance(markets, dict) or not markets:
        raise ValueError("research result must contain market results")
    declared_count = result.get("market_count")
    if isinstance(declared_count, bool) or not isinstance(declared_count, int) or declared_count != len(markets):
        raise ValueError("source market_count must match the markets mapping")
    planned_fits = result.get("planned_model_fit_calls")
    fit_cap = result.get("model_fit_cap")
    if (isinstance(planned_fits, bool) or not isinstance(planned_fits, int) or planned_fits < 1
            or isinstance(fit_cap, bool) or not isinstance(fit_cap, int)
            or fit_cap < planned_fits or fit_cap > 32):
        raise ValueError("source model-fit counts are missing or exceed the governed cap")
    summaries = {}
    for key, market in sorted(markets.items()):
        if isinstance(market, dict) and market.get("status") == "evaluation_failed":
            error = market.get("error")
            summaries[key] = {
                "market_key": key,
                "evidence_status": "evaluation_failed",
                "error_type": error.get("type") if isinstance(error, dict) else None,
                "model_promotion_authority": False,
                "execution_authority": False,
            }
        else:
            summaries[key] = summarize_market_research(
                key, market, minimum_folds=minimum_folds
            )
    return {
        "schema_version": "eti.research_quality_report.v1",
        "source_schema_version": result["schema_version"],
        "market_count": len(summaries),
        "markets": summaries,
        "model_promotion_authority": False,
        "budget_authority": False,
        "execution_authority": False,
        "warning": "Evidence summary only. Human review and independent validation remain mandatory; no profitability, promotion, or execution claim is made.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ETI nested out-of-sample research evidence")
    parser.add_argument("input", type=Path, help="eti.multi_market_research_result.v1 JSON")
    parser.add_argument("--output", type=Path, default=Path("artifacts/research-quality-report.json"))
    parser.add_argument("--minimum-folds", type=int, default=5)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    report = build_research_quality_report(source, minimum_folds=args.minimum_folds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "market_count": report["market_count"],
        "execution_enabled": False,
        "model_promotion_authority": False,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
