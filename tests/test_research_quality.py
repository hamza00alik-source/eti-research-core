import pytest

from eti.research_quality import build_research_quality_report, summarize_market_research


def market_result(returns=(0.03, 0.02, -0.01, 0.04, 0.01), stressed=0.004):
    folds = []
    for value in returns:
        folds.append({
            "trading": {
                "model": {"total_return": value, "max_drawdown": -0.12, "sharpe_annualized": 0.8},
                "buy_and_hold": {"total_return": 0.01},
                "no_trade": {"total_return": 0.0},
            },
            "classification": {"balanced_accuracy": 0.55},
        })
    count = len(folds)
    return {
        "status": "evaluated",
        "report": {
            "training_performed": True,
            "trading_performance": {"folds": folds},
            "cost_sensitivity": {"scenarios": [
                {"cost_multiplier": 1.0, "mean_net_return": sum(returns) / count,
                 "positive_return_folds": sum(value > 0 for value in returns), "fold_count": count},
                {"cost_multiplier": 2.0, "mean_net_return": stressed,
                 "positive_return_folds": 3, "fold_count": count},
            ]},
        },
    }



def market_with_calibration_evidence():
    market = market_result()
    report = market["report"]
    count = len(report["trading_performance"]["folds"])
    report["folds"] = []
    for index in range(count):
        report["folds"].append({
            "test_actual": [0, 1, 2],
            "probability_calibration": {
                "status": "calibrated_and_outer_test_confirmed",
                "outer_test_confirmation": "calibration_confirmed_on_outer_test",
                "outer_test_class_coverage_passed": True,
                "outer_test_nll_not_worse": True,
                "outer_test_brier_not_worse": True,
                "calibration_fit": {
                    "status": "fit", "passed": True, "method": "temperature_scaling"
                },
                "raw": {
                    "sample_count": 3, "class_counts": {"0": 1, "1": 1, "2": 1},
                    "negative_log_likelihood": 1.2, "brier_score": 0.6
                },
                "temperature_scaled": {
                    "sample_count": 3, "class_counts": {"0": 1, "1": 1, "2": 1},
                    "negative_log_likelihood": 1.1, "brier_score": 0.55
                },
            }
        })
    report["probability_calibration"] = {
        "method": "fold_local_temperature_scaling",
        "fold_count": count,
        "calibration_attempted_folds": count,
        "outer_test_assessed_folds": count,
        "outer_test_confirmed_folds": count,
        "mean_raw_outer_test_nll": 1.2,
        "mean_calibrated_outer_test_nll": 1.1,
        "mean_raw_outer_test_brier": 0.6,
        "mean_calibrated_outer_test_brier": 0.55,
    }
    return market

def result_for(market):
    return {
        "schema_version": "eti.multi_market_research_result.v1",
        "execution_enabled": False,
        "training_performed": True,
        "planned_model_fit_calls": 10,
        "model_fit_cap": 32,
        "market_count": 1,
        "markets": {"coinbase:spot:BTC/USD": market},
    }


def test_reports_outer_test_and_two_x_cost_evidence_without_promotion_authority():
    report = build_research_quality_report(result_for(market_result()))
    metric = report["markets"]["coinbase:spot:BTC/USD"]
    assert metric["outer_test_fold_count"] == 5
    assert metric["mean_net_return"] == pytest.approx(0.018)
    assert metric["folds_beating_buy_and_hold"] == 3
    assert metric["evidence_status"] == "human_review_required"
    assert report["model_promotion_authority"] is False
    assert report["execution_authority"] is False


def test_negative_two_x_cost_result_is_not_robust():
    metric = summarize_market_research(
        "venue:spot:ABC/USD", market_result(stressed=-0.01)
    )
    assert metric["evidence_status"] == "not_robust_at_2x_costs"


def test_fewer_than_minimum_folds_remains_insufficient():
    metric = summarize_market_research(
        "venue:spot:ABC/USD", market_result(returns=(0.1, 0.1, 0.1)),
        minimum_folds=5,
    )
    assert metric["evidence_status"] == "insufficient_folds"


def test_failed_market_is_reported_without_fabricating_metrics():
    report = build_research_quality_report(result_for({
        "status": "evaluation_failed", "error": {"type": "ValueError"}
    }))
    assert report["markets"]["coinbase:spot:BTC/USD"]["evidence_status"] == "evaluation_failed"
    assert "mean_net_return" not in report["markets"]["coinbase:spot:BTC/USD"]


@pytest.mark.parametrize("mutate", [
    lambda m: m["report"]["trading_performance"]["folds"][0]["trading"]["model"].update(total_return=float("nan")),
    lambda m: m["report"]["trading_performance"]["folds"][0]["classification"].update(balanced_accuracy=1.5),
    lambda m: m["report"]["cost_sensitivity"]["scenarios"].pop(),
])
def test_invalid_or_incomplete_evidence_fails_closed(mutate):
    market = market_result()
    mutate(market)
    with pytest.raises(ValueError):
        summarize_market_research("venue:spot:ABC/USD", market)


def test_execution_must_remain_disabled():
    source = result_for(market_result())
    source["execution_enabled"] = True
    with pytest.raises(ValueError):
        build_research_quality_report(source)

def test_source_artifact_must_confirm_training_was_performed():
    source = result_for(market_result())
    source["training_performed"] = False
    with pytest.raises(ValueError, match="training_performed=true"):
        build_research_quality_report(source)


@pytest.mark.parametrize("field,value", [
    ("market_count", 2),
    ("planned_model_fit_calls", 33),
    ("model_fit_cap", 2),
])
def test_source_artifact_fit_and_market_counts_are_validated(field, value):
    source = result_for(market_result())
    source[field] = value
    with pytest.raises(ValueError):
        build_research_quality_report(source)


def test_quality_report_includes_verified_fold_local_calibration_evidence():
    report = build_research_quality_report(result_for(market_with_calibration_evidence()))
    calibration = report["markets"]["coinbase:spot:BTC/USD"]["probability_calibration"]
    assert calibration["status"] == "confirmed_all_folds"
    assert calibration["outer_test_assessed_folds"] == 5
    assert calibration["outer_test_confirmed_folds"] == 5
    assert calibration["mean_calibrated_outer_test_nll"] == pytest.approx(1.1)
    assert calibration["mean_calibrated_outer_test_brier"] == pytest.approx(0.55)


def test_quality_report_rejects_fabricated_calibration_aggregate_counts():
    market = market_with_calibration_evidence()
    market["report"]["probability_calibration"]["outer_test_confirmed_folds"] = 4
    with pytest.raises(ValueError, match="does not match fold evidence"):
        summarize_market_research("venue:spot:ABC/USD", market)


def test_legacy_quality_report_marks_probability_calibration_unavailable():
    metric = summarize_market_research("venue:spot:ABC/USD", market_result())
    assert metric["probability_calibration"]["status"] == "not_available"


def test_calibration_confirmation_requires_a_real_passed_temperature_fit():
    market = market_with_calibration_evidence()
    fold = market["report"]["folds"][0]["probability_calibration"]
    fold["calibration_fit"]["passed"] = False
    with pytest.raises(ValueError, match="without a passed temperature fit"):
        summarize_market_research("venue:spot:ABC/USD", market)


def test_calibration_metrics_must_match_outer_test_sample_count():
    market = market_with_calibration_evidence()
    fold = market["report"]["folds"][0]
    fold["probability_calibration"]["temperature_scaled"]["sample_count"] = 2
    with pytest.raises(ValueError, match="sample/class counts are invalid"):
        summarize_market_research("venue:spot:ABC/USD", market)


def test_calibration_confirmation_flags_must_be_consistent():
    market = market_with_calibration_evidence()
    fold = market["report"]["folds"][0]["probability_calibration"]
    fold["outer_test_brier_not_worse"] = False
    with pytest.raises(ValueError, match="confirmation flags are inconsistent"):
        summarize_market_research("venue:spot:ABC/USD", market)
