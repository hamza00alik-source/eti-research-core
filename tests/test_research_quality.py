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


def result_for(market):
    return {
        "schema_version": "eti.multi_market_research_result.v1",
        "execution_enabled": False,
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
