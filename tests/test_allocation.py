import pytest
from eti.allocation import plan_paper_allocations
from eti.agents import default_capabilities


def research_report(*, mean_return=0.04, two_x_return=0.02,
                    positive_folds=5, two_x_positive_folds=5, beating_no_trade=5):
    return {
        "training_performed": True, "execution_enabled": False,
        "fold_count": 5, "folds": [{}, {}, {}, {}, {}],
        "trading_performance": {"summary": {
            "mean_net_return": mean_return,
            "folds_beating_no_trade": beating_no_trade,
        }},
        "cost_sensitivity": {"scenarios": [
            {"cost_multiplier": 1.0, "fold_count": 5,
             "mean_net_return": mean_return, "positive_return_folds": positive_folds},
            {"cost_multiplier": 2.0, "fold_count": 5,
             "mean_net_return": two_x_return, "positive_return_folds": two_x_positive_folds},
        ]},
    }


def candidate(symbol="BTC/USD", group="crypto_major", **overrides):
    item = {
        "symbol": symbol, "strategy": "momentum", "correlation_group": group,
        "mean_max_drawdown": 0.08, "stop_distance_fraction": 0.04,
        "research_report": research_report(),
    }
    item.update(overrides)
    return item


def test_allocator_proposes_bounded_paper_allocations_only():
    result = plan_paper_allocations({
        "equity": 10000,
        "candidates": [candidate(), candidate("ETH/USD", "crypto_major"),
                       candidate("EURUSD", "fx_eur")],
        "cash_buffer_fraction": 0.20, "max_gross_exposure_fraction": 0.50,
        "max_single_asset_fraction": 0.10, "max_correlated_group_fraction": 0.15,
        "max_risk_per_trade": 0.005, "max_positions": 3,
    })
    assert result["status"] == "proposals_ready"
    assert result["execution_enabled"] is False
    assert result["paper_only"] is True
    assert result["owner_review_required"] is True
    assert result["orders_submitted"] is False
    assert result["proposed_gross_exposure_fraction"] <= 0.50
    assert all(item["equity_fraction"] <= 0.10 for item in result["allocations"])
    assert sum(item["equity_fraction"] for item in result["allocations"]
               if item["correlation_group"] == "crypto_major") <= 0.15


def test_allocator_rejects_no_go_and_missing_evidence():
    no_go = candidate("BTC/USD", research_report=research_report(
        mean_return=-0.01, two_x_return=-0.03,
        positive_folds=0, two_x_positive_folds=0, beating_no_trade=0,
    ))
    missing = candidate("ETH/USD", research_report=None)
    result = plan_paper_allocations({"equity": 10000, "candidates": [no_go, missing]})
    assert result["allocations"] == []
    assert {item["reason"] for item in result["rejected"]} == {
        "economic_evidence_not_eligible_for_paper_review", "missing_research_report"
    }


def test_allocator_caps_position_by_stop_risk():
    result = plan_paper_allocations({
        "equity": 10000, "candidates": [candidate()],
        "cash_buffer_fraction": 0.40, "max_gross_exposure_fraction": 0.80,
        "max_single_asset_fraction": 0.50, "max_correlated_group_fraction": 0.80,
        "max_risk_per_trade": 0.005,
    })
    assert result["allocations"][0]["proposed_notional"] <= 1250.0


def test_allocator_respects_existing_exposure_and_position_count():
    result = plan_paper_allocations({
        "equity": 10000, "current_positions": {"SOL/USD": 1000},
        "correlated_group_exposure": {"crypto_major": 1000},
        "candidates": [candidate()], "max_positions": 1,
    })
    assert result["allocations"] == []


def test_allocator_rejects_invalid_limits_and_duplicate_candidates():
    with pytest.raises(ValueError, match="cash_buffer_fraction"):
        plan_paper_allocations({"equity": 10000, "cash_buffer_fraction": 1.0,
                                "candidates": []})
    result = plan_paper_allocations({"equity": 10000, "candidates": [candidate(), candidate()]})
    assert len(result["allocations"]) == 1
    assert {"symbol": "BTC/USD", "reason": "duplicate_candidate_symbol"} in result["rejected"]


def test_allocator_is_registered_as_strategy_department_capability():
    registry = default_capabilities()
    assert "portfolio.allocate" in registry.capabilities()
    result = registry.execute("portfolio.allocate", {"equity": 10000, "candidates": []})
    assert result["execution_enabled"] is False


def test_allocator_rejects_forged_or_incomplete_fold_counts():
    weak = candidate("XAUUSD", research_report=research_report(
        two_x_positive_folds=4,
    ))
    result = plan_paper_allocations({"equity": 10000, "candidates": [weak]})
    assert result["allocations"] == []
    assert result["rejected"][0]["reason"] == "economic_evidence_not_eligible_for_paper_review"


def test_allocator_ignores_forged_cached_economic_status():
    forged = candidate("BTC/USD")
    forged["economic_assessment"] = {
        "status": "human_review_required", "live_execution_eligible": False,
        "fold_count": 5, "mean_net_return": 0.9, "two_x_mean_net_return": 0.8,
        "base_cost_positive_return_folds": 5, "two_x_positive_return_folds": 5,
        "folds_beating_no_trade": 5,
    }
    forged["research_report"] = research_report(mean_return=-0.02, two_x_return=-0.04,
                                                positive_folds=0, two_x_positive_folds=0,
                                                beating_no_trade=0)
    result = plan_paper_allocations({"equity": 10000, "candidates": [forged]})
    assert result["allocations"] == []
    assert result["rejected"][0]["reason"] == "economic_evidence_not_eligible_for_paper_review"
