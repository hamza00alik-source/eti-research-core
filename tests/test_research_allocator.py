import pytest

from eti.research_allocator import allocate_research_budget


def candidate(symbol="BTC/USD", venue="kraken", age=300, missing=0, gap=3600,
              rows=1000, requested=1000, asset_class="crypto"):
    return {
        "status": "data_ready",
        "instrument": {"venue": venue, "symbol": symbol, "asset_class": asset_class},
        "provenance": {"requested_rows": requested, "timeframe": "1h"},
        "quality": {
            "rows": rows, "last_candle_age_seconds": age,
            "missing_interval_estimate": missing, "max_gap_seconds": gap,
            "ohlcv_validated": True, "timestamp_order": "ascending",
        },
    }


def scan(*items):
    return {"scan_only": True, "execution_enabled": False, "candidates": list(items)}


def test_allocator_ranks_data_readiness_without_claiming_profitability():
    result = allocate_research_budget(
        scan(candidate("A/USD", age=60), candidate("B/USD", age=900)),
        max_model_fit_calls=4, fits_per_candidate=2,
    )
    assert result["candidates"][0]["instrument"]["symbol"] == "A/USD"
    assert result["summary"]["selected_count"] == 2
    assert result["training_performed"] is False
    assert result["execution_enabled"] is False
    assert result["usd_budget"]["authorized"] is False
    assert result["candidates"][0]["economic_evaluation_eligible"] is False


def test_allocator_requires_validated_cost_model_for_economic_eligibility():
    item = candidate()
    result = allocate_research_budget(
        scan(item), cost_model_status={"kraken:BTC/USD": "validated"},
        estimated_run_cost_usd=2, daily_spend_to_date_usd=1,
        monthly_spend_to_date_usd=10,
    )
    row = result["candidates"][0]
    assert row["economic_evaluation_eligible"] is True
    assert result["usd_budget"]["authorized"] is True
    assert row["training_authorized"] is False


def test_allocator_fails_closed_on_daily_or_monthly_spend_caps():
    for daily, monthly in ((9, 99), (1, 99)):
        result = allocate_research_budget(
            scan(candidate()), estimated_run_cost_usd=2,
            daily_spend_to_date_usd=daily, monthly_spend_to_date_usd=monthly,
        )
        assert result["usd_budget"]["authorized"] is False
        assert result["training_authorized"] is False


def test_allocator_blocks_stale_or_gappy_candidates():
    result = allocate_research_budget(scan(
        candidate("STALE/USD", age=9000),
        candidate("GAPPY/USD", age=10, gap=20000),
    ))
    assert result["summary"]["selected_count"] == 0
    assert all(row["data_readiness_score"] == 0 for row in result["candidates"])


def test_allocator_never_exceeds_fit_budget_or_candidate_cap():
    items = [candidate(f"COIN{i}/USD", venue="kraken", age=i) for i in range(10)]
    result = allocate_research_budget(
        scan(*items), max_model_fit_calls=5, fits_per_candidate=2,
        max_candidates=10,
    )
    assert result["compute_budget"]["allocated_model_fit_calls"] <= 5
    assert result["summary"]["selected_count"] == 2


def test_allocator_rejects_non_scan_reports_and_invalid_limits():
    with pytest.raises(ValueError):
        allocate_research_budget({"candidates": []})
    with pytest.raises(ValueError):
        allocate_research_budget(scan(candidate()), max_model_fit_calls=0)



def test_session_based_assets_are_not_misclassified_without_calendar():
    equity = candidate("SPY", venue="yahoo", age=300, gap=86400, asset_class="equity")
    result = allocate_research_budget(scan(equity))
    row = result["candidates"][0]
    assert row["data_readiness_score"] > 0
    assert "session_calendar_not_applied" in row["eligibility_reasons"]
    assert row["economic_evaluation_eligible"] is False
