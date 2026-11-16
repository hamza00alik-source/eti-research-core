import pytest

from eti.research_allocator import allocate_research_budget


def candidate(symbol="BTC/USD", venue="kraken", age=300, missing=0, gap=3600,
              rows=1000, requested=1000, asset_class="crypto"):
    return {
        "status": "data_ready",
        "instrument": {"venue": venue, "symbol": symbol, "asset_class": asset_class, "market_type": "spot"},
        "provenance": {"requested_rows": requested, "timeframe": "1h"},
        "quality": {
            "rows": rows, "last_candle_age_seconds": age,
            "missing_interval_estimate": missing, "max_gap_seconds": gap,
            "ohlcv_validated": True, "timestamp_order": "ascending",
        },
    }


def cost_model(identity="kraken:spot:BTC/USD"):
    return {
        "schema_version": "eti.transaction_cost_model.v1",
        "instrument_identity": identity,
        "quote_currency": "USD", "settlement_currency": "USD", "fee_basis": "taker",
        "fee_bps": 6, "spread_bps": 2, "slippage_bps": 1,
        "market_impact_bps": 1, "funding_bps_per_bar": 0,
        "borrow_bps_per_bar": 0, "contract_multiplier": 1,
        "retrieved_at": "2026-01-01T00:00:00Z",
        "valid_from": "2026-01-01T00:00:00Z",
        "valid_until": "2030-01-01T00:00:00Z",
        "evidence": [{
            "source": "Published venue fee schedule",
            "uri": "https://example.com/fees",
            "license_status": "open",
            "observed_at": "2026-01-01T00:00:00Z",

            "covers": ["fee_bps", "spread_bps", "slippage_bps", "market_impact_bps", "funding_bps_per_bar", "borrow_bps_per_bar", "contract_multiplier", "fee_basis"],
        }],
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
        scan(item), cost_models={"kraken:spot:BTC/USD": cost_model()},
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


def test_allocator_cost_models_are_scoped_by_market_type():
    spot = candidate("BTC/USD", venue="kraken")
    swap = candidate("BTC/USD", venue="kraken")
    swap["instrument"]["market_type"] = "swap"
    result = allocate_research_budget(
        scan(spot, swap),
        cost_models={"kraken:spot:BTC/USD": cost_model()},
        estimated_run_cost_usd=1, daily_spend_to_date_usd=0,
        monthly_spend_to_date_usd=0,
    )
    rows = {row["instrument"]["market_type"]: row for row in result["candidates"]}
    assert rows["spot"]["economic_evaluation_eligible"] is True
    assert rows["swap"]["economic_evaluation_eligible"] is False


def test_selected_candidate_identity_includes_market_type():
    spot = candidate("BTC/USD", venue="kraken")
    swap = candidate("BTC/USD", venue="kraken")
    swap["instrument"]["market_type"] = "swap"
    result = allocate_research_budget(
        scan(spot, swap), max_model_fit_calls=2, fits_per_candidate=2,
        max_candidates=2, cost_models={},
    )
    rows = {row["instrument"]["market_type"]: row for row in result["candidates"]}
    assert rows["spot"]["research_priority_rank"] == 1
    assert rows["swap"]["research_priority_rank"] is None
    assert rows["swap"]["training_block_reason"] == "not_selected_within_research_budget"



def test_allocator_marks_derivatives_ineligible_even_with_valid_cost_model():
    item = candidate()
    item["instrument"]["market_type"] = "swap"
    identity = "kraken:swap:BTC/USD"
    result = allocate_research_budget(
        scan(item), cost_models={identity: cost_model(identity)},
        estimated_run_cost_usd=2, daily_spend_to_date_usd=1,
        monthly_spend_to_date_usd=10,
    )
    row = result["candidates"][0]
    assert row["data_readiness_score"] > 0
    assert row["transaction_cost_model_status"] == "validated"
    assert row["economic_evaluation_eligible"] is False
    assert "contract_level_derivative_economics_not_integrated" in row["eligibility_reasons"]
