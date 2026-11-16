import pandas as pd
import pytest

from eti.market_data_hub import MarketDataSnapshot
from eti.research_runner import run_approved_multi_market_research


def cost_model():
    return {
        "schema_version": "eti.transaction_cost_model.v1",
        "instrument_identity": "kraken:spot:BTC/USD",
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


def approval(**overrides):
    value = {
        "schema_version": "eti.research_approval.v1",
        "approved": True,
        "authorize_research_fits": True,
        "execution_enabled": False,
        "max_model_fit_calls": 32,
        "usd_budget": {
            "estimated_run_cost_usd": 2.0,
            "daily_spend_to_date_usd": 1.0,
            "monthly_spend_to_date_usd": 10.0,
        },
        "evaluation": {
            "horizon": 6, "min_train": 400, "validation_size": 100,
            "test_size": 100, "step_size": 200,
        },
        "instruments": {"kraken:spot:BTC/USD": {
            "fee_bps": 6.0, "slippage_bps": 1.0,
            "spread_bps": 2.0, "market_impact_bps": 1.0,
            "funding_bps_per_bar": 0.0, "borrow_bps_per_bar": 0.0,
            "allow_short": False,
        }},
    }
    value.update(overrides)
    return value


def allocation(eligible=True):
    return {
        "plan_only": True, "training_authorized": False,
        "candidates": [{
            "instrument": {
                "venue": "kraken", "symbol": "BTC/USD",
                "asset_class": "crypto", "market_type": "spot", "contract_multiplier": 1.0, "active": True,
                "supports_short": False, "supports_funding": False,
            },
            "economic_evaluation_eligible": eligible,
            "provenance": {"requested_rows": 1000, "timeframe": "1h"},
            "quality": {"rows": 1000, "ohlcv_validated": True, "timestamp_order": "ascending", "last_candle_age_seconds": 100, "missing_interval_estimate": 0, "max_gap_seconds": 3600},
            "transaction_cost_model_status": "validated" if eligible else "unknown",
            "transaction_cost_model": cost_model() if eligible else None,
            "training_authorized": False,
            "eligibility_reasons": [] if eligible else ["transaction_cost_model_not_validated"],
        }],
    }


def scan():
    return {
        "scan_only": True, "execution_enabled": False,
        "candidates": [{
            "status": "data_ready",
            "provenance": {"requested_rows": 1000, "timeframe": "1h"},
            "quality": {"rows": 1000, "ohlcv_validated": True, "timestamp_order": "ascending", "last_candle_age_seconds": 100, "missing_interval_estimate": 0, "max_gap_seconds": 3600},
            "instrument": {"venue": "kraken", "symbol": "BTC/USD", "market_type": "spot", "contract_multiplier": 1.0, "active": True, "supports_short": False, "supports_funding": False, "asset_class": "crypto"},
        }],
        "config": {
            "timeframe": "1h", "requested_rows": 1000,
            "minimum_valid_rows": 662,
        },
    }


class FakeHub:
    def fetch(self, instrument, timeframe, limit, min_rows):
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=1000, freq="h", tz="UTC"),
            "open": [100.0 + i * 0.01 for i in range(1000)],
            "high": [101.0 + i * 0.01 for i in range(1000)],
            "low": [99.0 + i * 0.01 for i in range(1000)],
            "close": [100.5 + i * 0.01 for i in range(1000)],
            "volume": [10.0] * 1000,
        })
        return MarketDataSnapshot(
            instrument, timeframe, frame,
            {"venue": instrument.venue, "symbol": instrument.symbol},
            {"rows": len(frame), "ohlcv_validated": True},
        )


def test_runner_requires_explicit_approval_and_keeps_execution_disabled():
    unapproved = approval(approved=False)
    with pytest.raises(ValueError, match="must both be explicit"):
        run_approved_multi_market_research(scan(), allocation(), unapproved, hub=FakeHub())


def test_runner_rejects_candidates_without_validated_economic_eligibility():
    with pytest.raises(ValueError, match="not eligible"):
        run_approved_multi_market_research(
            scan(), allocation(eligible=False), approval(), hub=FakeHub()
        )


@pytest.mark.parametrize("budget", [
    {"estimated_run_cost_usd": 11, "daily_spend_to_date_usd": 0, "monthly_spend_to_date_usd": 0},
    {"estimated_run_cost_usd": 2, "daily_spend_to_date_usd": 9, "monthly_spend_to_date_usd": 0},
    {"estimated_run_cost_usd": 2, "daily_spend_to_date_usd": 0, "monthly_spend_to_date_usd": 99},
])
def test_runner_fails_closed_on_usd_ceilings(budget):
    with pytest.raises(ValueError, match="ceiling exceeded"):
        run_approved_multi_market_research(
            scan(), allocation(), approval(usd_budget=budget), hub=FakeHub()
        )


def test_runner_preflights_global_fit_cap_before_any_fit(monkeypatch):
    import eti.research_runner as runner

    monkeypatch.setattr(runner, "estimate_nested_model_fits_from_rows", lambda *a, **k: 40)
    def should_not_fit(*args, **kwargs):
        raise AssertionError("model fitting must not start when global cap is exceeded")
    monkeypatch.setattr(runner, "run_nested_experiment", should_not_fit)

    with pytest.raises(ValueError, match="stopped before any model fitting"):
        run_approved_multi_market_research(
            scan(), allocation(), approval(max_model_fit_calls=32), hub=FakeHub()
        )


def test_runner_rejects_execution_enabled_approval():
    with pytest.raises(ValueError, match="execution_enabled=false"):
        run_approved_multi_market_research(
            scan(), allocation(), approval(execution_enabled=True), hub=FakeHub()
        )


def test_runner_rejects_zero_contract_multiplier_instead_of_defaulting():
    from eti.research_runner import _instrument
    row = allocation()["candidates"][0]
    row["instrument"]["contract_multiplier"] = 0
    with pytest.raises(ValueError, match="contract_multiplier"):
        _instrument(row)


def test_runner_rejects_inactive_instrument():
    inactive_scan = scan()
    inactive_allocation = allocation()
    inactive_scan["candidates"][0]["instrument"]["active"] = False
    inactive_allocation["candidates"][0]["instrument"]["active"] = False
    with pytest.raises(ValueError, match="inactive"):
        run_approved_multi_market_research(
            inactive_scan, inactive_allocation, approval(), hub=FakeHub()
        )


def test_runner_rejects_modified_scan_quality_in_allocation():
    altered = allocation()
    altered["candidates"][0]["quality"]["last_candle_age_seconds"] = 0
    with pytest.raises(ValueError, match="provenance/quality differs"):
        run_approved_multi_market_research(scan(), altered, approval(), hub=FakeHub())


def test_signed_funding_number_is_accepted_without_relaxing_other_costs():
    from eti.research_runner import _number
    assert _number("funding", -3.25, allow_negative=True) == -3.25
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        _number("fee", -3.25)


def test_derivative_economic_evaluation_is_blocked_before_market_data_fetch():
    candidate_scan = scan()
    candidate_allocation = allocation()
    for report in (candidate_scan, candidate_allocation):
        instrument = report["candidates"][0]["instrument"]
        instrument["market_type"] = "swap"
        instrument["supports_funding"] = True
    candidate_allocation["candidates"][0]["transaction_cost_model"]["instrument_identity"] = "kraken:swap:BTC/USD"
    approved = approval()
    approved["instruments"]["kraken:swap:BTC/USD"] = approved["instruments"].pop("kraken:spot:BTC/USD")
    class NoFetchHub:
        def fetch(self, *args, **kwargs):
            raise AssertionError("derivative data must not be fetched for unsupported economics")

    with pytest.raises(ValueError, match="derivative economic evaluation is blocked"):
        run_approved_multi_market_research(
            candidate_scan, candidate_allocation, approved, hub=NoFetchHub()
        )



def test_funding_event_alignment_preflight_rejects_off_bar_settlement():
    from eti.research_runner import _validate_funding_event_alignment

    bars = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")
    with pytest.raises(ValueError, match="does not align.*before model fitting"):
        _validate_funding_event_alignment(
            [{"timestamp": "2026-01-01T02:30:00Z", "rate_bps": 2.0}],
            pd.Series(bars), "kraken:swap:BTC/USD",
        )


def test_funding_event_alignment_preflight_accepts_exact_utc_bar():
    from eti.research_runner import _validate_funding_event_alignment

    bars = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")
    _validate_funding_event_alignment(
        [{"timestamp": "2026-01-01T02:00:00+00:00", "rate_bps": 2.0}],
        pd.Series(bars), "kraken:swap:BTC/USD",
    )



def test_spot_short_economic_evaluation_is_blocked_until_borrow_costs_supported():
    approved = approval()
    approved["instruments"]["kraken:spot:BTC/USD"]["allow_short"] = True
    candidate_scan = scan()
    candidate_allocation = allocation()
    candidate_scan["candidates"][0]["instrument"]["supports_short"] = True
    candidate_allocation["candidates"][0]["instrument"]["supports_short"] = True
    with pytest.raises(ValueError, match="short research is blocked"):
        run_approved_multi_market_research(
            candidate_scan, candidate_allocation, approved, hub=FakeHub()
        )
