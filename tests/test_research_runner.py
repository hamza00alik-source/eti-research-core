import pandas as pd
import pytest

from eti.market_data_hub import MarketDataSnapshot
from eti.research_runner import run_approved_multi_market_research


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
        "instruments": {"kraken:BTC/USD": {
            "fee_bps": 6.0, "slippage_bps": 2.0,
            "funding_bps_per_bar": 0.0, "allow_short": False,
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
                "asset_class": "crypto", "market_type": "spot",
                "supports_short": False, "supports_funding": False,
            },
            "economic_evaluation_eligible": eligible,
            "transaction_cost_model_status": "validated" if eligible else "unknown",
            "training_authorized": False,
            "eligibility_reasons": [] if eligible else ["transaction_cost_model_not_validated"],
        }],
    }


def scan():
    return {
        "scan_only": True,
        "candidates": [{
            "status": "data_ready",
            "instrument": {"venue": "kraken", "symbol": "BTC/USD"},
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
