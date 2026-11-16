from datetime import datetime, timezone

import numpy as np
import pytest

from eti.agents import default_capabilities
from eti.specialists import (
    execution_cost_specialist,
    learning_economic_assessment_specialist,
    macro_as_of_specialist,
    market_regime_specialist,
    risk_assessment_specialist,
    strategy_selection_specialist,
)


def cost_model():
    return {
        "schema_version": "eti.transaction_cost_model.v1",
        "instrument_identity": "kraken:spot:BTC/USD",
        "quote_currency": "USD", "settlement_currency": "USD", "fee_basis": "taker",
        "fee_bps": 6, "spread_bps": 2, "slippage_bps": 1,
        "market_impact_bps": 1, "funding_bps_per_bar": 0.5,
        "borrow_bps_per_bar": 2, "contract_multiplier": 1,
        "retrieved_at": "2026-01-01T00:00:00Z",
        "valid_from": "2026-01-01T00:00:00Z",
        "valid_until": "2030-01-01T00:00:00Z",
        "evidence": [{
            "source": "Venue fee schedule", "uri": "https://example.com/fees",
            "license_status": "open", "observed_at": "2026-01-01T00:00:00Z",
            "covers": ["fee_bps", "spread_bps", "slippage_bps", "market_impact_bps",
                       "funding_bps_per_bar", "borrow_bps_per_bar",
                       "contract_multiplier", "fee_basis"],
        }],
    }


def economic_report():
    return {
        "training_performed": True, "execution_enabled": False,
        "fold_count": 2, "folds": [{}, {}],
        "trading_performance": {"summary": {
            "mean_net_return": 0.02, "folds_beating_no_trade": 2,
        }},
        "cost_sensitivity": {"scenarios": [
            {"cost_multiplier": 1.0, "fold_count": 2, "mean_net_return": 0.02,
             "positive_return_folds": 2},
            {"cost_multiplier": 2.0, "fold_count": 2, "mean_net_return": 0.01,
             "positive_return_folds": 2},
        ]},
    }


def test_market_regime_specialist_returns_causal_descriptive_state():
    close = (100 * np.exp(np.cumsum(np.sin(np.arange(180) / 9) * 0.001))).tolist()
    result = market_regime_specialist({"close": close})
    assert result["kind"] == "market_regime_specialist"
    assert result["regime"] in {"unknown", "high_volatility", "uptrend", "downtrend", "range"}
    assert result["observations"] == 180
    assert result["execution_enabled"] is False


def test_market_regime_rejects_short_or_invalid_history():
    with pytest.raises(ValueError, match="at least 122"):
        market_regime_specialist({"close": [100.0] * 100})
    with pytest.raises(ValueError, match="positive"):
        market_regime_specialist({"close": [100.0] * 121 + [0.0]})


def test_strategy_specialist_selects_only_from_validation_slice():
    close = (100 * np.exp(np.cumsum(np.sin(np.arange(180) / 7) * 0.003))).tolist()
    result = strategy_selection_specialist({"close": close})
    assert result["kind"] == "strategy_selection_specialist"
    assert result["candidate"] in {"ma", "momentum", "breakout", "mean_reversion"}
    assert result["test_evidence"] == "not_assessed"
    assert result["execution_enabled"] is False


def test_risk_specialist_applies_deterministic_rejection():
    result = risk_assessment_specialist({"request": {
        "equity": 10000, "entry_price": 100, "stop_price": 99,
        "requested_notional": 1000, "daily_pnl_fraction": -1.0,
    }})
    assert result["approved"] is False
    assert "daily_loss_limit" in result["reasons"]
    assert result["model_override_allowed"] is False
    assert result["execution_enabled"] is False


def test_macro_specialist_excludes_future_releases_and_revisions():
    result = macro_as_of_specialist({
        "series_id": "CPI", "provider": "official-fixture",
        "decision_time": "2026-01-15T00:00:00Z",
        "observations": [
            {"series_id": "CPI", "period": "2025-12-01T00:00:00Z",
             "value": 2.8, "unit": "percent",
             "released_at": "2026-01-10T00:00:00Z",
             "vintage_at": "2026-01-10T00:00:00Z",
             "source": "official", "license_status": "open", "provenance": "release-1"},
            {"series_id": "CPI", "period": "2025-12-01T00:00:00Z",
             "value": 2.7, "unit": "percent",
             "released_at": "2026-01-10T00:00:00Z",
             "vintage_at": "2026-01-20T00:00:00Z",
             "source": "official", "license_status": "open", "provenance": "revision-1"},
        ],
    })
    assert len(result["observations"]) == 1
    assert result["observations"][0]["value"] == 2.8
    assert result["lookahead_prevented"] is True


def test_macro_specialist_rejects_restricted_license():
    payload = {
        "series_id": "RATE", "decision_time": "2026-01-15T00:00:00Z",
        "observations": [{
            "series_id": "RATE", "period": "2025-12-01T00:00:00Z",
            "value": 4.0, "unit": "percent",
            "released_at": "2026-01-10T00:00:00Z",
            "vintage_at": "2026-01-10T00:00:00Z",
            "source": "unknown", "license_status": "restricted", "provenance": "test",
        }],
    }
    with pytest.raises(PermissionError):
        macro_as_of_specialist(payload)


def test_execution_cost_specialist_estimates_but_never_orders():
    result = execution_cost_specialist({
        "instrument_identity": "kraken:spot:BTC/USD", "cost_model": cost_model(),
        "side": "long", "notional": 10000, "holding_bars": 2,
    })
    assert result["round_trip_trading_cost_bps"] == 18
    assert result["funding_cost_bps"] == 1
    assert result["estimated_total_cost_bps"] == 19
    assert result["estimated_total_cost_quote"] == 19
    assert result["orders_submitted"] is False
    assert result["execution_enabled"] is False


def test_execution_cost_specialist_rejects_wrong_instrument_model():
    model = cost_model()
    model["instrument_identity"] = "kraken:swap:BTC/USD"
    with pytest.raises(ValueError, match="identity mismatch"):
        execution_cost_specialist({
            "instrument_identity": "kraken:spot:BTC/USD", "cost_model": model,
            "notional": 10000,
        })


def test_learning_specialist_returns_human_gate_without_authority():
    result = learning_economic_assessment_specialist({"report": economic_report()})
    assert result["status"] == "human_review_required"
    assert result["live_execution_eligible"] is False
    assert result["model_promotion_authority"] is False


def test_default_registry_registers_department_specialists_and_routes():
    registry = default_capabilities()
    expected = {
        "market.regime", "strategy.select", "risk.assess", "macro.as_of",
        "execution.cost_estimate", "learning.economic_assess",
    }
    assert expected.issubset(set(registry.capabilities()))
    assert registry.execute("learning.economic_assess", {"report": economic_report()})["status"] == "human_review_required"


def test_macro_worker_result_is_json_serializable():
    from eti.storage import _encode_json

    payload = {
        "series_id": "CPI", "provider": "official-fixture",
        "decision_time": "2026-01-15T00:00:00Z",
        "observations": [{
            "series_id": "CPI", "period": "2025-12-01T00:00:00Z",
            "value": 2.8, "unit": "percent",
            "released_at": "2026-01-10T00:00:00Z",
            "vintage_at": "2026-01-10T00:00:00Z",
            "source": "official", "license_status": "open", "provenance": "release-1",
        }],
    }
    result = default_capabilities().execute("macro.as_of", payload)
    assert _encode_json(result)
    assert result["observations"][0]["available_at"] == "2026-01-10T00:00:00+00:00"


def test_strategy_cost_inputs_are_validated_and_applied():
    with pytest.raises(ValueError, match="cannot be negative"):
        strategy_selection_specialist({
            "close": (100 * np.exp(np.cumsum(np.sin(np.arange(180) / 7) * 0.003))).tolist(),
            "fee_bps": -1,
        })
