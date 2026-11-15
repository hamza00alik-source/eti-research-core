from datetime import datetime, timezone

import pytest

from eti.transaction_costs import validate_transaction_cost_model


def model():
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
            "source": "Venue fee schedule",
            "uri": "https://example.com/fees",
            "license_status": "open",
            "observed_at": "2026-01-01T00:00:00Z",

            "covers": ["fee_bps", "spread_bps", "slippage_bps", "market_impact_bps", "funding_bps_per_bar", "borrow_bps_per_bar", "contract_multiplier", "fee_basis"],
        }],
    }


def test_valid_evidence_model_is_normalized():
    result = validate_transaction_cost_model(
        model(), identity="kraken:spot:BTC/USD",
        now=datetime(2026, 11, 15, tzinfo=timezone.utc),
    )
    assert result["fee_bps"] == 6
    assert result["instrument_identity"] == "kraken:spot:BTC/USD"


@pytest.mark.parametrize("change", [
    {"instrument_identity": "kraken:swap:BTC/USD"},
    {"fee_bps": None},
    {"borrow_bps_per_bar": -1},
    {"contract_multiplier": 0},
    {"valid_until": "2026-02-01T00:00:00Z"},
    {"evidence": []},
    {"evidence": [{"source": "unknown", "uri": "http://example.com", "license_status": "unknown", "observed_at": "2026-01-01T00:00:00Z"}]},
    {"retrieved_at": "2027-01-01T00:00:00Z"},
])
def test_invalid_or_unsupported_evidence_fails_closed(change):
    value = model()
    value.update(change)
    with pytest.raises(ValueError):
        validate_transaction_cost_model(
            value, identity="kraken:spot:BTC/USD",
            now=datetime(2026, 11, 15, tzinfo=timezone.utc),
        )


def test_naive_timestamps_are_rejected():
    value = model()
    value["valid_from"] = "2026-01-01T00:00:00"
    with pytest.raises(ValueError, match="timezone"):
        validate_transaction_cost_model(
            value, identity="kraken:spot:BTC/USD",
            now=datetime(2026, 11, 15, tzinfo=timezone.utc),
        )



def test_evidence_must_explicitly_cover_every_numeric_assumption():
    value = model()
    value["evidence"][0]["covers"] = ["fee_bps", "spread_bps"]
    with pytest.raises(ValueError, match="does not cover"):
        validate_transaction_cost_model(
            value, identity="kraken:spot:BTC/USD",
            now=datetime(2026, 11, 15, tzinfo=timezone.utc),
        )


def test_evidence_cannot_claim_unknown_or_duplicate_cost_fields():
    value = model()
    value["evidence"][0]["covers"] = [
        "fee_bps", "spread_bps", "slippage_bps", "market_impact_bps",
        "funding_bps_per_bar", "borrow_bps_per_bar", "contract_multiplier",
        "maker_fee_bps",
    ]
    with pytest.raises(ValueError, match="unsupported"):
        validate_transaction_cost_model(
            value, identity="kraken:spot:BTC/USD",
            now=datetime(2026, 11, 15, tzinfo=timezone.utc),
        )


def test_signed_funding_rate_is_allowed_but_other_costs_stay_nonnegative():
    value = model()
    value["funding_bps_per_bar"] = -4.5
    result = validate_transaction_cost_model(
        value, identity="kraken:spot:BTC/USD",
        now=datetime(2026, 11, 15, tzinfo=timezone.utc),
    )
    assert result["funding_bps_per_bar"] == -4.5

    value["fee_bps"] = -1
    with pytest.raises(ValueError, match="fee_bps"):
        validate_transaction_cost_model(
            value, identity="kraken:spot:BTC/USD",
            now=datetime(2026, 11, 15, tzinfo=timezone.utc),
        )


def test_funding_schedule_is_validated_and_normalized_as_events():
    value = model()
    value["funding_schedule_coverage"] = {
        "start": "2026-11-15T00:00:00Z", "end": "2026-11-15T10:00:00Z",
    }
    value["funding_schedule"] = [{
        "timestamp": "2026-11-15T08:00:00Z",
        "rate_bps": 3.25,
        "source": "Venue historical funding history",
        "uri": "https://example.com/funding",
        "license_status": "open",
        "observed_at": "2026-11-15T09:00:00Z",
    }]
    result = validate_transaction_cost_model(
        value, identity="kraken:spot:BTC/USD",
        now=datetime(2026, 11, 15, 10, tzinfo=timezone.utc),
    )
    assert result["funding_schedule"][0]["rate_bps"] == 3.25
    assert result["funding_schedule"][0]["timestamp"] == "2026-11-15T08:00:00+00:00"


@pytest.mark.parametrize("change", [
    {"timestamp": "2026-11-15T08:00:00"},
    {"rate_bps": True},
    {"rate_bps": float("nan")},
    {"uri": "http://example.com/funding"},
    {"license_status": "unknown"},
])
def test_invalid_funding_schedule_evidence_fails_closed(change):
    value = model()
    event = {
        "timestamp": "2026-11-15T08:00:00Z", "rate_bps": 2.0,
        "source": "Venue funding history", "uri": "https://example.com/funding",
        "license_status": "open", "observed_at": "2026-11-15T09:00:00Z",
    }
    event.update(change)
    value["funding_schedule_coverage"] = {
        "start": "2026-11-15T00:00:00Z", "end": "2026-11-15T10:00:00Z",
    }
    value["funding_schedule"] = [event]
    with pytest.raises(ValueError):
        validate_transaction_cost_model(
            value, identity="kraken:spot:BTC/USD",
            now=datetime(2026, 11, 15, 10, tzinfo=timezone.utc),
        )


def test_duplicate_funding_settlement_times_are_rejected():
    value = model()
    event = {
        "timestamp": "2026-11-15T08:00:00Z", "rate_bps": 2.0,
        "source": "Venue funding history", "uri": "https://example.com/funding",
        "license_status": "open", "observed_at": "2026-11-15T09:00:00Z",
    }
    value["funding_schedule_coverage"] = {
        "start": "2026-11-15T00:00:00Z", "end": "2026-11-15T10:00:00Z",
    }
    value["funding_schedule"] = [event, dict(event)]
    with pytest.raises(ValueError, match="duplicate"):
        validate_transaction_cost_model(
            value, identity="kraken:spot:BTC/USD",
            now=datetime(2026, 11, 15, 10, tzinfo=timezone.utc),
        )



def test_market_order_cost_model_rejects_maker_only_fee_basis():
    value = model()
    value["fee_basis"] = "maker"
    with pytest.raises(ValueError, match="fee_basis must be 'taker'"):
        validate_transaction_cost_model(
            value, identity="kraken:spot:BTC/USD",
            now=datetime(2026, 11, 15, tzinfo=timezone.utc),
        )


def test_fee_basis_must_be_supported_by_field_level_evidence():
    value = model()
    value["evidence"][0]["covers"].remove("fee_basis")
    with pytest.raises(ValueError, match="does not cover: fee_basis"):
        validate_transaction_cost_model(
            value, identity="kraken:spot:BTC/USD",
            now=datetime(2026, 11, 15, tzinfo=timezone.utc),
        )
