from datetime import date, datetime, time, timezone

import pandas as pd
import pytest

from eti.adapters import AdapterCapabilities
from eti.instruments import InstrumentSpec
from eti.market_data_hub import MarketDataHub
from eti.research_allocator import allocate_research_budget
from eti.session_calendar import (
    SessionCalendar, SessionCalendarRegistry, session_quality, timeframe_delta,
)


def ny_calendar(**kwargs):
    return SessionCalendar(
        calendar_id="TEST-NYSE", timezone_name="America/New_York",
        open_time=time(9, 30), close_time=time(16, 0), **kwargs,
    )


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_session_expected_bars_skip_weekends_holidays_and_early_closes():
    calendar = ny_calendar(
        holidays=frozenset({date(2026, 1, 6)}),
        early_closes=((date(2026, 1, 7), time(13, 0)),),
    )
    start, end = utc("2026-01-05T14:30:00Z"), utc("2026-01-09T20:30:00Z")
    expected = calendar.expected_bar_timestamps(start, end, "1h")
    # Mon: 7 bars, Tue holiday: 0, Wed early close: 4, Thu/Fri: 7 each.
    assert len(expected) == 25
    assert all(x.weekday() < 5 for x in expected)
    assert all(x.date() != date(2026, 1, 6) for x in expected)
    assert sum(x.date() == date(2026, 1, 7) for x in expected) == 4


def test_session_calendar_handles_daylight_saving_time():
    calendar = ny_calendar()
    # After the US spring DST transition, 09:30 New York is 13:30 UTC.
    expected = calendar.expected_bar_timestamps(
        utc("2026-03-09T13:30:00Z"), utc("2026-03-09T15:30:00Z"), "1h"
    )
    assert expected == [
        utc("2026-03-09T13:30:00Z"),
        utc("2026-03-09T14:30:00Z"),
        utc("2026-03-09T15:30:00Z"),
    ]


def test_session_quality_counts_only_missing_open_session_bars():
    calendar = ny_calendar()
    observed = [
        utc("2026-01-05T14:30:00Z"),
        utc("2026-01-05T15:30:00Z"),
        utc("2026-01-05T17:30:00Z"),
    ]
    result = session_quality(observed, "1h", calendar)
    assert result["expected_session_bars"] == 4
    assert result["observed_session_bars"] == 3
    assert result["missing_interval_estimate"] == 1
    assert result["session_calendar_applied"] is True


def test_session_quality_rejects_observations_outside_session():
    with pytest.raises(ValueError, match="outside"):
        session_quality(
            [utc("2026-01-05T13:30:00Z"), utc("2026-01-05T14:30:00Z")],
            "1h", ny_calendar(),
        )


def test_calendar_rejects_overnight_sessions_and_invalid_timeframes():
    with pytest.raises(ValueError, match="overnight"):
        SessionCalendar("bad", "UTC", time(22), time(2))
    with pytest.raises(ValueError):
        timeframe_delta("1M")


def test_calendar_registry_requires_exact_instrument_identity():
    registry = SessionCalendarRegistry()
    calendar = ny_calendar()
    registry.register("yahoo", "equity", "spot", calendar)
    assert registry.get("YAHOO", "EQUITY", "SPOT") is calendar
    assert registry.get("yahoo", "forex", "spot") is None
    with pytest.raises(ValueError, match="already registered"):
        registry.register("yahoo", "equity", "spot", calendar)


class EquityAdapter:
    def __init__(self):
        self.capabilities = AdapterCapabilities(
            venue="yahoo", asset_classes=("equity",), market_types=("spot",),
            public_market_data=True, order_execution=False,
        )

    def discover_instruments(self):
        return []

    def fetch_ohlcv(self, symbol, timeframe, limit):
        stamps = pd.to_datetime([
            "2026-01-05T14:30:00Z", "2026-01-05T15:30:00Z",
            "2026-01-05T16:30:00Z", "2026-01-05T17:30:00Z",
        ])
        return pd.DataFrame({
            "timestamp": stamps, "open": [100.0] * 4, "high": [101.0] * 4,
            "low": [99.0] * 4, "close": [100.5] * 4, "volume": [1000.0] * 4,
        })


def test_market_hub_applies_registered_calendar_to_equity_quality():
    registry = SessionCalendarRegistry()
    registry.register("yahoo", "equity", "spot", ny_calendar())
    hub = MarketDataHub([EquityAdapter()], calendar_registry=registry)
    instrument = InstrumentSpec(
        symbol="SPY", venue="yahoo", asset_class="equity", market_type="spot"
    )
    snapshot = hub.fetch(instrument, "1h", 4)
    assert snapshot.quality["session_calendar_applied"] is True
    assert snapshot.quality["expected_session_bars"] == 4
    assert snapshot.quality["missing_interval_estimate"] == 0
    assert snapshot.quality["session_calendar_id"] == "TEST-NYSE"


def test_allocator_uses_session_coverage_when_calendar_evidence_exists():
    item = {
        "status": "data_ready",
        "instrument": {"venue": "yahoo", "symbol": "SPY", "asset_class": "equity", "market_type": "spot"},
        "provenance": {"requested_rows": 100, "timeframe": "1h"},
        "quality": {
            "rows": 100, "last_candle_age_seconds": 300,
            "missing_interval_estimate": 0, "max_gap_seconds": None,
            "ohlcv_validated": True, "timestamp_order": "ascending",
            "session_calendar_applied": True, "session_calendar_id": "TEST-NYSE",
            "session_calendar_version": "1", "expected_session_bars": 100,
            "observed_session_bars": 100, "session_coverage": 1.0,
        },
    }
    result = allocate_research_budget(
        {"scan_only": True, "execution_enabled": False, "candidates": [item]},
        cost_models={
            "yahoo:spot:SPY": {
                "schema_version": "eti.transaction_cost_model.v1",
                "instrument_identity": "yahoo:spot:SPY",
                "quote_currency": "USD", "settlement_currency": "USD",
                "fee_bps": 0, "spread_bps": 1, "slippage_bps": 1,
                "market_impact_bps": 0, "funding_bps_per_bar": 0,
                "borrow_bps_per_bar": 0, "contract_multiplier": 1,
                "retrieved_at": "2026-01-01T00:00:00Z",
                "valid_from": "2026-01-01T00:00:00Z",
                "valid_until": "2030-01-01T00:00:00Z",
                "evidence": [{
                    "source": "Documented broker fee schedule",
                    "uri": "https://example.com/fees",
                    "license_status": "open",
                    "observed_at": "2026-01-01T00:00:00Z",

                    "covers": ["fee_bps", "spread_bps", "slippage_bps", "market_impact_bps", "funding_bps_per_bar", "borrow_bps_per_bar", "contract_multiplier"],
                }],
            }
        },
        estimated_run_cost_usd=1, daily_spend_to_date_usd=0,
        monthly_spend_to_date_usd=0,
    )
    row = result["candidates"][0]
    assert row["economic_evaluation_eligible"] is True
    assert "session_calendar_not_applied" not in row["eligibility_reasons"]
