# Session-aware market data quality

ETI does not infer exchange hours from asset class or ticker symbols. A session
calendar must be explicitly registered for a venue/asset-class/market-type
combination. The initial implementation is a deterministic calendar engine,
not a feed of official exchange holidays.

## Calendar contract

`WeeklySessionCalendar` accepts:
- an IANA timezone (DST transitions are handled by `zoneinfo`);
- local weekly open/close times, Monday=0 through Sunday=6;
- explicit holiday dates;
- explicit early-close dates and local close times.

Bars are anchored to each session open and are counted only when a complete
bar fits inside the session. Overnight sessions are supported. Calendar
closures are not counted as missing candles. Feed age is measured in scheduled
trading seconds, so a weekend does not make a closed-market feed appear stale.

## Governance

A calendar is a supplied configuration, not proof that the configuration is
official, current, or licensed. Operators must verify its source, holiday
coverage, special sessions, venue, instrument applicability, and update policy.
ETI does not yet ship exchange-specific holiday data or infer one venue's
calendar for another. Without a registered calendar, non-crypto continuity
remains blocked in the research allocator. Live execution remains disabled.

Programmatic registration:

```python
from datetime import date
from eti.market_data_hub import MarketDataHub
from eti.session_calendars import WeeklySessionCalendar

hub = MarketDataHub(adapters)
hub.register_session_calendar(
    venue="example",
    asset_class="equity",
    market_type="spot",
    calendar=WeeklySessionCalendar(
        calendar_id="EXAMPLE-EQUITY",
        timezone_name="America/New_York",
        sessions={i: ("09:30", "16:00") for i in range(5)},
        holidays=frozenset({date(2026, 1, 1)}),
    ),
)
```

The scanner and allocator then preserve the calendar diagnostics in the scan
artifact and require validated session coverage before a non-crypto candidate
can be considered economically evaluable.
