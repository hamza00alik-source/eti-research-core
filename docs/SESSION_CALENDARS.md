# Session-calendar-aware market data quality

ETI can now validate session-based instruments against an explicitly registered
calendar instead of treating every elapsed wall-clock interval as an expected
bar. This is important for equities and other markets with regular sessions,
holidays, early closes, and daylight-saving transitions.

## Current scope

- `SessionCalendar` defines an IANA timezone, regular same-day open/close,
  weekdays, full-day holidays, and date-specific early closes.
- Expected bar timestamps are generated in local market time and returned in
  UTC. This handles daylight-saving offset changes using the system IANA
  timezone database.
- `SessionCalendarRegistry` requires an exact `(venue, asset_class,
  market_type)` key. No calendar is inferred from an asset class or symbol.
- `MarketDataHub(..., calendar_registry=registry)` applies the matching
  calendar to non-crypto data quality. The quality record carries calendar ID,
  version, expected bars, observed bars, and missing in-session bars.
- The allocator awards non-crypto continuity credit only when calendar-aware
  quality evidence is present. Without a calendar, the candidate may retain a
  data-readiness ranking, but it remains ineligible for economic evaluation.
- Crypto continuity continues to use elapsed-time gap checks.

## Explicit limitations

This is a regular-session calendar primitive, not a complete exchange-calendar
database. Overnight sessions, multiple intraday segments, auction phases,
unscheduled halts, instrument-specific exceptions, and venue-specific trading
rules are not modeled. No real exchange calendars are bundled or asserted as
verified. An operator must supply calendar data from a suitable source and
track its source, license, validity and version before registering it.

The calendar registry is injected by the caller; the default public market hub
does not guess or auto-assign calendars. A missing calendar therefore fails
closed for economic evaluation.

## Example

```python
from datetime import time
from eti.session_calendar import SessionCalendar, SessionCalendarRegistry
from eti.market_data_hub import build_public_market_hub

calendars = SessionCalendarRegistry()
calendars.register(
    "yahoo", "equity", "spot",
    SessionCalendar(
        calendar_id="operator-reviewed-us-equity",
        timezone_name="America/New_York",
        open_time=time(9, 30),
        close_time=time(16, 0),
        weekdays=(0, 1, 2, 3, 4),
        holidays=frozenset(),  # Populate from a verified calendar source.
        version="2026.1",
    ),
)
hub = build_public_market_hub(calendar_registry=calendars)
```

A calendar registration is not proof of its correctness or licensing. It is
operator-supplied evidence and must be reviewed before it is used for research.
No model training, order execution, or profitability claim is enabled by this
feature.
