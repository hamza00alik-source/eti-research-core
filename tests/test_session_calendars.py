from datetime import date, datetime, timezone

import pytest

from eti.session_calendars import WeeklySessionCalendar


def nyse_calendar():
    return WeeklySessionCalendar(
        calendar_id="TEST-NYSE",
        timezone_name="America/New_York",
        sessions={i: ("09:30", "16:00") for i in range(5)},
        holidays=frozenset({date(2026, 1, 1)}),
        early_closes={date(2026, 11, 27): "13:00"},
    )


def test_session_calendar_respects_weekends_holidays_and_early_closes():
    calendar = nyse_calendar()
    expected = calendar.expected_bar_timestamps(
        datetime(2026, 11, 26, 14, 0, tzinfo=timezone.utc),
        datetime(2026, 11, 30, 22, 0, tzinfo=timezone.utc),
        "1h",
    )
    local = [value.astimezone(calendar.zone) for value in expected]
    assert all(value.weekday() < 5 for value in local)
    assert sum(value.date() == date(2026, 11, 27) for value in local) == 3
    assert not any(value.date() in {date(2026, 11, 28), date(2026, 11, 29)} for value in local)


def test_session_calendar_handles_dst_using_iana_timezone():
    calendar = nyse_calendar()
    before = calendar.expected_bar_timestamps(
        datetime(2026, 3, 6, tzinfo=timezone.utc),
        datetime(2026, 3, 6, 23, tzinfo=timezone.utc), "1h",
    )
    after = calendar.expected_bar_timestamps(
        datetime(2026, 3, 9, tzinfo=timezone.utc),
        datetime(2026, 3, 9, 23, tzinfo=timezone.utc), "1h",
    )
    assert before[0].hour == 14
    assert after[0].hour == 13


def test_session_calendar_elapsed_time_excludes_weekends_and_holidays():
    calendar = nyse_calendar()
    start = datetime(2026, 11, 27, 18, 0, tzinfo=timezone.utc)  # Friday 13:00 local
    end = datetime(2026, 11, 30, 15, 0, tzinfo=timezone.utc)  # Monday 10:00 local
    assert calendar.elapsed_trading_seconds(start, end) == 1800


@pytest.mark.parametrize("kwargs", [
    {"calendar_id": "", "timezone_name": "UTC", "sessions": {0: ("09:00", "17:00")}},
    {"calendar_id": "x", "timezone_name": "Not/AZone", "sessions": {0: ("09:00", "17:00")}},
    {"calendar_id": "x", "timezone_name": "UTC", "sessions": {}},
    {"calendar_id": "x", "timezone_name": "UTC", "sessions": {7: ("09:00", "17:00")}},
])
def test_invalid_session_calendar_configuration_fails_closed(kwargs):
    with pytest.raises(ValueError):
        WeeklySessionCalendar(**kwargs)


def test_session_calendar_rejects_naive_timestamps():
    calendar = nyse_calendar()
    with pytest.raises(ValueError, match="timezone-aware"):
        calendar.expected_bar_timestamps(
            datetime(2026, 1, 1), datetime(2026, 1, 2, tzinfo=timezone.utc), "1h"
        )
