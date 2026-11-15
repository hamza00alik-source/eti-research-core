"""Explicit session calendars for session-based market-data quality checks.

Calendars are supplied by the operator or a verified venue-calendar adapter.
There are deliberately no guessed defaults: an unknown venue/session remains
unvalidated and downstream economic evaluation must stay blocked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import re
from typing import Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def timeframe_delta(value: str) -> timedelta:
    """Parse a fixed bar interval; calendar months are intentionally unsupported."""
    if not isinstance(value, str):
        raise ValueError("timeframe must be a fixed-duration string")
    match = re.fullmatch(r"(\\d+)([smhdw])", value.strip().lower())
    if not match:
        raise ValueError(f"unsupported fixed-duration timeframe: {value}")
    amount, unit = int(match.group(1)), match.group(2)
    if amount < 1:
        raise ValueError("timeframe amount must be positive")
    return timedelta(seconds=amount * {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit])


@dataclass(frozen=True)
class SessionCalendar:
    """A deliberately simple regular-session calendar with holidays and early closes.

    Session bars are timestamped at their scheduled opening instant. Regular
    sessions must open and close on the same local calendar day. Markets with
    overnight sessions or multiple intraday segments need a richer calendar
    implementation rather than an approximation with this class.
    """

    calendar_id: str
    timezone_name: str
    open_time: time
    close_time: time
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    holidays: frozenset[date] = field(default_factory=frozenset)
    early_closes: tuple[tuple[date, time], ...] = ()
    version: str = "1"

    def __post_init__(self) -> None:
        if not isinstance(self.calendar_id, str) or not self.calendar_id.strip():
            raise ValueError("calendar_id must be non-empty")
        try:
            ZoneInfo(self.timezone_name)
        except (ZoneInfoNotFoundError, TypeError) as exc:
            raise ValueError("timezone_name must be a valid IANA timezone") from exc
        if not isinstance(self.open_time, time) or not isinstance(self.close_time, time):
            raise ValueError("open_time and close_time must be datetime.time values")
        if self.open_time.tzinfo is not None or self.close_time.tzinfo is not None:
            raise ValueError("session times must be local wall-clock times without tzinfo")
        if self.close_time <= self.open_time:
            raise ValueError("overnight or zero-length sessions are unsupported")
        if not self.weekdays or any(
            isinstance(day, bool) or not isinstance(day, int) or not 0 <= day <= 6
            for day in self.weekdays
        ):
            raise ValueError("weekdays must contain weekday integers from 0 through 6")
        if len(set(self.weekdays)) != len(self.weekdays):
            raise ValueError("weekdays cannot contain duplicates")
        holiday_set = frozenset(self.holidays)
        if any(not isinstance(day, date) or isinstance(day, datetime) for day in holiday_set):
            raise ValueError("holidays must contain datetime.date values")
        early = dict(self.early_closes)
        if len(early) != len(self.early_closes):
            raise ValueError("early close dates must be unique")
        for day, close in early.items():
            if not isinstance(day, date) or isinstance(day, datetime):
                raise ValueError("early close keys must be datetime.date values")
            if not isinstance(close, time) or close.tzinfo is not None:
                raise ValueError("early close values must be local wall-clock times")
            if not self.open_time < close <= self.close_time:
                raise ValueError("early close must be after open and no later than regular close")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("calendar version must be non-empty")
        object.__setattr__(self, "holidays", holiday_set)

    def expected_bar_timestamps(
        self, start: datetime, end: datetime, timeframe: str
    ) -> list[datetime]:
        """Return expected UTC bar-open timestamps within the inclusive range."""
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("start must be timezone-aware")
        if end.tzinfo is None or end.utcoffset() is None:
            raise ValueError("end must be timezone-aware")
        if end < start:
            raise ValueError("end must not precede start")
        interval = timeframe_delta(timeframe)
        zone = ZoneInfo(self.timezone_name)
        first_local = start.astimezone(zone).date()
        last_local = end.astimezone(zone).date()
        expected: list[datetime] = []
        day = first_local
        early_closes = dict(self.early_closes)
        while day <= last_local:
            if day.weekday() in self.weekdays and day not in self.holidays:
                close = early_closes.get(day, self.close_time)
                cursor = datetime.combine(day, self.open_time, tzinfo=zone)
                session_end = datetime.combine(day, close, tzinfo=zone)
                while cursor < session_end:
                    instant = cursor.astimezone(timezone.utc)
                    if start <= instant <= end:
                        expected.append(instant)
                    cursor += interval
            day += timedelta(days=1)
        return expected


class SessionCalendarRegistry:
    """Exact-key calendar registry; never infers a calendar from an asset label."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str, str], SessionCalendar] = {}

    def register(
        self, venue: str, asset_class: str, market_type: str, calendar: SessionCalendar
    ) -> None:
        key = (venue.strip().lower(), asset_class.strip().lower(), market_type.strip().lower())
        if not all(key):
            raise ValueError("venue, asset_class and market_type must be non-empty")
        if not isinstance(calendar, SessionCalendar):
            raise TypeError("calendar must be a SessionCalendar")
        if key in self._items:
            raise ValueError(f"Calendar already registered for {key}")
        self._items[key] = calendar

    def get(self, venue: str, asset_class: str, market_type: str) -> SessionCalendar | None:
        key = (venue.strip().lower(), asset_class.strip().lower(), market_type.strip().lower())
        return self._items.get(key)

    def describe(self) -> list[dict[str, str]]:
        return [
            {"venue": key[0], "asset_class": key[1], "market_type": key[2],
             "calendar_id": calendar.calendar_id, "version": calendar.version,
             "timezone": calendar.timezone_name}
            for key, calendar in sorted(self._items.items())
        ]


def session_quality(
    timestamps: Iterable[datetime], timeframe: str, calendar: SessionCalendar
) -> dict[str, object]:
    """Calculate expected session bars and missing bars using one explicit calendar."""
    observed = list(timestamps)
    if not observed:
        raise ValueError("timestamps cannot be empty")
    start, end = observed[0], observed[-1]
    expected = calendar.expected_bar_timestamps(start, end, timeframe)
    observed_utc = {
        stamp.astimezone(timezone.utc) for stamp in observed
        if stamp.tzinfo is not None and stamp.utcoffset() is not None
    }
    expected_set = set(expected)
    unexpected = sorted(observed_utc - expected_set)
    if unexpected:
        raise ValueError("observed bars fall outside the registered session calendar")
    missing = len(expected_set - observed_utc)
    return {
        "session_calendar_applied": True,
        "session_calendar_id": calendar.calendar_id,
        "session_calendar_version": calendar.version,
        "expected_session_bars": len(expected_set),
        "observed_session_bars": len(observed_utc),
        "missing_interval_estimate": missing,
        "session_coverage": (
            len(observed_utc) / len(expected_set) if expected_set else 0.0
        ),
    }
