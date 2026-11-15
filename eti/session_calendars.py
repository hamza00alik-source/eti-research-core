"""Explicit session calendars for market-aware OHLCV continuity checks.

No exchange calendar is guessed or downloaded implicitly. Callers must supply
an authoritative weekly schedule plus holiday and early-close dates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Mapping, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd


class SessionCalendar(Protocol):
    """Calendar contract consumed by the market-data quality layer."""

    calendar_id: str

    def expected_bar_timestamps(
        self, start: datetime, end: datetime, timeframe: str
    ) -> list[datetime]: ...


def _clock(value: str | time, name: str) -> time:
    if isinstance(value, time):
        if value.tzinfo is not None:
            raise ValueError(f"{name} must be a local wall-clock time")
        return value
    if not isinstance(value, str):
        raise ValueError(f"{name} must be HH:MM or a naive time")
    try:
        parsed = time.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be HH:MM") from exc
    if parsed.tzinfo is not None:
        raise ValueError(f"{name} must not include a timezone")
    return parsed


@dataclass(frozen=True)
class WeeklySessionCalendar:
    """Weekly local-time sessions with explicit holidays and early closes.

    Weekday keys use Python's convention: Monday=0 through Sunday=6.
    Each session is anchored at its local opening time. Bars are emitted only
    when a complete timeframe fits before that session's close.
    """

    calendar_id: str
    timezone_name: str
    sessions: Mapping[int, tuple[str | time, str | time]]
    holidays: frozenset[date] = field(default_factory=frozenset)
    early_closes: Mapping[date, str | time] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.calendar_id, str) or not self.calendar_id.strip():
            raise ValueError("calendar_id must be a non-empty string")
        try:
            zone = ZoneInfo(self.timezone_name)
        except (ZoneInfoNotFoundError, TypeError) as exc:
            raise ValueError("timezone_name must be a valid IANA timezone") from exc
        if not self.sessions:
            raise ValueError("sessions must define at least one weekday")
        normalized: dict[int, tuple[time, time]] = {}
        for weekday, bounds in self.sessions.items():
            if isinstance(weekday, bool) or not isinstance(weekday, int) or weekday not in range(7):
                raise ValueError("session weekday keys must be integers from 0 to 6")
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError("each session must contain an open and close time")
            normalized[weekday] = (
                _clock(bounds[0], "session open"),
                _clock(bounds[1], "session close"),
            )
        holidays = frozenset(self.holidays)
        if any(not isinstance(day, date) or isinstance(day, datetime) for day in holidays):
            raise ValueError("holidays must contain date values")
        early: dict[date, time] = {}
        for day, close in self.early_closes.items():
            if not isinstance(day, date) or isinstance(day, datetime):
                raise ValueError("early-close keys must be dates")
            early[day] = _clock(close, "early close")
        object.__setattr__(self, "timezone_name", zone.key)
        object.__setattr__(self, "sessions", normalized)
        object.__setattr__(self, "holidays", holidays)
        object.__setattr__(self, "early_closes", early)

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_name)

    def _session_bounds(self, session_day: date) -> tuple[datetime, datetime] | None:
        if session_day in self.holidays or session_day.weekday() not in self.sessions:
            return None
        open_time, configured_close = self.sessions[session_day.weekday()]
        close_time = self.early_closes.get(session_day, configured_close)
        start = datetime.combine(session_day, open_time, self.zone)
        end_day = session_day + timedelta(days=1) if close_time <= open_time else session_day
        end = datetime.combine(end_day, close_time, self.zone)
        return start, end

    @staticmethod
    def _duration(timeframe: str) -> timedelta:
        try:
            duration = pd.Timedelta(timeframe).to_pytimedelta()
        except (TypeError, ValueError) as exc:
            raise ValueError("timeframe must be a positive fixed duration") from exc
        if duration <= timedelta(0):
            raise ValueError("timeframe must be a positive fixed duration")
        return duration

    def expected_bar_timestamps(
        self, start: datetime, end: datetime, timeframe: str
    ) -> list[datetime]:
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            raise ValueError("start and end must be datetimes")
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware")
        if end < start:
            raise ValueError("end must not precede start")
        duration = self._duration(timeframe)
        local_start, local_end = start.astimezone(self.zone), end.astimezone(self.zone)
        expected: list[datetime] = []
        day = local_start.date() - timedelta(days=1)
        final_day = local_end.date() + timedelta(days=1)
        while day <= final_day:
            bounds = self._session_bounds(day)
            if bounds:
                session_open, session_close = bounds
                cursor = session_open
                while cursor + duration <= session_close:
                    if local_start <= cursor <= local_end:
                        expected.append(cursor.astimezone(timezone.utc))
                    cursor += duration
            day += timedelta(days=1)
        return expected

    def elapsed_trading_seconds(self, start: datetime, end: datetime) -> float:
        """Elapsed scheduled-open seconds; closures do not age a feed."""
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware")
        if end <= start:
            return 0.0
        local_start, local_end = start.astimezone(self.zone), end.astimezone(self.zone)
        total = 0.0
        day = local_start.date() - timedelta(days=1)
        while day <= local_end.date() + timedelta(days=1):
            bounds = self._session_bounds(day)
            if bounds:
                session_open, session_close = bounds
                left = max(local_start, session_open)
                right = min(local_end, session_close)
                if right > left:
                    total += (right - left).total_seconds()
            day += timedelta(days=1)
        return total
