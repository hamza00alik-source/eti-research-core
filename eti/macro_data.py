"""Vintage-aware macroeconomic observations with strict point-in-time access.

This module is intentionally separate from OHLCV. It provides no network
provider and does not claim that any source is free, reachable, or licensed.
A provider must return immutable observations with explicit release/vintage
timestamps and an explicit permitted-use assertion.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Iterable, Protocol


def _aware_utc(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class MacroObservation:
    """One value for one reference period in one published data vintage."""

    series_id: str
    observation_period: datetime
    value: float
    unit: str
    release_timestamp: datetime
    vintage_timestamp: datetime
    source: str
    license_status: str
    license_reference: str
    retrieved_at: datetime

    def __post_init__(self) -> None:
        for name in ("series_id", "unit", "source", "license_status", "license_reference"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("value must be a finite number")
        if not math.isfinite(float(self.value)):
            raise ValueError("value must be a finite number")
        for name in (
            "observation_period", "release_timestamp", "vintage_timestamp", "retrieved_at"
        ):
            object.__setattr__(self, name, _aware_utc(name, getattr(self, name)))
        if self.license_status.strip().lower() != "permitted":
            raise ValueError("macro observation is not marked as permitted for this use")
        if self.vintage_timestamp < self.release_timestamp:
            raise ValueError("vintage_timestamp cannot precede release_timestamp")
        if self.retrieved_at < self.release_timestamp:
            raise ValueError("retrieved_at cannot precede release_timestamp")

    @property
    def identity(self) -> tuple[str, datetime, datetime]:
        return (self.series_id, self.observation_period, self.vintage_timestamp)

    @property
    def available_at(self) -> datetime:
        """The value is usable only once both release and vintage are public."""
        return max(self.release_timestamp, self.vintage_timestamp)


class MacroObservationProvider(Protocol):
    """Provider contract; implementation owns authentication, terms and retrieval."""

    def fetch_observations(self, series_id: str) -> Iterable[MacroObservation]:
        ...


@dataclass(frozen=True)
class MacroSeriesSnapshot:
    series_id: str
    as_of: datetime
    observations: tuple[MacroObservation, ...]
    source: str
    license_status: str
    point_in_time: bool = True


class MacroDataHub:
    """Point-in-time macro series selector; never treats macro data as candles."""

    def __init__(self, providers: dict[str, MacroObservationProvider]):
        if not isinstance(providers, dict) or not providers:
            raise ValueError("at least one named macro provider is required")
        normalized: dict[str, MacroObservationProvider] = {}
        for name, provider in providers.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("provider names must be non-empty strings")
            if name in normalized:
                raise ValueError(f"duplicate provider: {name}")
            if not callable(getattr(provider, "fetch_observations", None)):
                raise TypeError(f"provider {name} does not implement fetch_observations")
            normalized[name] = provider
        self._providers = normalized

    def fetch_as_of(
        self, provider_name: str, series_id: str, decision_time: datetime
    ) -> MacroSeriesSnapshot:
        if not isinstance(series_id, str) or not series_id.strip():
            raise ValueError("series_id must be a non-empty string")
        decision = _aware_utc("decision_time", decision_time)
        try:
            provider = self._providers[provider_name]
        except KeyError as exc:
            raise LookupError(f"unknown macro provider: {provider_name}") from exc

        rows = tuple(provider.fetch_observations(series_id))
        if not rows:
            raise LookupError(f"provider returned no observations for {series_id}")
        seen: set[tuple[str, datetime, datetime]] = set()
        eligible: dict[datetime, MacroObservation] = {}
        for row in rows:
            if not isinstance(row, MacroObservation):
                raise TypeError("providers must return MacroObservation records")
            if row.series_id != series_id:
                raise ValueError("provider returned an observation for a different series")
            if row.identity in seen:
                raise ValueError(f"duplicate macro vintage identity: {row.identity}")
            seen.add(row.identity)
            if row.available_at > decision:
                continue
            previous = eligible.get(row.observation_period)
            if previous is None or (
                row.vintage_timestamp, row.release_timestamp
            ) > (
                previous.vintage_timestamp, previous.release_timestamp
            ):
                eligible[row.observation_period] = row

        if not eligible:
            raise LookupError(f"no {series_id} observations were available by {decision.isoformat()}")
        selected = tuple(sorted(eligible.values(), key=lambda item: item.observation_period))
        sources = {item.source for item in selected}
        licenses = {item.license_status for item in selected}
        if len(sources) != 1 or len(licenses) != 1:
            raise ValueError("a macro snapshot must have consistent source and license metadata")
        return MacroSeriesSnapshot(
            series_id=series_id, as_of=decision, observations=selected,
            source=next(iter(sources)), license_status=next(iter(licenses)),
        )


class InMemoryMacroProvider:
    """Deterministic fixture/provider useful for tests and offline validation."""

    def __init__(self, observations: Iterable[MacroObservation]):
        self._observations = tuple(observations)

    def fetch_observations(self, series_id: str) -> tuple[MacroObservation, ...]:
        return tuple(row for row in self._observations if row.series_id == series_id)
