"""Point-in-time macroeconomic observations with explicit release/vintage semantics.

This module is intentionally separate from OHLCV. It provides no network provider,
does not infer release times, and never substitutes today's revised value into
a historical decision. Integrations must supply trustworthy release/vintage data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Iterable, Protocol


LICENSE_STATUSES = frozenset({"open", "licensed", "restricted", "unknown"})


def _aware_utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware datetime")
    return value


@dataclass(frozen=True)
class MacroObservation:
    """One released value for a reference period and a specific data vintage.

    available_at is the later of the release timestamp and vintage timestamp.
    For an initial release these are commonly equal; revisions receive a later
    vintage_at. All timestamps must be timezone-aware.
    """

    series_id: str
    period: datetime
    value: float
    unit: str
    released_at: datetime
    vintage_at: datetime
    source: str
    license_status: str
    provenance: str

    def __post_init__(self) -> None:
        for name in ("series_id", "unit", "source", "provenance"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("period", "released_at", "vintage_at"):
            _aware_utc(getattr(self, name), name)
        if self.vintage_at < self.released_at:
            raise ValueError("vintage_at cannot precede released_at")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("value must be a finite number")
        if not math.isfinite(float(self.value)):
            raise ValueError("value must be a finite number")
        if self.license_status not in LICENSE_STATUSES:
            raise ValueError(f"license_status must be one of {sorted(LICENSE_STATUSES)}")

    @property
    def available_at(self) -> datetime:
        return max(self.released_at, self.vintage_at)


class MacroSeriesProvider(Protocol):
    """Provider contract; returned records must include actual release/vintage data."""

    name: str

    def fetch_series(self, series_id: str) -> Iterable[MacroObservation]:
        ...


class InMemoryMacroProvider:
    """Deterministic fixture/provider for tests and explicitly supplied datasets."""

    def __init__(self, name: str, observations: Iterable[MacroObservation]):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("provider name must be non-empty")
        self.name = name
        self._observations = tuple(observations)

    def fetch_series(self, series_id: str) -> tuple[MacroObservation, ...]:
        return tuple(item for item in self._observations if item.series_id == series_id)


@dataclass(frozen=True)
class MacroSnapshot:
    series_id: str
    decision_time: datetime
    observations: tuple[MacroObservation, ...]
    provider: str
    license_statuses: tuple[str, ...]


class MacroDataHub:
    """Fetch and select only macro vintages available at a historical decision time."""

    def __init__(
        self,
        providers: Iterable[MacroSeriesProvider] = (),
        *,
        allowed_license_statuses: Iterable[str] = ("open", "licensed"),
    ):
        self._providers: dict[str, MacroSeriesProvider] = {}
        allowed = frozenset(allowed_license_statuses)
        if not allowed or not allowed.issubset(LICENSE_STATUSES):
            raise ValueError("allowed_license_statuses must be a non-empty valid subset")
        self.allowed_license_statuses = allowed
        for provider in providers:
            self.register(provider)

    def register(self, provider: MacroSeriesProvider) -> None:
        name = getattr(provider, "name", None)
        fetch = getattr(provider, "fetch_series", None)
        if not isinstance(name, str) or not name.strip() or not callable(fetch):
            raise TypeError("provider must expose a non-empty name and fetch_series")
        key = name.strip().lower()
        if key in self._providers:
            raise ValueError(f"macro provider already registered: {key}")
        self._providers[key] = provider

    def fetch_as_of(
        self, provider_name: str, series_id: str, decision_time: datetime
    ) -> MacroSnapshot:
        """Return latest eligible vintage per reference period as of decision_time.

        A record is eligible only when both its release and vintage are available
        by decision_time. Restricted/unknown licenses fail closed by default.
        Duplicate (series, period, vintage) identities fail rather than choosing
        an arbitrary value. Different providers are never silently merged.
        """
        _aware_utc(decision_time, "decision_time")
        if not isinstance(series_id, str) or not series_id.strip():
            raise ValueError("series_id must be non-empty")
        key = provider_name.strip().lower()
        try:
            provider = self._providers[key]
        except (KeyError, AttributeError) as exc:
            raise LookupError(f"unknown macro provider: {provider_name}") from exc

        rows = tuple(provider.fetch_series(series_id.strip()))
        identities: set[tuple[str, datetime, datetime]] = set()
        eligible: dict[datetime, MacroObservation] = {}
        for row in rows:
            if not isinstance(row, MacroObservation):
                raise TypeError("provider returned a non-MacroObservation record")
            if row.series_id != series_id.strip():
                raise ValueError("provider returned an observation for a different series")
            identity = (row.series_id, row.period, row.vintage_at)
            if identity in identities:
                raise ValueError(f"duplicate macro observation identity: {identity}")
            identities.add(identity)
            if row.available_at > decision_time:
                continue
            if row.license_status not in self.allowed_license_statuses:
                raise PermissionError(
                    f"macro observation license is not approved: {row.license_status}"
                )
            current = eligible.get(row.period)
            if current is None or row.available_at > current.available_at:
                eligible[row.period] = row
            elif row.available_at == current.available_at:
                raise ValueError(
                    f"ambiguous vintages for {row.series_id} period {row.period}"
                )

        selected = tuple(sorted(eligible.values(), key=lambda item: item.period))
        return MacroSnapshot(
            series_id=series_id.strip(),
            decision_time=decision_time,
            observations=selected,
            provider=provider.name,
            license_statuses=tuple(sorted({item.license_status for item in selected})),
        )
