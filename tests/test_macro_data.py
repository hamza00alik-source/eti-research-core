from datetime import datetime, timezone

import pytest

from eti.macro_data import (
    InMemoryMacroProvider,
    MacroDataHub,
    MacroObservation,
)


UTC = timezone.utc


def dt(day, hour=0):
    return datetime(2024, 1, day, hour, tzinfo=UTC)


def obs(value, *, period=None, released=None, vintage=None, license_status="open",
        series="CPI", source="fixture", provenance="fixture:record"):
    released = released or dt(2)
    vintage = vintage or released
    return MacroObservation(
        series_id=series, period=period or dt(1), value=value, unit="index",
        released_at=released, vintage_at=vintage, source=source,
        license_status=license_status, provenance=provenance,
    )


def hub(rows):
    return MacroDataHub([InMemoryMacroProvider("fixture", rows)])


def test_as_of_selects_latest_vintage_available_at_decision_time():
    initial = obs(100.0)
    revision = obs(101.0, released=dt(2), vintage=dt(5))
    result = hub([initial, revision]).fetch_as_of("fixture", "CPI", dt(4))
    assert [item.value for item in result.observations] == [100.0]
    later = hub([initial, revision]).fetch_as_of("fixture", "CPI", dt(6))
    assert [item.value for item in later.observations] == [101.0]


def test_future_release_is_not_visible():
    future = obs(100.0, released=dt(5), vintage=dt(5))
    result = hub([future]).fetch_as_of("fixture", "CPI", dt(4))
    assert result.observations == ()


def test_future_revision_does_not_overwrite_initial_historical_vintage():
    initial = obs(2.0)
    revision = obs(3.0, released=dt(2), vintage=dt(7))
    result = hub([initial, revision]).fetch_as_of("fixture", "CPI", dt(6))
    assert result.observations[0].value == 2.0


@pytest.mark.parametrize("field", ["period", "released", "vintage"])
def test_naive_timestamps_are_rejected(field):
    values = {"period": dt(1), "released": dt(2), "vintage": dt(2)}
    values[field] = datetime(2024, 1, 1)
    with pytest.raises(ValueError, match="timezone-aware"):
        obs(1.0, **values)


def test_vintage_cannot_precede_release():
    with pytest.raises(ValueError, match="cannot precede"):
        obs(1.0, released=dt(5), vintage=dt(4))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_non_finite_or_boolean_values_are_rejected(value):
    with pytest.raises(ValueError, match="finite number"):
        obs(value)


def test_duplicate_observation_identity_fails_closed():
    row = obs(1.0)
    with pytest.raises(ValueError, match="duplicate macro observation"):
        hub([row, row]).fetch_as_of("fixture", "CPI", dt(6))


def test_restricted_or_unknown_license_fails_closed():
    row = obs(1.0, license_status="restricted")
    with pytest.raises(PermissionError, match="license is not approved"):
        hub([row]).fetch_as_of("fixture", "CPI", dt(6))


def test_provider_must_not_return_another_series():
    class MaliciousProvider:
        name = "malicious"

        def fetch_series(self, series_id):
            return (obs(1.0, series="GDP"),)

    data = MacroDataHub([MaliciousProvider()])
    with pytest.raises(ValueError, match="different series"):
        data.fetch_as_of("malicious", "CPI", dt(6))


def test_providers_are_not_implicitly_merged():
    first = InMemoryMacroProvider("first", [obs(1.0)])
    second = InMemoryMacroProvider("second", [obs(2.0, period=dt(3))])
    data = MacroDataHub([first, second])
    result = data.fetch_as_of("first", "CPI", dt(6))
    assert len(result.observations) == 1
    assert result.provider == "first"


def test_unknown_provider_fails_closed():
    with pytest.raises(LookupError, match="unknown macro provider"):
        hub([]).fetch_as_of("missing", "CPI", dt(6))


def test_vintage_must_be_timezone_aware_and_provider_name_unique():
    with pytest.raises(ValueError, match="timezone-aware"):
        MacroObservation("CPI", dt(1), 1.0, "index", dt(2), datetime(2024, 1, 2),
                         "source", "open", "fixture")
    provider = InMemoryMacroProvider("fixture", [obs(1.0)])
    with pytest.raises(ValueError, match="already registered"):
        MacroDataHub([provider, provider])
