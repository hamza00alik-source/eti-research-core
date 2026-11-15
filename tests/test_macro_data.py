from datetime import datetime, timezone

import pytest

from eti.macro_data import InMemoryMacroProvider, MacroDataHub, MacroObservation


def dt(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def obs(period, value, release, vintage, *, retrieved="2026-03-01T00:00:00Z",
        series="CPI", source="fixture", license_status="permitted"):
    return MacroObservation(
        series_id=series, observation_period=dt(period), value=value, unit="index",
        release_timestamp=dt(release), vintage_timestamp=dt(vintage),
        source=source, license_status=license_status,
        license_reference="fixture terms", retrieved_at=dt(retrieved),
    )


def hub(rows):
    return MacroDataHub({"fixture": InMemoryMacroProvider(rows)})


def test_as_of_selects_latest_vintage_that_was_public_at_decision_time():
    rows = [
        obs("2025-01-01T00:00:00Z", 100, "2025-02-10T00:00:00Z", "2025-02-10T00:00:00Z"),
        obs("2025-01-01T00:00:00Z", 101, "2025-03-10T00:00:00Z", "2025-03-10T00:00:00Z"),
    ]
    result = hub(rows).fetch_as_of("fixture", "CPI", dt("2025-02-20T00:00:00Z"))
    assert len(result.observations) == 1
    assert result.observations[0].value == 100
    assert result.point_in_time is True


def test_revised_value_is_used_only_after_its_revision_vintage():
    rows = [
        obs("2025-01-01T00:00:00Z", 100, "2025-02-10T00:00:00Z", "2025-02-10T00:00:00Z"),
        obs("2025-01-01T00:00:00Z", 98, "2025-02-10T00:00:00Z", "2025-04-01T00:00:00Z"),
    ]
    before = hub(rows).fetch_as_of("fixture", "CPI", dt("2025-03-01T00:00:00Z"))
    after = hub(rows).fetch_as_of("fixture", "CPI", dt("2025-04-02T00:00:00Z"))
    assert before.observations[0].value == 100
    assert after.observations[0].value == 98


def test_future_release_or_vintage_is_filtered_out():
    rows = [
        obs("2025-01-01T00:00:00Z", 10, "2025-02-10T00:00:00Z", "2025-02-10T00:00:00Z"),
        obs("2025-02-01T00:00:00Z", 20, "2025-04-10T00:00:00Z", "2025-04-10T00:00:00Z"),
    ]
    result = hub(rows).fetch_as_of("fixture", "CPI", dt("2025-03-01T00:00:00Z"))
    assert [item.value for item in result.observations] == [10]


@pytest.mark.parametrize("field", ["observation_period", "release_timestamp", "vintage_timestamp", "retrieved_at"])
def test_naive_timestamps_are_rejected(field):
    values = dict(
        series_id="CPI", observation_period=dt("2025-01-01T00:00:00Z"),
        value=1.0, unit="index", release_timestamp=dt("2025-02-01T00:00:00Z"),
        vintage_timestamp=dt("2025-02-01T00:00:00Z"), source="fixture",
        license_status="permitted", license_reference="terms",
        retrieved_at=dt("2025-03-01T00:00:00Z"),
    )
    values[field] = datetime(2025, 1, 1)
    with pytest.raises(ValueError, match="timezone-aware"):
        MacroObservation(**values)


def test_duplicate_series_period_vintage_is_rejected():
    row = obs("2025-01-01T00:00:00Z", 1, "2025-02-01T00:00:00Z", "2025-02-01T00:00:00Z")
    with pytest.raises(ValueError, match="duplicate macro vintage"):
        hub([row, row]).fetch_as_of("fixture", "CPI", dt("2025-03-01T00:00:00Z"))


def test_unpermitted_or_unknown_license_status_fails_closed():
    with pytest.raises(ValueError, match="not marked as permitted"):
        obs("2025-01-01T00:00:00Z", 1, "2025-02-01T00:00:00Z",
            "2025-02-01T00:00:00Z", license_status="unknown")


def test_non_finite_values_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        obs("2025-01-01T00:00:00Z", float("nan"), "2025-02-01T00:00:00Z",
            "2025-02-01T00:00:00Z")


def test_wrong_series_and_inconsistent_provenance_are_rejected():
    with pytest.raises(ValueError, match="different series"):
        hub([obs("2025-01-01T00:00:00Z", 1, "2025-02-01T00:00:00Z",
                 "2025-02-01T00:00:00Z", series="GDP")]).fetch_as_of(
            "fixture", "CPI", dt("2025-03-01T00:00:00Z")
        )
    rows = [
        obs("2025-01-01T00:00:00Z", 1, "2025-02-01T00:00:00Z",
            "2025-02-01T00:00:00Z", source="source-a"),
        obs("2025-02-01T00:00:00Z", 2, "2025-02-02T00:00:00Z",
            "2025-02-02T00:00:00Z", source="source-b"),
    ]
    with pytest.raises(ValueError, match="consistent source"):
        hub(rows).fetch_as_of("fixture", "CPI", dt("2025-03-01T00:00:00Z"))


def test_empty_as_of_result_fails_instead_of_backfilling_latest_value():
    rows = [obs("2025-01-01T00:00:00Z", 1, "2025-02-01T00:00:00Z",
                "2025-02-01T00:00:00Z")]
    with pytest.raises(LookupError, match="no CPI observations"):
        hub(rows).fetch_as_of("fixture", "CPI", dt("2025-01-15T00:00:00Z"))
