import math

from eti.derivatives_quality import assess_observation_history


def test_rejects_bad_epochs_nonfinite_and_negative_open_interest():
    rows = [
        {"timestamp": 1_700_000_000_000, "open_interest": 10},
        {"timestamp": 1_700_000_000, "open_interest": 12},  # seconds, not milliseconds
        {"timestamp": 1_700_000_000_001, "open_interest": float("nan")},
        {"timestamp": 1_700_000_000_002, "open_interest": -1},
        {"timestamp": 1_700_000_000_003, "open_interest": float("inf")},
    ]
    clean, quality = assess_observation_history(
        rows, value_key="open_interest", allow_negative=False
    )
    assert len(clean) == 1
    assert quality["invalid_timestamp_count"] == 1
    assert quality["invalid_value_count"] == 3


def test_reports_duplicates_and_out_of_order_without_leaking_bad_rows():
    rows = [
        {"timestamp": 1_700_000_000_000, "funding_rate": 0.001},
        {"timestamp": 1_700_028_800_000, "funding_rate": -0.001},
        {"timestamp": 1_700_000_000_000, "funding_rate": 0.002},
    ]
    clean, quality = assess_observation_history(
        rows, value_key="funding_rate", allow_negative=True
    )
    assert [item["timestamp"] for item in clean] == sorted(item["timestamp"] for item in clean)
    assert quality["duplicate_timestamp_count"] == 1
    assert quality["out_of_order_count"] == 1
    assert clean[0]["funding_rate"] == 0.001


def test_preserves_zero_open_interest_and_distinguishes_units():
    rows = [
        {"timestamp": 1_700_000_000_000, "open_interest": 0, "open_interest_unit": "contracts"},
        {"timestamp": 1_700_003_600_000, "open_interest": 5, "open_interest_unit": "contracts"},
    ]
    clean, quality = assess_observation_history(
        rows, value_key="open_interest", allow_negative=False
    )
    assert clean[0]["open_interest"] == 0.0
    assert quality["valid_observations"] == 2
    assert quality["interval_regularity"] == "insufficient_samples"


def test_funding_intervals_are_reported_not_assumed_to_be_eight_hours():
    start = 1_700_000_000_000
    rows = [
        {"timestamp": start, "funding_rate": 0.001},
        {"timestamp": start + 3_600_000, "funding_rate": 0.002},
        {"timestamp": start + 7_200_000, "funding_rate": -0.001},
    ]
    clean, quality = assess_observation_history(
        rows, value_key="funding_rate", allow_negative=True
    )
    assert len(clean) == 3
    assert quality["median_observation_interval_ms"] == 3_600_000
    assert quality["interval_regularity"] == "regular"
