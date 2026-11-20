import json

import pandas as pd
import pytest

from eti.research_evidence import archive_ohlcv_snapshot, verify_ohlcv_archive


def frame():
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
        "open": [100.0, 101.0, 102.0],
        "high": [102.0, 103.0, 104.0],
        "low": [99.0, 100.0, 101.0],
        "close": [101.0, 102.0, 103.0],
        "volume": [10.0, 11.0, 12.0],
    })


def test_archive_is_content_addressed_and_verifiable(tmp_path):
    first = archive_ohlcv_snapshot(frame(), tmp_path)
    second = archive_ohlcv_snapshot(frame(), tmp_path)
    assert first == second
    assert verify_ohlcv_archive(first, tmp_path)["verified"] is True


def test_changed_market_data_produces_a_different_object(tmp_path):
    original = frame()
    first = archive_ohlcv_snapshot(original, tmp_path)
    changed = original.copy()
    changed.loc[1, "close"] = 999.0
    second = archive_ohlcv_snapshot(changed, tmp_path)
    assert first["sha256"] != second["sha256"]
    assert first["relative_path"] != second["relative_path"]


def test_archive_verification_detects_tampering(tmp_path):
    receipt = archive_ohlcv_snapshot(frame(), tmp_path)
    path = tmp_path / receipt["relative_path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_ohlcv_archive(receipt, tmp_path)


def test_archive_verification_rejects_path_traversal(tmp_path):
    receipt = archive_ohlcv_snapshot(frame(), tmp_path)
    receipt["relative_path"] = "../outside.json"
    with pytest.raises(ValueError, match="path does not match"):
        verify_ohlcv_archive(receipt, tmp_path)


@pytest.mark.parametrize("column,value", [
    ("close", float("nan")),
    ("volume", float("inf")),
])
def test_archive_rejects_non_finite_values(tmp_path, column, value):
    data = frame()
    data.loc[0, column] = value
    with pytest.raises(ValueError, match="finite numeric"):
        archive_ohlcv_snapshot(data, tmp_path)


def test_archive_rejects_naive_timestamps(tmp_path):
    data = frame()
    data["timestamp"] = data["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        archive_ohlcv_snapshot(data, tmp_path)

def test_archive_rejects_symlinked_hash_directory_escape(tmp_path):
    probe = archive_ohlcv_snapshot(frame(), tmp_path / "probe")
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / probe["sha256"][:2]).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="directory escapes"):
        archive_ohlcv_snapshot(frame(), root)
