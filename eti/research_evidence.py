"""Content-addressed, tamper-evident archival for research OHLCV inputs.

This local archive preserves the exact validated values used by a research run.
It detects accidental or malicious content changes on verification, but it is
not WORM storage and does not authenticate the external data provider.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd

_SCHEMA = "eti.ohlcv_archive.v1"
_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _source_snapshot_sha256(rows: list[list[Any]]) -> str:
    payload = {
        "schema_version": "eti.ohlcv_snapshot.v1",
        "columns": _COLUMNS,
        "rows": rows,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_payload(frame: pd.DataFrame) -> bytes:
    if not isinstance(frame, pd.DataFrame) or any(c not in frame.columns for c in _COLUMNS):
        raise ValueError("OHLCV archive requires a DataFrame with timestamp and OHLCV columns")
    if frame.empty:
        raise ValueError("OHLCV archive cannot store an empty snapshot")
    rows = []
    for index, values in enumerate(frame[_COLUMNS].itertuples(index=False, name=None)):
        timestamp, *numbers = values
        stamp = pd.Timestamp(timestamp)
        if stamp.tzinfo is None:
            raise ValueError(f"OHLCV row {index} timestamp must be timezone-aware")
        encoded_numbers = []
        for name, value in zip(_COLUMNS[1:], numbers):
            if isinstance(value, bool):
                raise ValueError(f"OHLCV row {index} {name} must be finite numeric data")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"OHLCV row {index} {name} must be finite numeric data")
            encoded_numbers.append(number.hex())
        rows.append([stamp.tz_convert("UTC").isoformat(), *encoded_numbers])
    payload = {
        "schema_version": _SCHEMA,
        "columns": _COLUMNS,
        "row_count": len(rows),
        "rows": rows,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def archive_ohlcv_snapshot(frame: pd.DataFrame, archive_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Persist exact OHLCV values at a SHA-256-derived path without overwriting."""
    payload = _canonical_payload(frame)
    digest = hashlib.sha256(payload).hexdigest()
    relative = Path(digest[:2]) / f"{digest}.json"
    root = Path(archive_root).expanduser().resolve()
    directory = root / relative.parent
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / relative.name

    descriptor, temporary_name = tempfile.mkstemp(prefix=".eti-", dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink():
                raise RuntimeError("content-addressed archive object must not be a symbolic link")
            existing = destination.read_bytes()
            if existing != payload:
                raise RuntimeError("content-addressed archive collision or tampered existing object")
        return {
            "schema_version": "eti.ohlcv_archive_receipt.v1",
            "sha256": digest,
            "source_snapshot_sha256": _source_snapshot_sha256(json.loads(payload)["rows"]),
            "relative_path": relative.as_posix(),
            "byte_size": len(payload),
            "row_count": len(frame),
            "first_timestamp": pd.Timestamp(frame["timestamp"].iloc[0]).tz_convert("UTC").isoformat()
                if len(frame) else None,
            "last_timestamp": pd.Timestamp(frame["timestamp"].iloc[-1]).tz_convert("UTC").isoformat()
                if len(frame) else None,
            "storage": "local_content_addressed",
            "immutability": "tamper_evident_not_worm",
        }
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def verify_ohlcv_archive(receipt: dict[str, Any], archive_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Verify receipt shape, canonical path, byte digest, schema and row count."""
    if not isinstance(receipt, dict) or receipt.get("schema_version") != "eti.ohlcv_archive_receipt.v1":
        raise ValueError("unsupported OHLCV archive receipt")
    digest = receipt.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("receipt sha256 must be a lowercase SHA-256 digest")
    expected_relative = (Path(digest[:2]) / f"{digest}.json").as_posix()
    if receipt.get("relative_path") != expected_relative:
        raise ValueError("receipt path does not match its content digest")
    root = Path(archive_root).expanduser().resolve()
    path = (root / expected_relative).resolve()
    if root not in path.parents:
        raise ValueError("archive object path escapes archive root")
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != digest:
        raise ValueError("archived OHLCV content digest mismatch")
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("archived OHLCV object is not valid JSON") from exc
    if decoded.get("schema_version") != _SCHEMA or decoded.get("columns") != _COLUMNS:
        raise ValueError("archived OHLCV schema mismatch")
    if decoded.get("row_count") != receipt.get("row_count") or len(decoded.get("rows", [])) != receipt.get("row_count"):
        raise ValueError("archived OHLCV row count mismatch")
    source_digest = _source_snapshot_sha256(decoded["rows"])
    if source_digest != receipt.get("source_snapshot_sha256"):
        raise ValueError("archived OHLCV snapshot fingerprint mismatch")
    if len(payload) != receipt.get("byte_size"):
        raise ValueError("archived OHLCV byte size mismatch")
    return {"verified": True, "sha256": digest, "row_count": decoded["row_count"], "byte_size": len(payload)}
