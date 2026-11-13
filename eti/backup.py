"""Encrypted SQLite snapshots and S3-compatible off-site backup/restore."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any
from uuid import uuid4

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


MAGIC = b"ETIENC1"


def _key_from_env(value: str | None = None) -> bytes:
    raw = value or os.environ.get("ETI_BACKUP_ENCRYPTION_KEY", "")
    if not raw:
        raise RuntimeError("ETI_BACKUP_ENCRYPTION_KEY is required")
    try:
        key = base64.urlsafe_b64decode(raw.encode("ascii"))
    except Exception as exc:
        raise ValueError("backup encryption key must be URL-safe base64") from exc
    if len(key) != 32:
        raise ValueError("backup encryption key must decode to exactly 32 bytes")
    return key


def validate_backup_configuration() -> None:
    required = (
        "ETI_BACKUP_S3_ENDPOINT",
        "ETI_BACKUP_S3_BUCKET",
        "ETI_BACKUP_S3_ACCESS_KEY_ID",
        "ETI_BACKUP_S3_SECRET_ACCESS_KEY",
        "ETI_BACKUP_ENCRYPTION_KEY",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError("missing required backup settings: " + ", ".join(missing))
    _key_from_env()
    retention_days = int(os.environ.get("ETI_BACKUP_RETENTION_DAYS", "30"))
    if not 7 <= retention_days <= 365:
        raise ValueError("ETI_BACKUP_RETENTION_DAYS must be between 7 and 365")
    if os.environ.get("ETI_BACKUP_S3_ADDRESSING_STYLE", "path") not in {"path", "virtual"}:
        raise ValueError("ETI_BACKUP_S3_ADDRESSING_STYLE must be path or virtual")


def snapshot_sqlite(database_path: str | os.PathLike[str]) -> bytes:
    """Take a transactionally consistent SQLite snapshot and verify its integrity."""
    source_path = Path(database_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"database does not exist: {source_path}")
    with tempfile.TemporaryDirectory(prefix="eti-backup-") as directory:
        snapshot_path = Path(directory) / "snapshot.sqlite3"
        with sqlite3.connect(str(source_path), timeout=30) as source:
            with sqlite3.connect(str(snapshot_path)) as target:
                source.backup(target)
                result = target.execute("PRAGMA integrity_check").fetchone()
                if not result or result[0] != "ok":
                    raise RuntimeError("SQLite backup failed integrity_check")
        return snapshot_path.read_bytes()


def encrypt_snapshot(snapshot: bytes, key: bytes | None = None) -> bytes:
    if not isinstance(snapshot, bytes) or not snapshot:
        raise ValueError("snapshot must be non-empty bytes")
    nonce = os.urandom(12)
    return MAGIC + nonce + AESGCM(key or _key_from_env()).encrypt(nonce, snapshot, MAGIC)


def decrypt_snapshot(encrypted: bytes, key: bytes | None = None) -> bytes:
    if not encrypted.startswith(MAGIC) or len(encrypted) <= len(MAGIC) + 12:
        raise ValueError("invalid ETI encrypted backup format")
    nonce = encrypted[len(MAGIC):len(MAGIC) + 12]
    return AESGCM(key or _key_from_env()).decrypt(
        nonce, encrypted[len(MAGIC) + 12:], MAGIC
    )


def restore_snapshot(encrypted: bytes, destination: str | os.PathLike[str],
                     key: bytes | None = None) -> None:
    """Verify a decrypted snapshot before atomically replacing the destination DB."""
    payload = decrypt_snapshot(encrypted, key)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.restore")
    try:
        temporary.write_bytes(payload)
        with sqlite3.connect(str(temporary)) as connection:
            result = connection.execute("PRAGMA integrity_check").fetchone()
            if not result or result[0] != "ok":
                raise RuntimeError("restored SQLite database failed integrity_check")
        Path(str(target) + "-wal").unlink(missing_ok=True)
        Path(str(target) + "-shm").unlink(missing_ok=True)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _s3_client():
    import boto3
    from botocore.config import Config

    endpoint = os.environ["ETI_BACKUP_S3_ENDPOINT"]
    access_key = os.environ["ETI_BACKUP_S3_ACCESS_KEY_ID"]
    secret_key = os.environ["ETI_BACKUP_S3_SECRET_ACCESS_KEY"]
    region = os.environ.get("ETI_BACKUP_S3_REGION", "us-east-1")
    style = os.environ.get("ETI_BACKUP_S3_ADDRESSING_STYLE", "path")
    if style == "virtual-host":
        style = "virtual"
    if style not in {"path", "virtual"}:
        raise ValueError("ETI_BACKUP_S3_ADDRESSING_STYLE must be path or virtual-host")
    return boto3.client(
        "s3", endpoint_url=endpoint, region_name=region,
        aws_access_key_id=access_key, aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": style}),
    )


def backup_once() -> str:
    validate_backup_configuration()
    bucket = os.environ["ETI_BACKUP_S3_BUCKET"]
    database = os.environ.get("ETI_DATABASE_PATH", "/app/data/eti.sqlite3")
    prefix = os.environ.get("ETI_BACKUP_S3_PREFIX", "eti/sqlite").strip("/")
    retention_days = int(os.environ.get("ETI_BACKUP_RETENTION_DAYS", "30"))
    if not 7 <= retention_days <= 365:
        raise ValueError("ETI_BACKUP_RETENTION_DAYS must be between 7 and 365")
    encrypted = encrypt_snapshot(snapshot_sqlite(database))
    now = datetime.now(timezone.utc)
    key = f"{prefix}/{now:%Y/%m/%d}/eti-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}.sqlite3.enc"
    client = _s3_client()
    client.put_object(
        Bucket=bucket, Key=key, Body=encrypted,
        ContentType="application/octet-stream",
        Metadata={"format": "eti-encrypted-sqlite-v1"},
    )
    cutoff = now - timedelta(days=retention_days)
    continuation = None
    while True:
        request: dict[str, Any] = {"Bucket": bucket, "Prefix": f"{prefix}/"}
        if continuation:
            request["ContinuationToken"] = continuation
        page = client.list_objects_v2(**request)
        stale = [
            {"Key": item["Key"]} for item in page.get("Contents", [])
            if item["Key"] != key and item["LastModified"] < cutoff
        ]
        for offset in range(0, len(stale), 1000):
            client.delete_objects(Bucket=bucket, Delete={"Objects": stale[offset:offset + 1000]})
        if not page.get("IsTruncated"):
            break
        continuation = page.get("NextContinuationToken")
    return key


def restore_from_s3(object_key: str, destination: str) -> None:
    bucket = os.environ["ETI_BACKUP_S3_BUCKET"]
    response = _s3_client().get_object(Bucket=bucket, Key=object_key)
    restore_snapshot(response["Body"].read(), destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="ETI encrypted backup and restore")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("backup", help="create and upload a verified encrypted backup")
    restore = sub.add_parser("restore", help="download, verify and restore an encrypted backup")
    restore.add_argument("--key", required=True, help="exact S3 object key")
    restore.add_argument("--destination", required=True, help="database file to replace")
    args = parser.parse_args()
    if args.command == "backup":
        print(backup_once())
    else:
        restore_from_s3(args.key, args.destination)
        print("restore complete; SQLite integrity_check passed")


if __name__ == "__main__":
    main()
