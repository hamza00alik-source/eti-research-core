import base64
import sqlite3

from eti.backup import (decrypt_snapshot, encrypt_snapshot, restore_snapshot, snapshot_sqlite,
                        validate_backup_configuration)


def test_encrypted_backup_round_trip_and_integrity(tmp_path):
    source = tmp_path / "source.sqlite3"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO sample(value) VALUES ('private-data')")
    key = b"e" * 32
    snapshot = snapshot_sqlite(source)
    encrypted = encrypt_snapshot(snapshot, key)
    assert b"private-data" not in encrypted
    assert decrypt_snapshot(encrypted, key) == snapshot

    destination = tmp_path / "restored.sqlite3"
    restore_snapshot(encrypted, destination, key)
    with sqlite3.connect(destination) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT value FROM sample").fetchone()[0] == "private-data"


def test_backup_authentication_rejects_tampering():
    key = b"k" * 32
    encrypted = bytearray(encrypt_snapshot(b"sqlite snapshot", key))
    encrypted[-1] ^= 1
    try:
        decrypt_snapshot(bytes(encrypted), key)
    except Exception:
        pass
    else:
        raise AssertionError("tampered backup must not decrypt")


def test_backup_configuration_fails_closed_when_secrets_are_missing(monkeypatch):
    import pytest

    for name in (
        "ETI_BACKUP_S3_ENDPOINT", "ETI_BACKUP_S3_BUCKET",
        "ETI_BACKUP_S3_ACCESS_KEY_ID", "ETI_BACKUP_S3_SECRET_ACCESS_KEY",
        "ETI_BACKUP_ENCRYPTION_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="missing required backup settings"):
        validate_backup_configuration()


def test_backup_configuration_rejects_invalid_retention(monkeypatch):
    import base64
    import pytest

    monkeypatch.setenv("ETI_BACKUP_S3_ENDPOINT", "https://objects.example.invalid")
    monkeypatch.setenv("ETI_BACKUP_S3_BUCKET", "eti-backups")
    monkeypatch.setenv("ETI_BACKUP_S3_ACCESS_KEY_ID", "test-access")
    monkeypatch.setenv("ETI_BACKUP_S3_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("ETI_BACKUP_ENCRYPTION_KEY", base64.urlsafe_b64encode(b"x" * 32).decode())
    monkeypatch.setenv("ETI_BACKUP_RETENTION_DAYS", "2")
    with pytest.raises(ValueError, match="RETENTION"):
        validate_backup_configuration()
