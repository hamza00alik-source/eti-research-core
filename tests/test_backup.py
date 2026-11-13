import base64
import os
import stat
import sqlite3

from eti.backup import decrypt_snapshot, encrypt_snapshot, restore_snapshot, snapshot_sqlite


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
    if os.name == "posix":
        assert stat.S_IMODE(destination.stat().st_mode) == 0o600
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
