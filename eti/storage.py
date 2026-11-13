"""Persistent, account-scoped workspace memory primitives.

This module is a storage layer only. Callers MUST pass owner_id obtained from a
verified authentication principal; never trust an owner_id supplied by a client.
No unauthenticated HTTP routes expose this store.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterator
from uuid import uuid4


MAX_JSON_BYTES = 65_536


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _required_text(value: str, field: str, max_length: int = 128) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    value = value.strip()
    if not value or len(value) > max_length:
        raise ValueError(f"{field} must contain 1 to {max_length} characters")
    return value


def _encode_json(value: Any) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be valid JSON data with finite numbers") from exc
    if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError(f"JSON value exceeds {MAX_JSON_BYTES} bytes")
    return encoded


class MemoryStore:
    """SQLite-backed workspace and key/value memory store.

    The default database path can be overridden with ETI_DATABASE_PATH.
    SQLite is suitable for local development and single-instance deployments;
    production multi-instance deployments should migrate this interface to
    PostgreSQL before enabling shared writes across replicas.
    """

    def __init__(self, database_path: str | os.PathLike[str] | None = None):
        configured = database_path or os.environ.get("ETI_DATABASE_PATH", "data/eti.sqlite3")
        self.database_path = str(configured)
        if self.database_path == ":memory:":
            raise ValueError("Use a file-backed database so connections share persistent state")
        Path(self.database_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.database_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(owner_id, name)
                );
                CREATE INDEX IF NOT EXISTS idx_workspaces_owner
                    ON workspaces(owner_id);

                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    owner_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    item_key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(workspace_id, namespace, item_key)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_owner_workspace
                    ON memory_items(owner_id, workspace_id, namespace);

                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_owner_time
                    ON audit_events(owner_id, created_at);
                """
            )

    def create_workspace(self, owner_id: str, name: str) -> dict[str, str]:
        owner_id = _required_text(owner_id, "owner_id")
        name = _required_text(name, "name", 80)
        workspace_id, created_at = str(uuid4()), _utc_now()
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO workspaces(id, owner_id, name, created_at) VALUES (?, ?, ?, ?)",
                (workspace_id, owner_id, name, created_at),
            )
        return {"id": workspace_id, "owner_id": owner_id, "name": name, "created_at": created_at}

    def list_workspaces(self, owner_id: str) -> list[dict[str, str]]:
        owner_id = _required_text(owner_id, "owner_id")
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT id, owner_id, name, created_at FROM workspaces "
                "WHERE owner_id = ? ORDER BY created_at, id",
                (owner_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _owns_workspace(conn: sqlite3.Connection, owner_id: str, workspace_id: str) -> bool:
        return conn.execute(
            "SELECT 1 FROM workspaces WHERE id = ? AND owner_id = ?",
            (workspace_id, owner_id),
        ).fetchone() is not None

    def put_memory(
        self, owner_id: str, workspace_id: str, namespace: str, key: str, value: Any
    ) -> dict[str, Any]:
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        namespace = _required_text(namespace, "namespace", 80)
        key = _required_text(key, "key", 160)
        encoded = _encode_json(value)
        now = _utc_now()
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            conn.execute(
                """
                INSERT INTO memory_items
                    (id, workspace_id, owner_id, namespace, item_key, value_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id, namespace, item_key) DO UPDATE SET
                    value_json = excluded.value_json, updated_at = excluded.updated_at
                """,
                (str(uuid4()), workspace_id, owner_id, namespace, key, encoded, now, now),
            )
            row = conn.execute(
                "SELECT id, namespace, item_key, value_json, created_at, updated_at "
                "FROM memory_items WHERE workspace_id = ? AND owner_id = ? "
                "AND namespace = ? AND item_key = ?",
                (workspace_id, owner_id, namespace, key),
            ).fetchone()
        return {
            "id": row["id"], "namespace": row["namespace"], "key": row["item_key"],
            "value": json.loads(row["value_json"]), "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_memory(
        self, owner_id: str, workspace_id: str, namespace: str, key: str
    ) -> dict[str, Any] | None:
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        namespace = _required_text(namespace, "namespace", 80)
        key = _required_text(key, "key", 160)
        with self._connection() as conn:
            row = conn.execute(
                "SELECT id, namespace, item_key, value_json, created_at, updated_at "
                "FROM memory_items WHERE owner_id = ? AND workspace_id = ? "
                "AND namespace = ? AND item_key = ?",
                (owner_id, workspace_id, namespace, key),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"], "namespace": row["namespace"], "key": row["item_key"],
            "value": json.loads(row["value_json"]), "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_memory(
        self, owner_id: str, workspace_id: str, namespace: str | None = None
    ) -> list[dict[str, Any]]:
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        if namespace is not None:
            namespace = _required_text(namespace, "namespace", 80)
        query = (
            "SELECT id, namespace, item_key, value_json, created_at, updated_at "
            "FROM memory_items WHERE owner_id = ? AND workspace_id = ?"
        )
        params: list[Any] = [owner_id, workspace_id]
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        query += " ORDER BY namespace, item_key"
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {"id": row["id"], "namespace": row["namespace"], "key": row["item_key"],
             "value": json.loads(row["value_json"]), "created_at": row["created_at"],
             "updated_at": row["updated_at"]}
            for row in rows
        ]

    def delete_memory(self, owner_id: str, workspace_id: str, namespace: str, key: str) -> bool:
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        namespace = _required_text(namespace, "namespace", 80)
        key = _required_text(key, "key", 160)
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM memory_items WHERE owner_id = ? AND workspace_id = ? "
                "AND namespace = ? AND item_key = ?",
                (owner_id, workspace_id, namespace, key),
            )
        return cursor.rowcount == 1

    def append_audit_event(
        self, owner_id: str, event_type: str, payload: Any, workspace_id: str | None = None
    ) -> dict[str, Any]:
        owner_id = _required_text(owner_id, "owner_id")
        event_type = _required_text(event_type, "event_type", 120)
        encoded = _encode_json(payload)
        event_id, created_at = str(uuid4()), _utc_now()
        with self._connection() as conn:
            if workspace_id is not None:
                workspace_id = _required_text(workspace_id, "workspace_id")
                if not self._owns_workspace(conn, owner_id, workspace_id):
                    raise LookupError("workspace not found")
            conn.execute(
                "INSERT INTO audit_events(id, owner_id, workspace_id, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, owner_id, workspace_id, event_type, encoded, created_at),
            )
        return {"id": event_id, "owner_id": owner_id, "workspace_id": workspace_id,
                "event_type": event_type, "payload": json.loads(encoded), "created_at": created_at}

    def list_audit_events(self, owner_id: str, limit: int = 100) -> list[dict[str, Any]]:
        owner_id = _required_text(owner_id, "owner_id")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be an integer from 1 to 500")
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT id, workspace_id, event_type, payload_json, created_at "
                "FROM audit_events WHERE owner_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                (owner_id, limit),
            ).fetchall()
        return [
            {"id": row["id"], "workspace_id": row["workspace_id"],
             "event_type": row["event_type"], "payload": json.loads(row["payload_json"]),
             "created_at": row["created_at"]}
            for row in rows
        ]
