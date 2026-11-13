"""Persistent, account-scoped workspace memory primitives.

This module is a storage layer only. Callers MUST pass owner_id obtained from a
verified authentication principal; never trust an owner_id supplied by a client.
No unauthenticated HTTP routes expose this store.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
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

                CREATE TABLE IF NOT EXISTS agent_tasks (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    requester TEXT NOT NULL,
                    target_capability TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed','cancelled')),
                    input_json TEXT NOT NULL,
                    result_json TEXT,
                    deadline_at TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    lease_expires_at TEXT,
                    lease_token TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_agent_tasks_owner_status
                    ON agent_tasks(owner_id, workspace_id, status, created_at);

                CREATE TABLE IF NOT EXISTS agent_messages (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    task_id TEXT REFERENCES agent_tasks(id) ON DELETE CASCADE,
                    sender TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_agent_messages_task
                    ON agent_messages(owner_id, workspace_id, task_id, created_at);

                CREATE TABLE IF NOT EXISTS model_versions (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    artifact_uri TEXT NOT NULL,
                    artifact_sha256 TEXT NOT NULL,
                    input_schema_json TEXT NOT NULL,
                    supported_classes_json TEXT NOT NULL,
                    training_data_ref TEXT NOT NULL,
                    evaluation_report_json TEXT NOT NULL,
                    calibration_status TEXT NOT NULL CHECK(calibration_status IN ('unknown','calibrated','failed')),
                    lifecycle TEXT NOT NULL CHECK(lifecycle IN ('candidate','approved','retired')),
                    created_by TEXT NOT NULL,
                    approved_by TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(model_id, version)
                );
                CREATE INDEX IF NOT EXISTS idx_model_versions_lifecycle
                    ON model_versions(model_id, lifecycle);
                """
            )
            task_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(agent_tasks)").fetchall()
            }
            for column, definition in (
                ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
                ("max_attempts", "INTEGER NOT NULL DEFAULT 3"),
                ("lease_expires_at", "TEXT"),
                ("lease_token", "TEXT"),
            ):
                if column not in task_columns:
                    conn.execute(f"ALTER TABLE agent_tasks ADD COLUMN {column} {definition}")

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


    def create_agent_task(
        self, owner_id: str, workspace_id: str, requester: str,
        target_capability: str, input_data: Any, deadline_at: str | None = None,
        max_attempts: int = 3
    ) -> dict[str, Any]:
        """Queue a typed task; this records work but does not execute an agent."""
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        requester = _required_text(requester, "requester", 80)
        target_capability = _required_text(target_capability, "target_capability", 120)
        encoded = _encode_json(input_data)
        if deadline_at is not None:
            deadline_at = _required_text(deadline_at, "deadline_at", 40)
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 10:
            raise ValueError("max_attempts must be an integer from 1 to 10")
        task_id, now = str(uuid4()), _utc_now()
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            conn.execute(
                "INSERT INTO agent_tasks(id, owner_id, workspace_id, requester, target_capability, "
                "status, input_json, deadline_at, max_attempts, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)",
                (task_id, owner_id, workspace_id, requester, target_capability,
                 encoded, deadline_at, max_attempts, now, now),
            )
            conn.execute(
                "INSERT INTO audit_events(id, owner_id, workspace_id, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid4()), owner_id, workspace_id, "agent.task.queued",
                 _encode_json({"task_id": task_id, "capability": target_capability}), now),
            )
        return {"id": task_id, "owner_id": owner_id, "workspace_id": workspace_id,
                "requester": requester, "target_capability": target_capability,
                "status": "queued", "input": json.loads(encoded), "result": None,
                "deadline_at": deadline_at, "attempt_count": 0, "max_attempts": max_attempts,
                "lease_expires_at": None, "created_at": now, "updated_at": now}

    def list_agent_tasks(
        self, owner_id: str, workspace_id: str, status: str | None = None,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        if status is not None and status not in {"queued", "running", "completed", "failed", "cancelled"}:
            raise ValueError("invalid task status filter")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be an integer from 1 to 500")
        query = ("SELECT * FROM agent_tasks WHERE owner_id = ? AND workspace_id = ?")
        params: list[Any] = [owner_id, workspace_id]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC, id LIMIT ?"
        params.append(limit)
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {"id": row["id"], "owner_id": row["owner_id"], "workspace_id": row["workspace_id"],
             "requester": row["requester"], "target_capability": row["target_capability"],
             "status": row["status"], "input": json.loads(row["input_json"]),
             "result": json.loads(row["result_json"]) if row["result_json"] is not None else None,
             "deadline_at": row["deadline_at"], "attempt_count": row["attempt_count"],
             "max_attempts": row["max_attempts"], "lease_expires_at": row["lease_expires_at"],
             "created_at": row["created_at"], "updated_at": row["updated_at"]}
            for row in rows
        ]

    def claim_next_agent_task(
        self, lease_seconds: int = 120, now: int | None = None
    ) -> dict[str, Any] | None:
        """Atomically lease the oldest queued task and recover expired worker leases."""
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, int) or not 5 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be an integer from 5 to 3600")
        current = datetime.now(timezone.utc) if now is None else datetime.fromtimestamp(now, timezone.utc)
        now_text = current.isoformat(timespec="microseconds").replace("+00:00", "Z")
        lease_text = (current + timedelta(seconds=lease_seconds)).isoformat(
            timespec="microseconds"
        ).replace("+00:00", "Z")
        with sqlite3.connect(self.database_path, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 10000")
            conn.execute("BEGIN IMMEDIATE")

            expired = conn.execute(
                "SELECT id, owner_id, workspace_id, attempt_count, max_attempts "
                "FROM agent_tasks WHERE status = 'running' AND lease_expires_at IS NOT NULL "
                "AND lease_expires_at <= ?",
                (now_text,),
            ).fetchall()
            for row in expired:
                exhausted = row["attempt_count"] >= row["max_attempts"]
                next_status = "failed" if exhausted else "queued"
                error = {"error": "worker lease expired", "attempt_count": row["attempt_count"]}
                conn.execute(
                    "UPDATE agent_tasks SET status = ?, lease_expires_at = NULL, lease_token = NULL, "
                    "result_json = CASE WHEN ? = 1 THEN ? ELSE result_json END, updated_at = ? "
                    "WHERE id = ? AND status = 'running'",
                    (next_status, int(exhausted), _encode_json(error), now_text, row["id"]),
                )
                conn.execute(
                    "INSERT INTO audit_events(id, owner_id, workspace_id, event_type, payload_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (str(uuid4()), row["owner_id"], row["workspace_id"],
                     "agent.task.lease_expired", _encode_json({
                         "task_id": row["id"], "requeued": not exhausted
                     }), now_text),
                )

            row = conn.execute(
                "SELECT * FROM agent_tasks WHERE status = 'queued' "
                "ORDER BY created_at, id LIMIT 1"
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            lease_token = str(uuid4())
            conn.execute(
                "UPDATE agent_tasks SET status = 'running', attempt_count = attempt_count + 1, "
                "lease_expires_at = ?, lease_token = ?, updated_at = ? "
                "WHERE id = ? AND status = 'queued'",
                (lease_text, lease_token, now_text, row["id"]),
            )
            conn.execute(
                "INSERT INTO audit_events(id, owner_id, workspace_id, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid4()), row["owner_id"], row["workspace_id"], "agent.task.claimed",
                 _encode_json({"task_id": row["id"], "attempt": row["attempt_count"] + 1}), now_text),
            )
            claimed = conn.execute(
                "SELECT * FROM agent_tasks WHERE id = ?", (row["id"],)
            ).fetchone()
            conn.commit()
        return {
            "id": claimed["id"], "owner_id": claimed["owner_id"],
            "workspace_id": claimed["workspace_id"], "requester": claimed["requester"],
            "target_capability": claimed["target_capability"], "status": claimed["status"],
            "input": json.loads(claimed["input_json"]),
            "result": json.loads(claimed["result_json"]) if claimed["result_json"] else None,
            "deadline_at": claimed["deadline_at"], "attempt_count": claimed["attempt_count"],
            "max_attempts": claimed["max_attempts"], "lease_expires_at": claimed["lease_expires_at"],
            "lease_token": lease_token,
            "created_at": claimed["created_at"], "updated_at": claimed["updated_at"],
        }


    def renew_agent_task_lease(
        self, task_id: str, lease_token: str, lease_seconds: int = 120
    ) -> str:
        task_id = _required_text(task_id, "task_id")
        lease_token = _required_text(lease_token, "lease_token")
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, int) or not 5 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be an integer from 5 to 3600")
        current = datetime.now(timezone.utc)
        now_text = current.isoformat(timespec="microseconds").replace("+00:00", "Z")
        lease_text = (current + timedelta(seconds=lease_seconds)).isoformat(
            timespec="microseconds"
        ).replace("+00:00", "Z")
        with self._connection() as conn:
            row = conn.execute(
                "SELECT status, lease_token, lease_expires_at FROM agent_tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                raise LookupError("task not found")
            if (row["status"] != "running" or row["lease_token"] != lease_token
                    or row["lease_expires_at"] is None or row["lease_expires_at"] <= now_text):
                raise ValueError("task lease is invalid or expired")
            conn.execute(
                "UPDATE agent_tasks SET lease_expires_at = ?, updated_at = ? "
                "WHERE id = ? AND status = 'running' AND lease_token = ?",
                (lease_text, now_text, task_id, lease_token),
            )
        return lease_text

    def complete_claimed_agent_task(
        self, task_id: str, lease_token: str, result: Any
    ) -> dict[str, Any]:
        task_id = _required_text(task_id, "task_id")
        lease_token = _required_text(lease_token, "lease_token")
        encoded = _encode_json(result)
        now = _utc_now()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT owner_id, workspace_id, status, lease_token, lease_expires_at "
                "FROM agent_tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                raise LookupError("task not found")
            if (row["status"] != "running" or row["lease_token"] != lease_token
                    or row["lease_expires_at"] is None or row["lease_expires_at"] <= now):
                raise ValueError("task lease is invalid or expired")
            conn.execute(
                "UPDATE agent_tasks SET status = 'completed', result_json = ?, "
                "lease_expires_at = NULL, lease_token = NULL, updated_at = ? "
                "WHERE id = ? AND status = 'running' AND lease_token = ?",
                (encoded, now, task_id, lease_token),
            )
            conn.execute(
                "INSERT INTO audit_events(id, owner_id, workspace_id, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, 'agent.task.completed', ?, ?)",
                (str(uuid4()), row["owner_id"], row["workspace_id"],
                 _encode_json({"task_id": task_id}), now),
            )
        return self.get_agent_task(row["owner_id"], row["workspace_id"], task_id)  # type: ignore[return-value]

    def fail_claimed_agent_task(
        self, task_id: str, lease_token: str, error: Any, *, retryable: bool = True
    ) -> dict[str, Any]:
        task_id = _required_text(task_id, "task_id")
        lease_token = _required_text(lease_token, "lease_token")
        if not isinstance(retryable, bool):
            raise ValueError("retryable must be a boolean")
        encoded = _encode_json(error)
        now = _utc_now()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT owner_id, workspace_id, status, lease_token, lease_expires_at, "
                "attempt_count, max_attempts FROM agent_tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                raise LookupError("task not found")
            if (row["status"] != "running" or row["lease_token"] != lease_token
                    or row["lease_expires_at"] is None or row["lease_expires_at"] <= now):
                raise ValueError("task lease is invalid or expired")
            should_retry = retryable and row["attempt_count"] < row["max_attempts"]
            next_status = "queued" if should_retry else "failed"
            conn.execute(
                "UPDATE agent_tasks SET status = ?, result_json = ?, "
                "lease_expires_at = NULL, lease_token = NULL, updated_at = ? "
                "WHERE id = ? AND status = 'running' AND lease_token = ?",
                (next_status, encoded, now, task_id, lease_token),
            )
            conn.execute(
                "INSERT INTO audit_events(id, owner_id, workspace_id, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid4()), row["owner_id"], row["workspace_id"],
                 "agent.task.requeued" if should_retry else "agent.task.failed",
                 _encode_json({"task_id": task_id, "retryable": retryable,
                               "attempt_count": row["attempt_count"]}), now),
            )
        return self.get_agent_task(row["owner_id"], row["workspace_id"], task_id)  # type: ignore[return-value]

    def get_agent_task(self, owner_id: str, workspace_id: str, task_id: str) -> dict[str, Any] | None:
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        task_id = _required_text(task_id, "task_id")
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM agent_tasks WHERE id = ? AND owner_id = ? AND workspace_id = ?",
                (task_id, owner_id, workspace_id),
            ).fetchone()
        if row is None:
            return None
        return {"id": row["id"], "owner_id": row["owner_id"], "workspace_id": row["workspace_id"],
                "requester": row["requester"], "target_capability": row["target_capability"],
                "status": row["status"], "input": json.loads(row["input_json"]),
                "result": json.loads(row["result_json"]) if row["result_json"] is not None else None,
                "deadline_at": row["deadline_at"], "attempt_count": row["attempt_count"],
                "max_attempts": row["max_attempts"], "lease_expires_at": row["lease_expires_at"],
                "created_at": row["created_at"], "updated_at": row["updated_at"]}

    def update_agent_task(
        self, owner_id: str, workspace_id: str, task_id: str,
        status: str, result: Any = None
    ) -> dict[str, Any]:
        """Apply an allowed state transition; terminal tasks cannot be reopened."""
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        task_id = _required_text(task_id, "task_id")
        status = _required_text(status, "status", 20)
        allowed = {
            "queued": {"running", "cancelled", "failed"},
            "running": {"completed", "failed", "cancelled"},
            "completed": set(), "failed": set(), "cancelled": set(),
        }
        if status not in allowed:
            raise ValueError("invalid task status")
        encoded_result = None if result is None else _encode_json(result)
        now = _utc_now()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT status FROM agent_tasks WHERE id = ? AND owner_id = ? AND workspace_id = ?",
                (task_id, owner_id, workspace_id),
            ).fetchone()
            if row is None:
                raise LookupError("task not found")
            if status not in allowed[row["status"]]:
                raise ValueError(f"invalid task transition: {row['status']} -> {status}")
            if status == "completed" and encoded_result is None:
                raise ValueError("completed tasks require a result")
            conn.execute(
                "UPDATE agent_tasks SET status = ?, result_json = COALESCE(?, result_json), "
                "lease_expires_at = NULL, lease_token = NULL, updated_at = ? "
                "WHERE id = ? AND owner_id = ? AND workspace_id = ?",
                (status, encoded_result, now, task_id, owner_id, workspace_id),
            )
            conn.execute(
                "INSERT INTO audit_events(id, owner_id, workspace_id, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid4()), owner_id, workspace_id, f"agent.task.{status}",
                 _encode_json({"task_id": task_id}), now),
            )
        return self.get_agent_task(owner_id, workspace_id, task_id)  # type: ignore[return-value]

    def send_agent_message(
        self, owner_id: str, workspace_id: str, sender: str, recipient: str,
        message_type: str, payload: Any, task_id: str | None = None
    ) -> dict[str, Any]:
        """Persist an agent message as data; messages are never executed as code."""
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        sender = _required_text(sender, "sender", 80)
        recipient = _required_text(recipient, "recipient", 80)
        message_type = _required_text(message_type, "message_type", 80)
        encoded = _encode_json(payload)
        message_id, now = str(uuid4()), _utc_now()
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            if task_id is not None:
                task_id = _required_text(task_id, "task_id")
                exists = conn.execute(
                    "SELECT 1 FROM agent_tasks WHERE id = ? AND owner_id = ? AND workspace_id = ?",
                    (task_id, owner_id, workspace_id),
                ).fetchone()
                if exists is None:
                    raise LookupError("task not found")
            conn.execute(
                "INSERT INTO agent_messages(id, owner_id, workspace_id, task_id, sender, recipient, "
                "message_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (message_id, owner_id, workspace_id, task_id, sender, recipient,
                 message_type, encoded, now),
            )
            conn.execute(
                "INSERT INTO audit_events(id, owner_id, workspace_id, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid4()), owner_id, workspace_id, "agent.message.created",
                 _encode_json({"message_id": message_id, "task_id": task_id,
                               "sender": sender, "recipient": recipient}), now),
            )
        return {"id": message_id, "owner_id": owner_id, "workspace_id": workspace_id,
                "task_id": task_id, "sender": sender, "recipient": recipient,
                "message_type": message_type, "payload": json.loads(encoded), "created_at": now}

    def list_agent_messages(
        self, owner_id: str, workspace_id: str, task_id: str | None = None,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be an integer from 1 to 500")
        query = ("SELECT * FROM agent_messages WHERE owner_id = ? AND workspace_id = ?")
        params: list[Any] = [owner_id, workspace_id]
        if task_id is not None:
            query += " AND task_id = ?"
            params.append(_required_text(task_id, "task_id"))
        query += " ORDER BY created_at, id LIMIT ?"
        params.append(limit)
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {"id": row["id"], "owner_id": row["owner_id"], "workspace_id": row["workspace_id"],
             "task_id": row["task_id"], "sender": row["sender"], "recipient": row["recipient"],
             "message_type": row["message_type"], "payload": json.loads(row["payload_json"]),
             "created_at": row["created_at"]}
            for row in rows
        ]


    def register_model_version(
        self, model_id: str, version: str, artifact_uri: str, artifact_sha256: str,
        input_schema: Any, supported_classes: list[str], training_data_ref: str,
        evaluation_report: Any, calibration_status: str, created_by: str
    ) -> dict[str, Any]:
        """Register immutable model metadata as a candidate; never promotes it."""
        import re
        model_id = _required_text(model_id, "model_id", 120)
        version = _required_text(version, "version", 80)
        artifact_uri = _required_text(artifact_uri, "artifact_uri", 500)
        training_data_ref = _required_text(training_data_ref, "training_data_ref", 300)
        created_by = _required_text(created_by, "created_by", 128)
        if not isinstance(artifact_sha256, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", artifact_sha256):
            raise ValueError("artifact_sha256 must be a 64-character SHA-256 hex digest")
        if not isinstance(supported_classes, list) or not supported_classes:
            raise ValueError("supported_classes must be a non-empty list")
        normalized_classes = sorted({_required_text(item, "supported_class", 80) for item in supported_classes})
        if calibration_status not in {"unknown", "calibrated", "failed"}:
            raise ValueError("invalid calibration_status")
        if not isinstance(input_schema, dict):
            raise ValueError("input_schema must be a JSON object")
        if not isinstance(evaluation_report, dict):
            raise ValueError("evaluation_report must be a JSON object")
        schema_json, classes_json = _encode_json(input_schema), _encode_json(normalized_classes)
        evaluation_json = _encode_json(evaluation_report)
        model_version_id, now = str(uuid4()), _utc_now()
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO model_versions(id, model_id, version, artifact_uri, artifact_sha256, "
                "input_schema_json, supported_classes_json, training_data_ref, evaluation_report_json, "
                "calibration_status, lifecycle, created_by, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate', ?, ?, ?)",
                (model_version_id, model_id, version, artifact_uri, artifact_sha256.lower(),
                 schema_json, classes_json, training_data_ref, evaluation_json,
                 calibration_status, created_by, now, now),
            )
        return self.get_model_version(model_id, version)

    def get_model_version(self, model_id: str, version: str) -> dict[str, Any] | None:
        model_id = _required_text(model_id, "model_id", 120)
        version = _required_text(version, "version", 80)
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE model_id = ? AND version = ?",
                (model_id, version),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"], "model_id": row["model_id"], "version": row["version"],
            "artifact_uri": row["artifact_uri"], "artifact_sha256": row["artifact_sha256"],
            "input_schema": json.loads(row["input_schema_json"]),
            "supported_classes": json.loads(row["supported_classes_json"]),
            "training_data_ref": row["training_data_ref"],
            "evaluation_report": json.loads(row["evaluation_report_json"]),
            "calibration_status": row["calibration_status"], "lifecycle": row["lifecycle"],
            "created_by": row["created_by"], "approved_by": row["approved_by"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    def approve_model_version(self, model_id: str, version: str, approver: str) -> dict[str, Any]:
        """Approve only independently evaluated and calibrated candidates."""
        model_id = _required_text(model_id, "model_id", 120)
        version = _required_text(version, "version", 80)
        approver = _required_text(approver, "approver", 128)
        now = _utc_now()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE model_id = ? AND version = ?",
                (model_id, version),
            ).fetchone()
            if row is None:
                raise LookupError("model version not found")
            if row["lifecycle"] != "candidate":
                raise ValueError("only candidate model versions can be approved")
            report = json.loads(row["evaluation_report_json"])
            if report.get("passed") is not True:
                raise ValueError("model evaluation must explicitly pass before approval")
            if row["calibration_status"] != "calibrated":
                raise ValueError("model must be calibrated before approval")
            if approver == row["created_by"]:
                raise ValueError("model creator cannot approve their own model version")
            conn.execute(
                "UPDATE model_versions SET lifecycle = 'approved', approved_by = ?, updated_at = ? "
                "WHERE model_id = ? AND version = ? AND lifecycle = 'candidate'",
                (approver, now, model_id, version),
            )
        return self.get_model_version(model_id, version)

    def retire_model_version(self, model_id: str, version: str) -> dict[str, Any]:
        model_id = _required_text(model_id, "model_id", 120)
        version = _required_text(version, "version", 80)
        now = _utc_now()
        with self._connection() as conn:
            cursor = conn.execute(
                "UPDATE model_versions SET lifecycle = 'retired', updated_at = ? "
                "WHERE model_id = ? AND version = ? AND lifecycle = 'approved'",
                (now, model_id, version),
            )
            if cursor.rowcount != 1:
                raise ValueError("only an approved model version can be retired")
        return self.get_model_version(model_id, version)

    def list_model_versions(self, lifecycle: str | None = None) -> list[dict[str, Any]]:
        if lifecycle is not None and lifecycle not in {"candidate", "approved", "retired"}:
            raise ValueError("invalid lifecycle filter")
        query = "SELECT model_id, version FROM model_versions"
        params: tuple[Any, ...] = ()
        if lifecycle is not None:
            query += " WHERE lifecycle = ?"
            params = (lifecycle,)
        query += " ORDER BY model_id, version"
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self.get_model_version(row["model_id"], row["version"]) for row in rows]
