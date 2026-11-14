"""Persistent, account-scoped workspace memory primitives.

This module is a storage layer only. Callers MUST pass owner_id obtained from a
verified authentication principal; never trust an owner_id supplied by a client.
No unauthenticated HTTP routes expose this store.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import hashlib
import math
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

                CREATE TABLE IF NOT EXISTS solana_research_events (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    event_key TEXT NOT NULL,
                    source TEXT NOT NULL,
                    event_type TEXT NOT NULL CHECK(event_type IN ('new_token','migration')),
                    mint TEXT NOT NULL,
                    source_timestamp TEXT,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    UNIQUE(owner_id, workspace_id, event_key)
                );
                CREATE INDEX IF NOT EXISTS idx_solana_events_owner_time
                    ON solana_research_events(owner_id, workspace_id, received_at);
                CREATE TABLE IF NOT EXISTS solana_research_snapshots (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    task_id TEXT NOT NULL,
                    capability TEXT NOT NULL CHECK(capability IN ('solana.market.scan','solana.token.audit')),
                    mint TEXT NOT NULL,
                    pool_address TEXT,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    UNIQUE(owner_id, workspace_id, task_id)
                );
                CREATE INDEX IF NOT EXISTS idx_solana_snapshots_owner_mint_time
                    ON solana_research_snapshots(owner_id, workspace_id, mint, fetched_at);
                CREATE TABLE IF NOT EXISTS solana_research_refresh_jobs (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    mint TEXT NOT NULL,
                    pool_address TEXT NOT NULL,
                    interval_seconds INTEGER NOT NULL CHECK(interval_seconds BETWEEN 300 AND 86400),
                    next_due_at TEXT NOT NULL,
                    last_enqueued_at TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0,1)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(owner_id, workspace_id, mint, pool_address)
                );
                CREATE INDEX IF NOT EXISTS idx_solana_refresh_due
                    ON solana_research_refresh_jobs(enabled, next_due_at);
                CREATE TABLE IF NOT EXISTS solana_research_labels (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    mint TEXT NOT NULL,
                    pool_address TEXT NOT NULL,
                    baseline_snapshot_id TEXT NOT NULL,
                    target_snapshot_id TEXT NOT NULL,
                    horizon_seconds INTEGER NOT NULL,
                    label_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(owner_id, workspace_id, baseline_snapshot_id, horizon_seconds)
                );
                CREATE INDEX IF NOT EXISTS idx_solana_labels_owner_mint
                    ON solana_research_labels(owner_id, workspace_id, mint, horizon_seconds);
                CREATE TABLE IF NOT EXISTS solana_execution_cost_evidence (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    label_id TEXT NOT NULL,
                    mint TEXT NOT NULL,
                    pool_address TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    evidence_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(owner_id, workspace_id, label_id, evidence_sha256)
                );
                CREATE INDEX IF NOT EXISTS idx_solana_cost_evidence_label
                    ON solana_execution_cost_evidence(owner_id, workspace_id, label_id, created_at);
                CREATE TABLE IF NOT EXISTS solana_transaction_attestations (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    label_id TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    mint TEXT NOT NULL,
                    pool_address TEXT NOT NULL,
                    wallet_address TEXT NOT NULL,
                    attestation_json TEXT NOT NULL,
                    attestation_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(owner_id, workspace_id, label_id, signature, attestation_sha256)
                );
                CREATE INDEX IF NOT EXISTS idx_solana_tx_attestations_label
                    ON solana_transaction_attestations(owner_id, workspace_id, label_id, created_at);
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

                CREATE TABLE IF NOT EXISTS agent_workers (
                    worker_id TEXT PRIMARY KEY,
                    last_seen_at TEXT NOT NULL,
                    capabilities_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS autonomy_profiles (
                    owner_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    enabled INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN (0,1)),
                    interval_seconds INTEGER NOT NULL DEFAULT 900,
                    venues_json TEXT NOT NULL,
                    symbols_json TEXT NOT NULL,
                    timeframes_json TEXT NOT NULL,
                    last_scheduled_at TEXT,
                    model_training_enabled INTEGER NOT NULL DEFAULT 0,
                    model_training_interval_seconds INTEGER NOT NULL DEFAULT 86400,
                    last_model_training_at TEXT,
                    updated_at TEXT NOT NULL
                );

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

                CREATE TABLE IF NOT EXISTS exchange_credentials (
                    owner_id TEXT NOT NULL,
                    venue TEXT NOT NULL CHECK(venue IN ('binance','mexc')),
                    api_key_ciphertext TEXT NOT NULL,
                    api_secret_ciphertext TEXT NOT NULL,
                    key_suffix TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner_id, venue)
                );

                CREATE TABLE IF NOT EXISTS live_execution_profiles (
                    owner_id TEXT PRIMARY KEY,
                    venue TEXT NOT NULL CHECK(venue IN ('binance','mexc')),
                    symbols_json TEXT NOT NULL,
                    timeframe TEXT NOT NULL CHECK(timeframe IN ('1h','4h','1d')),
                    capital_allocation_usdt REAL NOT NULL CHECK(capital_allocation_usdt >= 20),
                    risk_config_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN (0,1)),
                    daily_date TEXT,
                    daily_start_equity REAL NOT NULL DEFAULT 0,
                    peak_equity REAL NOT NULL DEFAULT 0,
                    authorized_at TEXT,
                    kill_switch_at TEXT,
                    last_scheduled_at TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS live_execution_states (
                    owner_id TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    last_processed_candle TEXT,
                    position_qty REAL NOT NULL DEFAULT 0,
                    entry_price REAL,
                    daily_date TEXT,
                    daily_start_equity REAL NOT NULL DEFAULT 0,
                    peak_equity REAL NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner_id,venue,symbol,timeframe)
                );

                CREATE TABLE IF NOT EXISTS live_orders (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK(side IN ('buy','sell')),
                    amount REAL NOT NULL CHECK(amount > 0),
                    client_order_id TEXT NOT NULL UNIQUE,
                    exchange_order_id TEXT,
                    status TEXT NOT NULL CHECK(status IN ('submitting','open','closed','canceled','rejected','unknown')),
                    filled REAL NOT NULL DEFAULT 0,
                    average REAL,
                    cost REAL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_live_orders_owner
                    ON live_orders(owner_id,created_at);

                CREATE TABLE IF NOT EXISTS paper_sessions (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    model_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    initial_cash REAL NOT NULL CHECK(initial_cash > 0),
                    cash REAL NOT NULL,
                    position_qty REAL NOT NULL DEFAULT 0 CHECK(position_qty >= 0),
                    entry_price REAL,
                    position_cost REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL CHECK(status IN ('paused','running','stopped')),
                    risk_config_json TEXT NOT NULL,
                    equity REAL NOT NULL,
                    peak_equity REAL NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0,
                    max_drawdown REAL NOT NULL DEFAULT 0,
                    daily_date TEXT,
                    daily_start_equity REAL NOT NULL,
                    last_processed_candle TEXT,
                    last_scheduled_at TEXT,
                    last_tick_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_paper_sessions_owner
                    ON paper_sessions(owner_id, workspace_id, status);

                CREATE TABLE IF NOT EXISTS paper_trades (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES paper_sessions(id) ON DELETE CASCADE,
                    owner_id TEXT NOT NULL,
                    event_type TEXT NOT NULL CHECK(event_type IN ('entry','exit','risk_stop')),
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    notional REAL NOT NULL,
                    fee REAL NOT NULL,
                    slippage_bps REAL NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0,
                    prediction INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    risk_decision_json TEXT NOT NULL,
                    candle_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_paper_trades_session
                    ON paper_trades(owner_id, session_id, created_at);
                """
            )
            autonomy_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(autonomy_profiles)").fetchall()
            }
            for column, definition in (
                ("market_types_json", "TEXT NOT NULL DEFAULT '[\"spot\"]'"),
                ("model_training_enabled", "INTEGER NOT NULL DEFAULT 0"),
                ("model_training_interval_seconds", "INTEGER NOT NULL DEFAULT 86400"),
                ("last_model_training_at", "TEXT"),
            ):
                if column not in autonomy_columns:
                    conn.execute(f"ALTER TABLE autonomy_profiles ADD COLUMN {column} {definition}")
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




    def configure_autonomy(
        self, owner_id: str, workspace_id: str, enabled: bool,
        venues: list[str] | None = None, symbols: list[str] | None = None,
        timeframes: list[str] | None = None, interval_seconds: int = 300,
        model_training_enabled: bool = False,
        model_training_interval_seconds: int = 86400,
        market_types: list[str] | None = None
    ) -> dict[str, Any]:
        """Persist an owner-scoped autonomous public-market research schedule."""
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be boolean")
        venues = venues or ["mexc"]
        symbols = symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
        timeframes = timeframes or ["1h"]
        market_types = market_types or ["spot"]
        if not isinstance(market_types, list) or not market_types or any(
            value not in {"spot", "swap", "future"} for value in market_types
        ):
            raise ValueError("market_types must contain spot, swap and/or future; dated futures require an exact contract symbol")
        if not isinstance(venues, list) or not venues or any(v not in {"binance", "mexc"} for v in venues):
            raise ValueError("venues must contain binance and/or mexc")
        if not isinstance(symbols, list) or not 1 <= len(symbols) <= 20 or any(
            not isinstance(v, str) or "/" not in v or len(v) > 40 for v in symbols
        ):
            raise ValueError("symbols must be 1-20 valid CCXT crypto symbols")
        if not isinstance(timeframes, list) or not timeframes or any(
            v not in {"5m", "15m", "30m", "1h", "4h", "1d"} for v in timeframes
        ):
            raise ValueError("unsupported timeframes")
        if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, int) or not 300 <= interval_seconds <= 86400:
            raise ValueError("interval_seconds must be between 300 and 86400")
        if not isinstance(model_training_enabled, bool):
            raise ValueError("model_training_enabled must be boolean")
        if (isinstance(model_training_interval_seconds, bool)
                or not isinstance(model_training_interval_seconds, int)
                or not 86400 <= model_training_interval_seconds <= 604800):
            raise ValueError("model_training_interval_seconds must be between 86400 and 604800")
        if model_training_enabled and len(set(venues)) * len(set(symbols)) * len(set(timeframes)) * len(set(market_types)) > 8:
            raise ValueError("model training schedule supports at most 10 market combinations")
        now = _utc_now()
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            conn.execute(
                """INSERT INTO autonomy_profiles
                (owner_id, workspace_id, enabled, interval_seconds, venues_json, symbols_json,
                 timeframes_json, market_types_json, last_scheduled_at, model_training_enabled,
                 model_training_interval_seconds, last_model_training_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL, ?)
                ON CONFLICT(owner_id) DO UPDATE SET workspace_id=excluded.workspace_id,
                enabled=excluded.enabled, interval_seconds=excluded.interval_seconds,
                venues_json=excluded.venues_json, symbols_json=excluded.symbols_json,
                timeframes_json=excluded.timeframes_json,
                market_types_json=excluded.market_types_json,
                model_training_enabled=excluded.model_training_enabled,
                model_training_interval_seconds=excluded.model_training_interval_seconds,
                updated_at=excluded.updated_at""",
                (owner_id, workspace_id, int(enabled), interval_seconds,
                 _encode_json(sorted(set(venues))), _encode_json(sorted(set(symbols))),
                 _encode_json(sorted(set(timeframes))), _encode_json(sorted(set(market_types))),
                 int(model_training_enabled), model_training_interval_seconds, now),
            )
            row=conn.execute("SELECT * FROM autonomy_profiles WHERE owner_id=?", (owner_id,)).fetchone()
            conn.execute(
                "INSERT INTO audit_events(id,owner_id,workspace_id,event_type,payload_json,created_at) VALUES(?,?,?,?,?,?)",
                (str(uuid4()), owner_id, workspace_id, "autonomy.enabled" if enabled else "autonomy.disabled",
                 _encode_json({"enabled": enabled, "interval_seconds": interval_seconds,
                              "model_training_enabled": model_training_enabled,
                              "model_training_interval_seconds": model_training_interval_seconds}), now),
            )
        return self._autonomy_profile_dict(row)

    @staticmethod
    def _autonomy_profile_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {"owner_id": row["owner_id"], "workspace_id": row["workspace_id"],
                "enabled": bool(row["enabled"]), "interval_seconds": row["interval_seconds"],
                "venues": json.loads(row["venues_json"]), "symbols": json.loads(row["symbols_json"]),
                "timeframes": json.loads(row["timeframes_json"]),
                "market_types": json.loads(row["market_types_json"]) if "market_types_json" in row.keys() else ["spot"],
                "model_training_enabled": bool(row["model_training_enabled"]),
                "model_training_interval_seconds": row["model_training_interval_seconds"],
                "last_model_training_at": row["last_model_training_at"],
                "last_scheduled_at": row["last_scheduled_at"], "updated_at": row["updated_at"]}

    def get_autonomy_profile(self, owner_id: str) -> dict[str, Any] | None:
        owner_id = _required_text(owner_id, "owner_id")
        with self._connection() as conn:
            row=conn.execute("SELECT * FROM autonomy_profiles WHERE owner_id=?", (owner_id,)).fetchone()
        return self._autonomy_profile_dict(row) if row else None

    def enqueue_due_autonomy_tasks(self, now: datetime | None = None) -> int:
        """Atomically enqueue scheduled spot/swap research and model training tasks."""
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        now_text = current.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
        queued = 0
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM autonomy_profiles WHERE enabled=1").fetchall()
            for row in rows:
                previous = row["last_scheduled_at"]
                scan_due = not previous or (
                    current.astimezone(timezone.utc) - datetime.fromisoformat(
                        previous.replace("Z", "+00:00")
                    )
                ).total_seconds() >= row["interval_seconds"]
                venues = json.loads(row["venues_json"])
                symbols = json.loads(row["symbols_json"])
                timeframes = json.loads(row["timeframes_json"])
                market_types = json.loads(row["market_types_json"]) if "market_types_json" in row.keys() else ["spot"]
                if scan_due:
                    conn.execute(
                        "UPDATE autonomy_profiles SET last_scheduled_at=?, updated_at=? WHERE owner_id=?",
                        (now_text, now_text, row["owner_id"]),
                    )
                    for market_type in market_types:
                        for venue in venues:
                            for symbol in symbols:
                                market_symbol = symbol
                                if market_type == "swap" and symbol.endswith("/USDT") and ":" not in symbol:
                                    market_symbol = symbol + ":USDT"
                                if market_type == "future" and (":" not in symbol or "-" not in symbol):
                                    continue
                                for timeframe in timeframes:
                                    task_id = str(uuid4())
                                    payload = {
                                        "venue": venue, "asset_class": "crypto", "market_type": market_type,
                                        "symbol": market_symbol, "timeframe": timeframe,
                                        "limit": 200, "scheduled_by": "eti.autonomy.supervisor",
                                    }
                                    conn.execute(
                                        """INSERT INTO agent_tasks
                                        (id,owner_id,workspace_id,requester,target_capability,status,input_json,
                                         max_attempts,created_at,updated_at)
                                        VALUES(?,?,?,?,?,'queued',?,3,?,?)""",
                                        (task_id, row["owner_id"], row["workspace_id"],
                                         "eti.autonomy.supervisor", "market.snapshot",
                                         _encode_json(payload), now_text, now_text),
                                    )
                                    conn.execute(
                                        """INSERT INTO audit_events
                                        (id,owner_id,workspace_id,event_type,payload_json,created_at)
                                        VALUES(?,?,?,?,?,?)""",
                                        (str(uuid4()), row["owner_id"], row["workspace_id"],
                                         "autonomy.scan.queued",
                                         _encode_json({"task_id": task_id, "venue": venue, "asset_class": "crypto",
                                                       "market_type": market_type, "symbol": market_symbol,
                                                       "timeframe": timeframe}), now_text),
                                    )
                                    queued += 1
                    # Small public benchmark universe extends autonomous research beyond crypto.
                    global_watchlist = [
                        ("forex", "spot", "EURUSD=X"),
                        ("equity", "spot", "AAPL"),
                        ("commodity", "future", "GC=F"),
                        ("index", "spot", "^GSPC"),
                    ]
                    for asset_class, market_type, symbol in global_watchlist:
                        timeframe = timeframes[0]
                        task_id = str(uuid4())
                        payload = {
                            "venue": "yahoo", "asset_class": asset_class,
                            "market_type": market_type, "symbol": symbol,
                            "timeframe": timeframe, "limit": 200,
                            "scheduled_by": "eti.autonomy.supervisor",
                        }
                        conn.execute(
                            """INSERT INTO agent_tasks
                            (id,owner_id,workspace_id,requester,target_capability,status,input_json,
                             max_attempts,created_at,updated_at)
                            VALUES(?,?,?,?,?,'queued',?,3,?,?)""",
                            (task_id, row["owner_id"], row["workspace_id"],
                             "eti.autonomy.supervisor", "market.snapshot",
                             _encode_json(payload), now_text, now_text),
                        )
                        conn.execute(
                            """INSERT INTO audit_events
                            (id,owner_id,workspace_id,event_type,payload_json,created_at)
                            VALUES(?,?,?,?,?,?)""",
                            (str(uuid4()), row["owner_id"], row["workspace_id"],
                             "autonomy.scan.queued",
                             _encode_json({"task_id": task_id, "venue": "yahoo",
                                           "asset_class": asset_class, "market_type": market_type,
                                           "symbol": symbol, "timeframe": timeframe}), now_text),
                        )
                        queued += 1
                training_enabled = bool(row["model_training_enabled"])
                training_previous = row["last_model_training_at"]
                training_due = training_enabled and (
                    not training_previous or
                    (current.astimezone(timezone.utc) - datetime.fromisoformat(
                        training_previous.replace("Z", "+00:00")
                    )).total_seconds() >= row["model_training_interval_seconds"]
                )
                if training_due:
                    conn.execute(
                        "UPDATE autonomy_profiles SET last_model_training_at=?, updated_at=? WHERE owner_id=?",
                        (now_text, now_text, row["owner_id"]),
                    )
                    # Discover the live crypto universe each day, then train on its highest-volume instruments.
                    discovery_jobs = [
                        (venue, market_type)
                        for venue in ("binance", "mexc")
                        for market_type in ("spot", "swap", "future")
                    ]
                    for venue, market_type in discovery_jobs:
                        task_id = str(uuid4())
                        payload = {
                            "venue": venue, "asset_class": "crypto",
                            "market_type": market_type, "limit": 100,
                            "scheduled_by": "eti.autonomy.model_training",
                        }
                        conn.execute(
                            """INSERT INTO agent_tasks
                            (id,owner_id,workspace_id,requester,target_capability,status,input_json,
                             max_attempts,created_at,updated_at)
                            VALUES(?,?,?,?,?,'queued',?,2,?,?)""",
                            (task_id, row["owner_id"], row["workspace_id"],
                             "eti.autonomy.model_training", "market.discover",
                             _encode_json(payload), now_text, now_text),
                        )
                        conn.execute(
                            """INSERT INTO audit_events
                            (id,owner_id,workspace_id,event_type,payload_json,created_at)
                            VALUES(?,?,?,?,?,?)""",
                            (str(uuid4()), row["owner_id"], row["workspace_id"],
                             "autonomy.market_discovery.scheduled",
                             _encode_json({"task_id": task_id, **payload}), now_text),
                        )
                        queued += 1
                    # Four cross-asset reference models complement the dynamically discovered crypto candidates.
                    training_jobs = [
                        {"venue": "yahoo", "asset_class": "forex", "market_type": "spot", "symbol": "EURUSD=X"},
                        {"venue": "yahoo", "asset_class": "equity", "market_type": "spot", "symbol": "AAPL"},
                        {"venue": "yahoo", "asset_class": "commodity", "market_type": "future", "symbol": "GC=F"},
                        {"venue": "yahoo", "asset_class": "index", "market_type": "spot", "symbol": "^GSPC"},
                    ]
                    for job in training_jobs:
                        timeframe = timeframes[0]
                        task_id = str(uuid4())
                        payload = {
                            **job, "timeframe": timeframe, "limit": 3000, "horizon": 6,
                            "scheduled_by": "eti.autonomy.model_training",
                        }
                        conn.execute(
                            """INSERT INTO agent_tasks
                            (id,owner_id,workspace_id,requester,target_capability,status,input_json,
                             max_attempts,created_at,updated_at)
                            VALUES(?,?,?,?,?,'queued',?,2,?,?)""",
                            (task_id, row["owner_id"], row["workspace_id"],
                             "eti.autonomy.model_training", "model.train_research",
                             _encode_json(payload), now_text, now_text),
                        )
                        conn.execute(
                            """INSERT INTO audit_events
                            (id,owner_id,workspace_id,event_type,payload_json,created_at)
                            VALUES(?,?,?,?,?,?)""",
                            (str(uuid4()), row["owner_id"], row["workspace_id"],
                             "autonomy.model_training.scheduled",
                             _encode_json({"task_id": task_id, **job, "timeframe": timeframe}),
                             now_text),
                        )
                        queued += 1

        return queued

    def record_agent_worker_heartbeat(
        self, worker_id: str, capabilities: list[str], now: int | None = None
    ) -> dict[str, Any]:
        worker_id = _required_text(worker_id, "worker_id", 80)
        if not isinstance(capabilities, list) or any(
            not isinstance(item, str) or not item.strip() or len(item) > 120
            for item in capabilities
        ):
            raise ValueError("capabilities must be a list of non-empty strings")
        normalized = sorted(set(item.strip() for item in capabilities))
        encoded = _encode_json(normalized)
        current = datetime.now(timezone.utc) if now is None else datetime.fromtimestamp(now, timezone.utc)
        seen = current.isoformat(timespec="microseconds").replace("+00:00", "Z")
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO agent_workers(worker_id, last_seen_at, capabilities_json) "
                "VALUES (?, ?, ?) ON CONFLICT(worker_id) DO UPDATE SET "
                "last_seen_at = excluded.last_seen_at, capabilities_json = excluded.capabilities_json",
                (worker_id, seen, encoded),
            )
        return {"worker_id": worker_id, "last_seen_at": seen, "capabilities": normalized}

    def list_agent_workers(
        self, online_window_seconds: int = 60, now: int | None = None
    ) -> list[dict[str, Any]]:
        if (isinstance(online_window_seconds, bool) or
                not isinstance(online_window_seconds, int) or
                not 5 <= online_window_seconds <= 300):
            raise ValueError("online_window_seconds must be an integer from 5 to 300")
        current = datetime.now(timezone.utc) if now is None else datetime.fromtimestamp(now, timezone.utc)
        cutoff = (current - timedelta(seconds=online_window_seconds)).isoformat(
            timespec="microseconds"
        ).replace("+00:00", "Z")
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT worker_id, last_seen_at, capabilities_json "
                "FROM agent_workers ORDER BY worker_id"
            ).fetchall()
        return [
            {"worker_id": row["worker_id"], "last_seen_at": row["last_seen_at"],
             "capabilities": json.loads(row["capabilities_json"]),
             "online": row["last_seen_at"] >= cutoff}
            for row in rows
        ]

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
                "FROM agent_tasks WHERE status = 'running' "
                "AND (lease_expires_at IS NULL OR lease_expires_at <= ?)",
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
        self, task_id: str, lease_token: str, lease_seconds: int = 120, now: int | None = None
    ) -> str:
        task_id = _required_text(task_id, "task_id")
        lease_token = _required_text(lease_token, "lease_token")
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, int) or not 5 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be an integer from 5 to 3600")
        current = datetime.now(timezone.utc) if now is None else datetime.fromtimestamp(now, timezone.utc)
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
        self, task_id: str, lease_token: str, result: Any, now: int | None = None
    ) -> dict[str, Any]:
        task_id = _required_text(task_id, "task_id")
        lease_token = _required_text(lease_token, "lease_token")
        encoded = _encode_json(result)
        now_dt = datetime.now(timezone.utc) if now is None else datetime.fromtimestamp(now, timezone.utc)
        now = now_dt.isoformat(timespec="microseconds").replace("+00:00", "Z")
        with self._connection() as conn:
            row = conn.execute(
                "SELECT owner_id, workspace_id, status, lease_token, lease_expires_at, "
                "requester, target_capability, input_json FROM agent_tasks WHERE id = ?",
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
            # Every completed Solana research task creates one immutable,
            # point-in-time snapshot in the same transaction as task completion.
            task_input = json.loads(row["input_json"])
            if (row["target_capability"] in {"solana.market.scan", "solana.token.audit"}
                    and isinstance(task_input, dict) and isinstance(result, dict)):
                mint = task_input.get("mint") or result.get("mint")
                if isinstance(mint, str) and len(mint) >= 32:
                    snapshot_id = str(uuid4())
                    snapshot_json = _encode_json(result)
                    snapshot_digest = hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()
                    snapshot_fetched_at = now
                    observed_at_ms = result.get("observed_at_ms")
                    if isinstance(observed_at_ms, (int, float)) and not isinstance(observed_at_ms, bool):
                        try:
                            snapshot_fetched_at = datetime.fromtimestamp(
                                float(observed_at_ms) / 1000.0, timezone.utc
                            ).isoformat(timespec="microseconds").replace("+00:00", "Z")
                        except (OverflowError, OSError, ValueError):
                            pass
                    pool_address = result.get("pool_address")
                    evidence = result.get("evidence")
                    if isinstance(evidence, dict):
                        pool_address = pool_address or evidence.get("pair_address")
                    conn.execute(
                        "INSERT OR IGNORE INTO solana_research_snapshots("
                        "id,owner_id,workspace_id,task_id,capability,mint,pool_address,"
                        "schema_version,payload_json,payload_sha256,fetched_at) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (snapshot_id, row["owner_id"], row["workspace_id"], task_id,
                         row["target_capability"], mint,
                         pool_address if isinstance(pool_address, str) else None,
                         1, snapshot_json, snapshot_digest, snapshot_fetched_at),
                    )
                    if row["target_capability"] == "solana.market.scan" and result.get("verdict") == "REJECT":
                        conn.execute(
                            "UPDATE solana_research_refresh_jobs SET enabled=0,updated_at=? "
                            "WHERE owner_id=? AND workspace_id=? AND mint=? AND enabled=1",
                            (now, row["owner_id"], row["workspace_id"], mint),
                        )
                    elif (row["target_capability"] == "solana.market.scan"
                            and row["requester"] == "eti.solana.ingest"
                            and isinstance(task_input, dict)
                            and task_input.get("scheduled_by") == "eti.solana.ingest"
                            and result.get("verdict") in {"WAIT", "UNKNOWN"}
                            and not pool_address):
                        # Keep newly discovered mints on a low-frequency watch
                        # until DexScreener indexes their first pool.
                        pending_pool = "__pending__"
                        refresh_now_dt = datetime.now(timezone.utc)
                        refresh_now = refresh_now_dt.isoformat(timespec="microseconds").replace("+00:00", "Z")
                        refresh_due = (refresh_now_dt + timedelta(seconds=300)).isoformat(
                            timespec="microseconds"
                        ).replace("+00:00", "Z")
                        conn.execute(
                            "INSERT INTO solana_research_refresh_jobs(id,owner_id,workspace_id,mint,"
                            "pool_address,interval_seconds,next_due_at,created_at,updated_at) "
                            "VALUES(?,?,?,?,?,300,?,?,?) ON CONFLICT(owner_id,workspace_id,mint,pool_address) "
                            "DO UPDATE SET enabled=1,updated_at=excluded.updated_at",
                            (str(uuid4()), row["owner_id"], row["workspace_id"], mint,
                             pending_pool, refresh_due, refresh_now, refresh_now),
                        )
                    elif isinstance(pool_address, str) and pool_address:
                        # A mint may migrate to a new dominant pool. Stop polling
                        # obsolete pool schedules before activating the current one.
                        conn.execute(
                            "UPDATE solana_research_refresh_jobs SET enabled=0,updated_at=? "
                            "WHERE owner_id=? AND workspace_id=? AND mint=? AND pool_address<>? AND enabled=1",
                            (now, row["owner_id"], row["workspace_id"], mint, pool_address),
                        )
                        refresh_now_dt = datetime.now(timezone.utc)
                        refresh_now = refresh_now_dt.isoformat(timespec="microseconds").replace("+00:00", "Z")
                        refresh_due = (refresh_now_dt + timedelta(seconds=300)).isoformat(
                            timespec="microseconds"
                        ).replace("+00:00", "Z")
                        conn.execute(
                            "INSERT INTO solana_research_refresh_jobs(id,owner_id,workspace_id,mint,"
                            "pool_address,interval_seconds,next_due_at,created_at,updated_at) "
                            "VALUES(?,?,?,?,?,900,?,?,?) ON CONFLICT(owner_id,workspace_id,mint,pool_address) "
                            "DO UPDATE SET enabled=1,updated_at=excluded.updated_at",
                            (str(uuid4()), row["owner_id"], row["workspace_id"], mint,
                             pool_address, refresh_due, refresh_now, refresh_now),
                        )
                    conn.execute(
                        "INSERT INTO audit_events(id,owner_id,workspace_id,event_type,payload_json,created_at) "
                        "VALUES(?,?,?,'solana.snapshot.persisted',?,?)",
                        (str(uuid4()), row["owner_id"], row["workspace_id"],
                         _encode_json({"task_id": task_id, "snapshot_id": snapshot_id,
                                       "capability": row["target_capability"], "mint": mint,
                                       "payload_sha256": snapshot_digest}), now),
                    )
            # Persist the audit follow-up in the same transaction as scan
            # completion, eliminating a crash window between completion and enqueue.
            if (row["target_capability"] == "solana.market.scan"
                    and row["requester"] == "eti.solana.ingest"
                    and isinstance(task_input, dict)
                    and task_input.get("scheduled_by") == "eti.solana.ingest"
                    and isinstance(result, dict)
                    and result.get("verdict") != "REJECT"):
                mint = task_input.get("mint")
                if isinstance(mint, str):
                    audit_task_id = str(uuid4())
                    audit_payload = {
                        "mint": mint, "source_scan_task_id": task_id,
                        "source_event_id": task_input.get("source_event_id"),
                        "scheduled_by": "eti.solana.ingest",
                    }
                    conn.execute(
                        "INSERT INTO agent_tasks(id,owner_id,workspace_id,requester,target_capability,"
                        "status,input_json,max_attempts,created_at,updated_at) "
                        "VALUES(?,?,?,?,'solana.token.audit','queued',?,2,?,?)",
                        (audit_task_id, row["owner_id"], row["workspace_id"],
                         "eti.solana.ingest", _encode_json(audit_payload), now, now),
                    )
                    conn.execute(
                        "INSERT INTO audit_events(id,owner_id,workspace_id,event_type,payload_json,created_at) "
                        "VALUES(?,?,?,'solana.audit.queued',?,?)",
                        (str(uuid4()), row["owner_id"], row["workspace_id"],
                         _encode_json({"scan_task_id": task_id, "audit_task_id": audit_task_id,
                                       "mint": mint}), now),
                    )
        return self.get_agent_task(row["owner_id"], row["workspace_id"], task_id)  # type: ignore[return-value]

    def fail_claimed_agent_task(
        self, task_id: str, lease_token: str, error: Any, *, retryable: bool = True,
        now: int | None = None
    ) -> dict[str, Any]:
        task_id = _required_text(task_id, "task_id")
        lease_token = _required_text(lease_token, "lease_token")
        if not isinstance(retryable, bool):
            raise ValueError("retryable must be a boolean")
        encoded = _encode_json(error)
        now_dt = datetime.now(timezone.utc) if now is None else datetime.fromtimestamp(now, timezone.utc)
        now = now_dt.isoformat(timespec="microseconds").replace("+00:00", "Z")
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

    def cancel_agent_task(
        self, owner_id: str, workspace_id: str, task_id: str
    ) -> dict[str, Any]:
        """Atomically cancel only a queued task; running work belongs to the worker."""
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        task_id = _required_text(task_id, "task_id")
        now = _utc_now()
        with self._connection() as conn:
            cursor = conn.execute(
                "UPDATE agent_tasks SET status = 'cancelled', lease_expires_at = NULL, updated_at = ? "
                "WHERE id = ? AND owner_id = ? AND workspace_id = ? AND status = 'queued'",
                (now, task_id, owner_id, workspace_id),
            )
            if cursor.rowcount != 1:
                exists = conn.execute(
                    "SELECT status FROM agent_tasks WHERE id = ? AND owner_id = ? AND workspace_id = ?",
                    (task_id, owner_id, workspace_id),
                ).fetchone()
                if exists is None:
                    raise LookupError("task not found")
                raise ValueError("only queued tasks can be cancelled")
            conn.execute(
                "INSERT INTO audit_events(id, owner_id, workspace_id, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid4()), owner_id, workspace_id, "agent.task.cancelled",
                 _encode_json({"task_id": task_id}), now),
            )
        return self.get_agent_task(owner_id, workspace_id, task_id)  # type: ignore[return-value]

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


    def delete_workspace(self, owner_id: str, workspace_id: str) -> bool:
        """Delete one owned workspace and its dependent memory/tasks/messages.

        Audit records are retained with workspace_id cleared by the FK's
        ON DELETE SET NULL policy, preserving an account-level deletion record.
        """
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        now = _utc_now()
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM workspaces WHERE id = ? AND owner_id = ?",
                (workspace_id, owner_id),
            )
            if cursor.rowcount != 1:
                return False
            conn.execute(
                "INSERT INTO audit_events(id, owner_id, workspace_id, event_type, payload_json, created_at) "
                "VALUES (?, ?, NULL, 'workspace.deleted', ?, ?)",
                (str(uuid4()), owner_id, _encode_json({"workspace_id": workspace_id}), now),
            )
        return True

    def export_account_data(self, owner_id: str) -> dict[str, Any]:
        """Return an account-scoped portable export without authentication secrets."""
        owner_id = _required_text(owner_id, "owner_id")
        with self._connection() as conn:
            workspaces = conn.execute(
                "SELECT id, name, created_at FROM workspaces WHERE owner_id = ? ORDER BY created_at, id",
                (owner_id,),
            ).fetchall()
            workspace_ids = [row["id"] for row in workspaces]
            memory, tasks, messages = [], [], []
            for workspace_id in workspace_ids:
                for row in conn.execute(
                    "SELECT id, workspace_id, namespace, item_key, value_json, created_at, updated_at "
                    "FROM memory_items WHERE owner_id = ? AND workspace_id = ? ORDER BY namespace, item_key",
                    (owner_id, workspace_id),
                ).fetchall():
                    memory.append({"id": row["id"], "workspace_id": row["workspace_id"],
                                   "namespace": row["namespace"], "key": row["item_key"],
                                   "value": json.loads(row["value_json"]),
                                   "created_at": row["created_at"], "updated_at": row["updated_at"]})
                for row in conn.execute(
                    "SELECT id, workspace_id, requester, target_capability, status, input_json, "
                    "result_json, deadline_at, attempt_count, max_attempts, created_at, updated_at "
                    "FROM agent_tasks WHERE owner_id = ? AND workspace_id = ? ORDER BY created_at, id",
                    (owner_id, workspace_id),
                ).fetchall():
                    tasks.append({"id": row["id"], "workspace_id": row["workspace_id"],
                                  "requester": row["requester"], "target_capability": row["target_capability"],
                                  "status": row["status"], "input": json.loads(row["input_json"]),
                                  "result": json.loads(row["result_json"]) if row["result_json"] else None,
                                  "deadline_at": row["deadline_at"], "attempt_count": row["attempt_count"],
                                  "max_attempts": row["max_attempts"], "created_at": row["created_at"],
                                  "updated_at": row["updated_at"]})
                for row in conn.execute(
                    "SELECT id, workspace_id, task_id, sender, recipient, message_type, payload_json, created_at "
                    "FROM agent_messages WHERE owner_id = ? AND workspace_id = ? ORDER BY created_at, id",
                    (owner_id, workspace_id),
                ).fetchall():
                    messages.append({"id": row["id"], "workspace_id": row["workspace_id"],
                                     "task_id": row["task_id"], "sender": row["sender"],
                                     "recipient": row["recipient"], "message_type": row["message_type"],
                                     "payload": json.loads(row["payload_json"]), "created_at": row["created_at"]})
            audit = conn.execute(
                "SELECT id, workspace_id, event_type, payload_json, created_at FROM audit_events "
                "WHERE owner_id = ? ORDER BY created_at, id", (owner_id,)
            ).fetchall()
        return {
            "format": "eti-account-export-v1",
            "account_id": owner_id,
            "workspaces": [dict(row) for row in workspaces],
            "memory": memory,
            "agent_tasks": tasks,
            "agent_messages": messages,
            "audit_events": [{"id": row["id"], "workspace_id": row["workspace_id"],
                              "event_type": row["event_type"], "payload": json.loads(row["payload_json"]),
                              "created_at": row["created_at"]} for row in audit],
        }

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


    def create_paper_session(
        self, owner_id: str, workspace_id: str, model_id: str, model_version: str,
        venue: str, symbol: str, timeframe: str, initial_cash: float,
        risk_config: dict[str, Any],
    ) -> dict[str, Any]:
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        model_id = _required_text(model_id, "model_id", 120)
        model_version = _required_text(model_version, "model_version", 80)
        if venue not in {"binance", "mexc"}:
            raise ValueError("venue must be binance or mexc")
        if not isinstance(symbol, str) or not symbol.endswith("/USDT") or len(symbol) > 40:
            raise ValueError("paper trading currently supports USDT-quoted spot pairs only")
        if timeframe not in {"1h", "4h", "1d"}:
            raise ValueError("paper trading supports 1h, 4h, or 1d candles")
        if isinstance(initial_cash, bool) or not isinstance(initial_cash, (int, float)) or not math.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be a finite positive amount")
        if not isinstance(risk_config, dict):
            raise ValueError("risk_config must be an object")
        risk_config = {
            "max_risk_per_trade": 0.005, "max_position_fraction": 0.10,
            "max_gross_exposure": 0.25, "max_daily_loss": 0.02,
            "max_drawdown": 0.08, "fee_bps": 6.0, "slippage_bps": 8.0,
            "confidence_threshold": 0.55, **risk_config,
        }
        safe_caps = {
            "max_risk_per_trade": 0.005, "max_position_fraction": 0.10,
            "max_gross_exposure": 0.25, "max_daily_loss": 0.02,
            "max_drawdown": 0.08,
        }
        for key, maximum in safe_caps.items():
            value = risk_config.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 < value <= maximum):
                raise ValueError(f"{key} exceeds the allowed paper risk limit")
        for key in ("fee_bps", "slippage_bps"):
            value = risk_config.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 100):
                raise ValueError(f"{key} must be between 0 and 100")
        threshold = risk_config.get("confidence_threshold")
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not 0.50 <= threshold <= 0.95):
            raise ValueError("confidence_threshold must be between 0.50 and 0.95")
        risk_config["allow_short"] = False
        risk_config["allow_live_orders"] = False
        model = self.get_model_version(model_id, model_version)
        if model is None or model["lifecycle"] != "approved":
            raise ValueError("paper trading requires an approved model version")
        if "crypto_spot" not in model["supported_classes"]:
            raise ValueError("approved model does not support crypto spot")
        session_id, now = str(uuid4()), _utc_now()
        cash = float(initial_cash)
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            conn.execute(
                """INSERT INTO paper_sessions
                (id,owner_id,workspace_id,model_id,model_version,venue,symbol,timeframe,
                 initial_cash,cash,position_qty,entry_price,position_cost,status,risk_config_json,
                 equity,peak_equity,realized_pnl,max_drawdown,daily_date,daily_start_equity,
                 last_processed_candle,last_scheduled_at,last_tick_at,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,0,NULL,0,'paused',?,?,?,0,0,NULL,?,NULL,NULL,NULL,?,?)""",
                (session_id, owner_id, workspace_id, model_id, model_version, venue, symbol,
                 timeframe, cash, cash, _encode_json(risk_config), cash, cash, cash, now, now),
            )
        return self.get_paper_session(owner_id, session_id)

    @staticmethod
    def _paper_session_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "owner_id": row["owner_id"], "workspace_id": row["workspace_id"],
            "model_id": row["model_id"], "model_version": row["model_version"],
            "venue": row["venue"], "symbol": row["symbol"], "timeframe": row["timeframe"],
            "initial_cash": row["initial_cash"], "cash": row["cash"],
            "position_qty": row["position_qty"], "entry_price": row["entry_price"],
            "position_cost": row["position_cost"], "status": row["status"],
            "risk_config": json.loads(row["risk_config_json"]), "equity": row["equity"],
            "peak_equity": row["peak_equity"], "realized_pnl": row["realized_pnl"],
            "max_drawdown": row["max_drawdown"], "daily_date": row["daily_date"],
            "daily_start_equity": row["daily_start_equity"],
            "last_processed_candle": row["last_processed_candle"],
            "last_tick_at": row["last_tick_at"], "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_paper_session(self, owner_id: str, session_id: str) -> dict[str, Any] | None:
        owner_id = _required_text(owner_id, "owner_id")
        session_id = _required_text(session_id, "session_id", 128)
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM paper_sessions WHERE owner_id=? AND id=?",
                (owner_id, session_id),
            ).fetchone()
        return self._paper_session_dict(row) if row else None

    def list_paper_sessions(self, owner_id: str, workspace_id: str) -> list[dict[str, Any]]:
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM paper_sessions WHERE owner_id=? AND workspace_id=? ORDER BY created_at DESC",
                (owner_id, workspace_id),
            ).fetchall()
        return [self._paper_session_dict(row) for row in rows]

    def list_paper_trades(self, owner_id: str, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT * FROM paper_trades WHERE owner_id=? AND session_id=?
                ORDER BY created_at DESC LIMIT ?""",
                (_required_text(owner_id, "owner_id"), _required_text(session_id, "session_id", 128), limit),
            ).fetchall()
        return [{
            "id": row["id"], "event_type": row["event_type"], "quantity": row["quantity"],
            "price": row["price"], "notional": row["notional"], "fee": row["fee"],
            "slippage_bps": row["slippage_bps"], "realized_pnl": row["realized_pnl"],
            "prediction": row["prediction"], "confidence": row["confidence"],
            "risk_decision": json.loads(row["risk_decision_json"]),
            "candle_at": row["candle_at"], "created_at": row["created_at"],
        } for row in rows]

    def set_paper_session_status(self, owner_id: str, session_id: str, status: str) -> dict[str, Any]:
        if status not in {"running", "paused", "stopped"}:
            raise ValueError("invalid paper session status")
        owner_id = _required_text(owner_id, "owner_id")
        session_id = _required_text(session_id, "session_id", 128)
        now = _utc_now()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT status FROM paper_sessions WHERE owner_id=? AND id=?",
                (owner_id, session_id),
            ).fetchone()
            if row is None:
                raise LookupError("paper session not found")
            current = row["status"]
            allowed = {
                "paused": {"running", "stopped"},
                "running": {"paused", "stopped"},
                "stopped": set(),
            }
            if status != current and status not in allowed[current]:
                raise ValueError("invalid paper session state transition")
            conn.execute(
                "UPDATE paper_sessions SET status=?, last_scheduled_at=NULL, updated_at=? WHERE owner_id=? AND id=?",
                (status, now, owner_id, session_id),
            )
        return self.get_paper_session(owner_id, session_id)

    def enqueue_due_paper_tasks(self, now: datetime | None = None) -> int:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        now_text = current.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
        intervals = {"1h": 3600, "4h": 14400, "1d": 86400}
        queued = 0
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM paper_sessions WHERE status='running'").fetchall()
            for row in rows:
                previous = row["last_scheduled_at"]
                if previous:
                    last = datetime.fromisoformat(previous.replace("Z", "+00:00"))
                    if (current.astimezone(timezone.utc) - last).total_seconds() < intervals[row["timeframe"]]:
                        continue
                conn.execute(
                    "UPDATE paper_sessions SET last_scheduled_at=?, updated_at=? WHERE id=?",
                    (now_text, now_text, row["id"]),
                )
                task_id = str(uuid4())
                payload = {"owner_id": row["owner_id"], "session_id": row["id"]}
                conn.execute(
                    """INSERT INTO agent_tasks
                    (id,owner_id,workspace_id,requester,target_capability,status,input_json,
                     max_attempts,created_at,updated_at)
                    VALUES(?,?,?,?,?,'queued',?,2,?,?)""",
                    (task_id, row["owner_id"], row["workspace_id"], "eti.paper.supervisor",
                     "paper.tick", _encode_json(payload), now_text, now_text),
                )
                conn.execute(
                    """INSERT INTO audit_events
                    (id,owner_id,workspace_id,event_type,payload_json,created_at)
                    VALUES(?,?,?,?,?,?)""",
                    (str(uuid4()), row["owner_id"], row["workspace_id"], "paper.tick.scheduled",
                     _encode_json({"task_id": task_id, "session_id": row["id"]}), now_text),
                )
                queued += 1
        return queued

    def apply_paper_tick(
        self, owner_id: str, session_id: str, candle_at: str,
        state: dict[str, Any], trade_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        owner_id = _required_text(owner_id, "owner_id")
        session_id = _required_text(session_id, "session_id", 128)
        candle_at = _required_text(candle_at, "candle_at", 80)
        if not isinstance(state, dict) or not isinstance(trade_events, list):
            raise ValueError("paper tick state and events are invalid")
        required = (
            "cash", "position_qty", "entry_price", "position_cost", "status",
            "equity", "peak_equity", "realized_pnl", "max_drawdown",
            "daily_date", "daily_start_equity",
        )
        if any(key not in state for key in required):
            raise ValueError("paper tick state is incomplete")
        numeric_keys = (
            "cash", "position_qty", "position_cost", "equity", "peak_equity",
            "realized_pnl", "max_drawdown", "daily_start_equity",
        )
        if any(isinstance(state[key], bool) or not isinstance(state[key], (int, float))
               or not math.isfinite(state[key]) for key in numeric_keys):
            raise ValueError("paper tick contains invalid numeric state")
        if (state["cash"] < -1e-8 or state["position_qty"] < 0
                or state["position_cost"] < 0 or state["equity"] <= 0
                or state["peak_equity"] < state["equity"]
                or state["status"] not in {"running", "stopped"}):
            raise ValueError("paper tick state violates account invariants")
        if state["entry_price"] is not None and (
            isinstance(state["entry_price"], bool)
            or not isinstance(state["entry_price"], (int, float))
            or not math.isfinite(state["entry_price"]) or state["entry_price"] <= 0
        ):
            raise ValueError("paper tick entry price is invalid")
        if any(not isinstance(event, dict)
               or event.get("event_type") not in {"entry", "exit", "risk_stop"}
               for event in trade_events):
            raise ValueError("paper tick trade event is invalid")
        now = _utc_now()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT status,last_processed_candle FROM paper_sessions WHERE owner_id=? AND id=?",
                (owner_id, session_id),
            ).fetchone()
            if row is None:
                raise LookupError("paper session not found")
            if row["status"] != "running":
                return self.get_paper_session(owner_id, session_id)
            if row["last_processed_candle"] and candle_at <= row["last_processed_candle"]:
                return self.get_paper_session(owner_id, session_id)
            conn.execute(
                """UPDATE paper_sessions SET cash=?,position_qty=?,entry_price=?,position_cost=?,
                status=?,equity=?,peak_equity=?,realized_pnl=?,max_drawdown=?,daily_date=?,
                daily_start_equity=?,last_processed_candle=?,last_tick_at=?,updated_at=?
                WHERE owner_id=? AND id=?""",
                (state["cash"], state["position_qty"], state["entry_price"], state["position_cost"],
                 state["status"], state["equity"], state["peak_equity"], state["realized_pnl"],
                 state["max_drawdown"], state["daily_date"], state["daily_start_equity"],
                 candle_at, now, now, owner_id, session_id),
            )
            for event in trade_events:
                conn.execute(
                    """INSERT INTO paper_trades
                    (id,session_id,owner_id,event_type,quantity,price,notional,fee,slippage_bps,
                     realized_pnl,prediction,confidence,risk_decision_json,candle_at,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (str(uuid4()), session_id, owner_id, event["event_type"], event["quantity"],
                     event["price"], event["notional"], event["fee"], event["slippage_bps"],
                     event["realized_pnl"], event["prediction"], event["confidence"],
                     _encode_json(event["risk_decision"]), candle_at, now),
                )
        return self.get_paper_session(owner_id, session_id)


    def upsert_exchange_credential(
        self, owner_id: str, venue: str, api_key_ciphertext: str,
        api_secret_ciphertext: str, key_suffix: str,
    ) -> dict[str, Any]:
        """Store already-encrypted exchange secrets; never accept plaintext here."""
        owner_id = _required_text(owner_id, "owner_id")
        if venue not in {"binance", "mexc"}:
            raise ValueError("unsupported exchange venue")
        api_key_ciphertext = _required_text(api_key_ciphertext, "api_key_ciphertext", 2048)
        api_secret_ciphertext = _required_text(api_secret_ciphertext, "api_secret_ciphertext", 2048)
        key_suffix = _required_text(key_suffix, "key_suffix", 8)
        now = _utc_now()
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO exchange_credentials
                (owner_id,venue,api_key_ciphertext,api_secret_ciphertext,key_suffix,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(owner_id,venue) DO UPDATE SET
                api_key_ciphertext=excluded.api_key_ciphertext,
                api_secret_ciphertext=excluded.api_secret_ciphertext,
                key_suffix=excluded.key_suffix,updated_at=excluded.updated_at""",
                (owner_id, venue, api_key_ciphertext, api_secret_ciphertext, key_suffix, now, now),
            )
            conn.execute(
                "INSERT INTO audit_events(id,owner_id,workspace_id,event_type,payload_json,created_at) VALUES(?,?,NULL,?,?,?)",
                (str(uuid4()), owner_id, "exchange.credentials.saved",
                 _encode_json({"venue": venue, "key_suffix": key_suffix}), now),
            )
        return {"venue": venue, "key_suffix": key_suffix, "created_at": now, "updated_at": now}

    def list_exchange_credentials(self, owner_id: str) -> list[dict[str, Any]]:
        owner_id = _required_text(owner_id, "owner_id")
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT venue,key_suffix,created_at,updated_at FROM exchange_credentials WHERE owner_id=? ORDER BY venue",
                (owner_id,),
            ).fetchall()
        return [{"venue": row["venue"], "masked_api_key": "••••" + row["key_suffix"],
                 "created_at": row["created_at"], "updated_at": row["updated_at"]}
                for row in rows]

    def get_exchange_credential(self, owner_id: str, venue: str) -> dict[str, Any] | None:
        owner_id = _required_text(owner_id, "owner_id")
        if venue not in {"binance", "mexc"}:
            raise ValueError("unsupported exchange venue")
        with self._connection() as conn:
            row = conn.execute(
                "SELECT owner_id,venue,api_key_ciphertext,api_secret_ciphertext,key_suffix,created_at,updated_at "
                "FROM exchange_credentials WHERE owner_id=? AND venue=?",
                (owner_id, venue),
            ).fetchone()
        return dict(row) if row else None

    def delete_exchange_credential(self, owner_id: str, venue: str) -> bool:
        owner_id = _required_text(owner_id, "owner_id")
        if venue not in {"binance", "mexc"}:
            raise ValueError("unsupported exchange venue")
        now = _utc_now()
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM exchange_credentials WHERE owner_id=? AND venue=?",
                (owner_id, venue),
            )
            if cursor.rowcount:
                conn.execute(
                    "INSERT INTO audit_events(id,owner_id,workspace_id,event_type,payload_json,created_at) VALUES(?,?,NULL,?,?,?)",
                    (str(uuid4()), owner_id, "exchange.credentials.deleted",
                     _encode_json({"venue": venue}), now),
                )
        return cursor.rowcount == 1


    @staticmethod
    def _live_profile_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "owner_id": row["owner_id"], "venue": row["venue"],
            "symbols": json.loads(row["symbols_json"]), "timeframe": row["timeframe"],
            "capital_allocation_usdt": row["capital_allocation_usdt"],
            "risk_config": json.loads(row["risk_config_json"]),
            "enabled": bool(row["enabled"]), "daily_date": row["daily_date"],
            "daily_start_equity": row["daily_start_equity"], "peak_equity": row["peak_equity"],
            "authorized_at": row["authorized_at"],
            "kill_switch_at": row["kill_switch_at"],
            "last_scheduled_at": row["last_scheduled_at"], "updated_at": row["updated_at"],
        }

    def save_live_execution_profile(
        self, owner_id: str, venue: str, symbols: list[str], timeframe: str,
        capital_allocation_usdt: float, risk_config: dict[str, Any],
    ) -> dict[str, Any]:
        owner_id = _required_text(owner_id, "owner_id")
        if not self.list_workspaces(owner_id):
            self.create_workspace(owner_id, "ETI Live Execution")
        if venue not in {"binance", "mexc"}:
            raise ValueError("venue must be binance or mexc")
        if not isinstance(symbols, list) or not 1 <= len(symbols) <= 8:
            raise ValueError("symbols must contain 1 to 8 spot pairs")
        normalized = sorted(set(_required_text(item, "symbol", 40) for item in symbols))
        if any("/" not in item or not item.endswith("/USDT") for item in normalized):
            raise ValueError("live spot symbols must be USDT-quoted pairs")
        if timeframe not in {"1h", "4h", "1d"}:
            raise ValueError("live execution timeframe must be 1h, 4h, or 1d")
        if (isinstance(capital_allocation_usdt, bool)
                or not isinstance(capital_allocation_usdt, (int, float))
                or not math.isfinite(capital_allocation_usdt)
                or not 20 <= capital_allocation_usdt <= 1_000_000):
            raise ValueError("capital allocation must be between 20 and 1000000 USDT")
        if not isinstance(risk_config, dict):
            raise ValueError("risk_config must be an object")
        defaults = {
            "max_risk_per_trade": 0.005, "max_position_fraction": 0.10,
            "max_gross_exposure": 0.25, "max_daily_loss": 0.02,
            "max_drawdown": 0.08, "confidence_threshold": 0.60,
        }
        config = {**defaults, **risk_config}
        ceilings = {
            "max_risk_per_trade": 0.005, "max_position_fraction": 0.10,
            "max_gross_exposure": 0.25, "max_daily_loss": 0.02,
            "max_drawdown": 0.08,
        }
        for name, ceiling in ceilings.items():
            value = config.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 < value <= ceiling):
                raise ValueError(f"{name} exceeds the hard live risk ceiling")
        threshold = config.get("confidence_threshold")
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not 0.55 <= threshold <= 0.95):
            raise ValueError("confidence_threshold must be between 0.55 and 0.95")
        config.update({"allow_short": False, "max_leverage": 1.0, "allow_live_orders": False})
        now = _utc_now()
        with self._connection() as conn:
            existing = conn.execute(
                "SELECT enabled,venue,timeframe FROM live_execution_profiles WHERE owner_id=?", (owner_id,)
            ).fetchone()
            if existing and existing["enabled"]:
                raise ValueError("pause live execution before changing its settings")
            open_positions = conn.execute(
                "SELECT venue,symbol,timeframe FROM live_execution_states WHERE owner_id=? AND position_qty>0",
                (owner_id,),
            ).fetchall()
            if any(item["venue"] != venue or item["timeframe"] != timeframe or item["symbol"] not in normalized
                   for item in open_positions):
                raise ValueError("close ETI-managed positions before changing venue, timeframe or watchlist")
            pending_orders = conn.execute(
                "SELECT venue,symbol FROM live_orders WHERE owner_id=? AND status IN ('submitting','open','unknown')",
                (owner_id,),
            ).fetchall()
            if any(item["venue"] != venue or item["symbol"] not in normalized
                   or (existing and existing["timeframe"] != timeframe)
                   for item in pending_orders):
                raise ValueError("reconcile pending live orders before changing venue, timeframe or watchlist")
            conn.execute(
                """INSERT INTO live_execution_profiles
                (owner_id,venue,symbols_json,timeframe,capital_allocation_usdt,risk_config_json,
                 enabled,authorized_at,kill_switch_at,last_scheduled_at,updated_at)
                VALUES(?,?,?,?,?,?,0,NULL,NULL,NULL,?)
                ON CONFLICT(owner_id) DO UPDATE SET venue=excluded.venue,
                symbols_json=excluded.symbols_json,timeframe=excluded.timeframe,
                capital_allocation_usdt=excluded.capital_allocation_usdt,
                risk_config_json=excluded.risk_config_json,last_scheduled_at=NULL,
                updated_at=excluded.updated_at""",
                (owner_id, venue, _encode_json(normalized), timeframe,
                 float(capital_allocation_usdt), _encode_json(config), now),
            )
            conn.execute(
                "INSERT INTO audit_events(id,owner_id,workspace_id,event_type,payload_json,created_at) VALUES(?,?,NULL,?,?,?)",
                (str(uuid4()), owner_id, "live.execution.settings_saved",
                 _encode_json({"venue": venue, "symbols": normalized, "timeframe": timeframe,
                               "capital_allocation_usdt": float(capital_allocation_usdt)}), now),
            )
        return self.get_live_execution_profile(owner_id)

    def get_live_execution_profile(self, owner_id: str) -> dict[str, Any] | None:
        owner_id = _required_text(owner_id, "owner_id")
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM live_execution_profiles WHERE owner_id=?", (owner_id,)
            ).fetchone()
        return self._live_profile_dict(row) if row else None

    def authorize_live_execution(self, owner_id: str) -> dict[str, Any]:
        owner_id = _required_text(owner_id, "owner_id")
        now = _utc_now()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM live_execution_profiles WHERE owner_id=?", (owner_id,)
            ).fetchone()
            if row is None:
                raise LookupError("live execution settings not found")
            credential = conn.execute(
                "SELECT 1 FROM exchange_credentials WHERE owner_id=? AND venue=?",
                (owner_id, row["venue"]),
            ).fetchone()
            if credential is None:
                raise ValueError("connect and verify the selected exchange before authorizing live execution")
            symbols = json.loads(row["symbols_json"])
            eligible = 0
            for symbol in symbols:
                model_id = f"eti-{row['venue']}-{symbol.replace('/', '-').replace(':', '-')}-{row['timeframe']}".lower()
                if conn.execute(
                    "SELECT 1 FROM model_versions WHERE model_id=? AND lifecycle='approved' LIMIT 1",
                    (model_id,),
                ).fetchone():
                    eligible += 1
            if eligible == 0:
                raise ValueError("at least one approved model is required for the selected live watchlist")
            conn.execute(
                "UPDATE live_execution_profiles SET enabled=1,authorized_at=?,kill_switch_at=NULL,last_scheduled_at=NULL,updated_at=? WHERE owner_id=?",
                (now, now, owner_id),
            )
            conn.execute(
                "INSERT INTO audit_events(id,owner_id,workspace_id,event_type,payload_json,created_at) VALUES(?,?,NULL,?,?,?)",
                (str(uuid4()), owner_id, "live.execution.authorized",
                 _encode_json({"venue": row["venue"], "eligible_models": eligible,
                               "symbols": symbols, "capital_allocation_usdt": row["capital_allocation_usdt"]}), now),
            )
        return self.get_live_execution_profile(owner_id)

    def disable_live_execution(self, owner_id: str, reason: str = "owner_kill_switch") -> dict[str, Any] | None:
        owner_id = _required_text(owner_id, "owner_id")
        reason = _required_text(reason, "reason", 120)
        now = _utc_now()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT workspace_id FROM autonomy_profiles WHERE owner_id=?", (owner_id,)
            ).fetchone()
            cursor = conn.execute(
                "UPDATE live_execution_profiles SET enabled=0,kill_switch_at=?,updated_at=? WHERE owner_id=?",
                (now, now, owner_id),
            )
            if cursor.rowcount:
                conn.execute(
                    "INSERT INTO audit_events(id,owner_id,workspace_id,event_type,payload_json,created_at) VALUES(?,?,NULL,?,?,?)",
                    (str(uuid4()), owner_id, "live.execution.disabled",
                     _encode_json({"reason": reason}), now),
                )
        return self.get_live_execution_profile(owner_id)

    def enqueue_due_live_execution_tasks(self, now: datetime | None = None) -> int:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        now_text = current.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
        intervals = {"1h": 3600, "4h": 14400, "1d": 86400}
        queued = 0
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM live_execution_profiles WHERE enabled=1"
            ).fetchall()
            for row in rows:
                previous = row["last_scheduled_at"]
                if previous:
                    last = datetime.fromisoformat(previous.replace("Z", "+00:00"))
                    if (current.astimezone(timezone.utc) - last).total_seconds() < intervals[row["timeframe"]]:
                        continue
                conn.execute(
                    "UPDATE live_execution_profiles SET last_scheduled_at=?,updated_at=? WHERE owner_id=?",
                    (now_text, now_text, row["owner_id"]),
                )
                for symbol in json.loads(row["symbols_json"]):
                    task_id = str(uuid4())
                    payload = {"owner_id": row["owner_id"], "venue": row["venue"],
                               "symbol": symbol, "timeframe": row["timeframe"],
                               "task_id": task_id, "scheduled_by": "eti.live.supervisor"}
                    conn.execute(
                        """INSERT INTO agent_tasks
                        (id,owner_id,workspace_id,requester,target_capability,status,input_json,
                         max_attempts,created_at,updated_at)
                        VALUES(?,?,(SELECT id FROM workspaces WHERE owner_id=? ORDER BY created_at LIMIT 1),
                        'eti.live.supervisor','live.tick','queued',?,2,?,?)""",
                        (task_id, row["owner_id"], row["owner_id"], _encode_json(payload), now_text, now_text),
                    )
                    conn.execute(
                        "INSERT INTO audit_events(id,owner_id,workspace_id,event_type,payload_json,created_at) VALUES(?,?,NULL,?,?,?)",
                        (str(uuid4()), row["owner_id"], "live.execution.tick_scheduled",
                         _encode_json({"task_id": task_id, "venue": row["venue"], "symbol": symbol}), now_text),
                    )
                    queued += 1
        return queued


    def list_live_execution_states(self, owner_id: str, venue: str, timeframe: str) -> list[dict[str, Any]]:
        owner_id = _required_text(owner_id, "owner_id")
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM live_execution_states WHERE owner_id=? AND venue=? AND timeframe=?",
                (owner_id, venue, timeframe),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_live_execution_account_state(
        self, owner_id: str, daily_date: str, daily_start_equity: float, peak_equity: float,
    ) -> dict[str, Any]:
        owner_id = _required_text(owner_id, "owner_id")
        daily_date = _required_text(daily_date, "daily_date", 10)
        for value in (daily_start_equity, peak_equity):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError("live account risk state must contain positive finite equity values")
        now = _utc_now()
        with self._connection() as conn:
            cursor = conn.execute(
                "UPDATE live_execution_profiles SET daily_date=?,daily_start_equity=?,peak_equity=?,updated_at=? WHERE owner_id=?",
                (daily_date, float(daily_start_equity), float(peak_equity), now, owner_id),
            )
            if cursor.rowcount != 1:
                raise LookupError("live execution profile not found")
        return self.get_live_execution_profile(owner_id)

    def get_live_execution_state(
        self, owner_id: str, venue: str, symbol: str, timeframe: str
    ) -> dict[str, Any]:
        owner_id = _required_text(owner_id, "owner_id")
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM live_execution_states WHERE owner_id=? AND venue=? AND symbol=? AND timeframe=?",
                (owner_id, venue, symbol, timeframe),
            ).fetchone()
        if row is None:
            return {"owner_id": owner_id, "venue": venue, "symbol": symbol,
                    "timeframe": timeframe, "last_processed_candle": None,
                    "position_qty": 0.0, "entry_price": None, "daily_date": None,
                    "daily_start_equity": 0.0, "peak_equity": 0.0}
        return dict(row)

    def apply_live_execution_state(
        self, owner_id: str, venue: str, symbol: str, timeframe: str,
        candle_at: str, state: dict[str, Any],
    ) -> bool:
        owner_id = _required_text(owner_id, "owner_id")
        candle_at = _required_text(candle_at, "candle_at", 80)
        required = ("position_qty", "entry_price", "daily_date", "daily_start_equity", "peak_equity")
        if not isinstance(state, dict) or any(key not in state for key in required):
            raise ValueError("live execution state is incomplete")
        for key in ("position_qty", "daily_start_equity", "peak_equity"):
            value = state[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("live execution state contains invalid numeric values")
        if state["entry_price"] is not None and (
            isinstance(state["entry_price"], bool) or not isinstance(state["entry_price"], (int, float))
            or not math.isfinite(state["entry_price"]) or state["entry_price"] <= 0
        ):
            raise ValueError("live execution entry price is invalid")
        now = _utc_now()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT last_processed_candle FROM live_execution_states WHERE owner_id=? AND venue=? AND symbol=? AND timeframe=?",
                (owner_id, venue, symbol, timeframe),
            ).fetchone()
            if row and row["last_processed_candle"] and candle_at <= row["last_processed_candle"]:
                return False
            conn.execute(
                """INSERT INTO live_execution_states
                (owner_id,venue,symbol,timeframe,last_processed_candle,position_qty,entry_price,
                 daily_date,daily_start_equity,peak_equity,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(owner_id,venue,symbol,timeframe) DO UPDATE SET
                last_processed_candle=excluded.last_processed_candle,
                position_qty=excluded.position_qty,entry_price=excluded.entry_price,
                daily_date=excluded.daily_date,daily_start_equity=excluded.daily_start_equity,
                peak_equity=excluded.peak_equity,updated_at=excluded.updated_at""",
                (owner_id, venue, symbol, timeframe, candle_at, float(state["position_qty"]),
                 state["entry_price"], state["daily_date"], float(state["daily_start_equity"]),
                 float(state["peak_equity"]), now),
            )
        return True

    def create_live_order_intent(
        self, owner_id: str, venue: str, symbol: str, side: str,
        amount: float, client_order_id: str,
    ) -> tuple[dict[str, Any], bool]:
        owner_id = _required_text(owner_id, "owner_id")
        if venue not in {"binance", "mexc"} or side not in {"buy", "sell"}:
            raise ValueError("invalid live order venue or side")
        if (isinstance(amount, bool) or not isinstance(amount, (int, float))
                or not math.isfinite(amount) or amount <= 0):
            raise ValueError("live order amount must be positive and finite")
        client_order_id = _required_text(client_order_id, "client_order_id", 40)
        now, order_id = _utc_now(), str(uuid4())
        created = False
        with self._connection() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO live_orders
                (id,owner_id,venue,symbol,side,amount,client_order_id,exchange_order_id,status,
                 filled,average,cost,result_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,NULL,'submitting',0,NULL,NULL,'{}',?,?)""",
                (order_id, owner_id, venue, symbol, side, float(amount), client_order_id, now, now),
            )
            row = conn.execute(
                "SELECT * FROM live_orders WHERE owner_id=? AND client_order_id=?",
                (owner_id, client_order_id),
            ).fetchone()
            if row is None:
                raise ValueError("client order id collision")
            created = row["id"] == order_id
        return self._live_order_dict(row), created

    @staticmethod
    def _live_order_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "owner_id": row["owner_id"], "venue": row["venue"],
            "symbol": row["symbol"], "side": row["side"], "amount": row["amount"],
            "client_order_id": row["client_order_id"], "exchange_order_id": row["exchange_order_id"],
            "status": row["status"], "filled": row["filled"], "average": row["average"],
            "cost": row["cost"], "result": json.loads(row["result_json"]),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    def update_live_order(
        self, owner_id: str, client_order_id: str, status: str,
        exchange_order_id: str | None = None, filled: float = 0.0,
        average: float | None = None, cost: float | None = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        owner_id = _required_text(owner_id, "owner_id")
        client_order_id = _required_text(client_order_id, "client_order_id", 40)
        if status not in {"submitting", "open", "closed", "canceled", "rejected", "unknown"}:
            raise ValueError("invalid live order status")
        for value in (filled, average, cost):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError("live order result contains invalid numeric values")
        now = _utc_now()
        with self._connection() as conn:
            cursor = conn.execute(
                """UPDATE live_orders SET status=?,exchange_order_id=COALESCE(?,exchange_order_id),
                filled=?,average=?,cost=?,result_json=?,updated_at=?
                WHERE owner_id=? AND client_order_id=?""",
                (status, exchange_order_id, float(filled), average, cost,
                 _encode_json(result or {}), now, owner_id, client_order_id),
            )
            if cursor.rowcount != 1:
                raise LookupError("live order not found")
            row = conn.execute(
                "SELECT * FROM live_orders WHERE owner_id=? AND client_order_id=?",
                (owner_id, client_order_id),
            ).fetchone()
        return self._live_order_dict(row)

    def get_live_order(self, owner_id: str, client_order_id: str) -> dict[str, Any] | None:
        owner_id = _required_text(owner_id, "owner_id")
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM live_orders WHERE owner_id=? AND client_order_id=?",
                (owner_id, client_order_id),
            ).fetchone()
        return self._live_order_dict(row) if row else None

    def list_live_orders(self, owner_id: str, limit: int = 100) -> list[dict[str, Any]]:
        owner_id = _required_text(owner_id, "owner_id")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM live_orders WHERE owner_id=? ORDER BY created_at DESC LIMIT ?",
                (owner_id, limit),
            ).fetchall()
        return [self._live_order_dict(row) for row in rows]



    def schedule_solana_research_refresh(
        self, owner_id: str, workspace_id: str, mint: str, pool_address: str,
        interval_seconds: int = 300,
    ) -> dict[str, Any]:
        """Create or refresh an owner-scoped recurring scan for a known pool."""
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        mint = _required_text(mint, "mint", 48)
        pool_address = _required_text(pool_address, "pool_address", 128)
        if len(mint) < 32:
            raise ValueError("mint must be a valid-length Solana address")
        if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, int) or not 300 <= interval_seconds <= 86400:
            raise ValueError("interval_seconds must be between 300 and 86400")
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat(timespec="microseconds").replace("+00:00", "Z")
        due = (now_dt + timedelta(seconds=interval_seconds)).isoformat(timespec="microseconds").replace("+00:00", "Z")
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            conn.execute(
                "INSERT INTO solana_research_refresh_jobs(id,owner_id,workspace_id,mint,pool_address,"
                "interval_seconds,next_due_at,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(owner_id,workspace_id,mint,pool_address) DO UPDATE SET "
                "interval_seconds=excluded.interval_seconds,next_due_at=excluded.next_due_at,enabled=1,updated_at=excluded.updated_at",
                (str(uuid4()), owner_id, workspace_id, mint, pool_address,
                 interval_seconds, due, now, now),
            )
            row = conn.execute(
                "SELECT * FROM solana_research_refresh_jobs WHERE owner_id=? AND workspace_id=? AND mint=? AND pool_address=?",
                (owner_id, workspace_id, mint, pool_address),
            ).fetchone()
        return dict(row)

    def list_solana_research_refresh_jobs(
        self, owner_id: str, workspace_id: str, mint: str | None = None,
    ) -> list[dict[str, Any]]:
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        query = (
            "SELECT id,mint,pool_address,interval_seconds,next_due_at,last_enqueued_at,"
            "enabled,created_at,updated_at FROM solana_research_refresh_jobs "
            "WHERE owner_id=? AND workspace_id=?"
        )
        params: list[Any] = [owner_id, workspace_id]
        if mint is not None:
            query += " AND mint=?"
            params.append(_required_text(mint, "mint", 48))
        query += " ORDER BY next_due_at,id"
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            rows = conn.execute(query, params).fetchall()
        return [
            {**dict(row), "status": "awaiting_pool" if row["pool_address"] == "__pending__"
             else "scheduled", "enabled": bool(row["enabled"])}
            for row in rows
        ]

    def enqueue_due_solana_research_refreshes(self, limit: int = 100) -> int:
        """Atomically enqueue due recurring scans; repeated worker polls are idempotent."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be from 1 to 500")
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat(timespec="microseconds").replace("+00:00", "Z")
        created = 0
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT * FROM solana_research_refresh_jobs WHERE enabled=1 AND next_due_at<=? "
                "ORDER BY next_due_at,id LIMIT ?", (now, limit)
            ).fetchall()
            for row in rows:
                task_id = str(uuid4())
                payload = {
                    "mint": row["mint"], "pool_address": row["pool_address"],
                    "scheduled_by": "eti.solana.refresh",
                    "refresh_job_id": row["id"],
                }
                conn.execute(
                    "INSERT INTO agent_tasks(id,owner_id,workspace_id,requester,target_capability,"
                    "status,input_json,max_attempts,created_at,updated_at) "
                    "VALUES(?,?,?,'eti.solana.refresh','solana.market.scan','queued',?,3,?,?)",
                    (task_id, row["owner_id"], row["workspace_id"], _encode_json(payload), now, now),
                )
                next_due = (now_dt + timedelta(seconds=row["interval_seconds"])).isoformat(
                    timespec="microseconds"
                ).replace("+00:00", "Z")
                conn.execute(
                    "UPDATE solana_research_refresh_jobs SET last_enqueued_at=?,next_due_at=?,updated_at=? "
                    "WHERE id=? AND next_due_at<=?",
                    (now, next_due, now, row["id"], now),
                )
                conn.execute(
                    "INSERT INTO audit_events(id,owner_id,workspace_id,event_type,payload_json,created_at) "
                    "VALUES(?,?,?,'solana.refresh.queued',?,?)",
                    (str(uuid4()), row["owner_id"], row["workspace_id"],
                     _encode_json({"task_id": task_id, "refresh_job_id": row["id"],
                                   "mint": row["mint"], "pool_address": row["pool_address"]}), now),
                )
                created += 1
        return created

    def generate_solana_research_labels(
        self, owner_id: str, workspace_id: str, now: datetime | None = None,
    ) -> dict[str, int]:
        """Persist only labels supported by later same-pool observations."""
        from .solana_labels import build_forward_labels
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            rows = conn.execute(
                "SELECT id,capability,mint,pool_address,schema_version,payload_json,fetched_at "
                "FROM solana_research_snapshots WHERE owner_id=? AND workspace_id=? "
                "ORDER BY fetched_at,id", (owner_id, workspace_id)
            ).fetchall()
            snapshots = []
            skipped_invalid_snapshots = 0
            for row in rows:
                # Label generation must not crash on damaged stored JSON or use
                # payloads whose schema has not been explicitly understood.
                if isinstance(row["schema_version"], bool) or row["schema_version"] != 1:
                    skipped_invalid_snapshots += 1
                    continue
                try:
                    payload = json.loads(row["payload_json"])
                except (TypeError, ValueError):
                    skipped_invalid_snapshots += 1
                    continue
                if not isinstance(payload, dict):
                    skipped_invalid_snapshots += 1
                    continue
                snapshots.append({
                    "id": row["id"], "capability": row["capability"], "mint": row["mint"],
                    "pool_address": row["pool_address"], "payload": payload,
                    "fetched_at": row["fetched_at"],
                })
            labels = build_forward_labels(snapshots, now=now)
            inserted = 0
            created_at = _utc_now()
            for label in labels:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO solana_research_labels(id,owner_id,workspace_id,mint,"
                    "pool_address,baseline_snapshot_id,target_snapshot_id,horizon_seconds,label_json,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (str(uuid4()), owner_id, workspace_id, label["mint"], label["pool_address"],
                     label["baseline_snapshot_id"], label["target_snapshot_id"],
                     label["horizon_seconds"], _encode_json(label), created_at),
                )
                inserted += cursor.rowcount
        return {
            "generated": inserted, "available": len(labels),
            "skipped_invalid_snapshots": skipped_invalid_snapshots,
        }

    def get_solana_research_dataset_report(
        self, owner_id: str, workspace_id: str,
    ) -> dict[str, Any]:
        """Return owner-scoped coverage diagnostics; never declares a model train-ready."""
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            snapshot_rows = conn.execute(
                "SELECT capability,mint,pool_address,fetched_at,payload_json "
                "FROM solana_research_snapshots WHERE owner_id=? AND workspace_id=? "
                "ORDER BY fetched_at,id", (owner_id, workspace_id)
            )
            snapshot_count = market_count = audit_count = 0
            mints: set[str] = set()
            pools: set[tuple[str, str]] = set()
            groups: dict[tuple[str, str], list[datetime]] = {}
            feature_counts = {
                "price_usd": 0, "liquidity_usd": 0, "volume_h24_usd": 0,
                "txns_h1_buys": 0, "txns_h1_sells": 0,
            }
            first_at: datetime | None = None
            last_at: datetime | None = None
            malformed_payloads = 0
            for row in snapshot_rows:
                snapshot_count += 1
                if row["capability"] == "solana.token.audit":
                    audit_count += 1
                    continue
                if row["capability"] != "solana.market.scan":
                    continue
                market_count += 1
                mint, pool = row["mint"], row["pool_address"]
                if isinstance(mint, str) and mint:
                    mints.add(mint)
                if isinstance(mint, str) and mint and isinstance(pool, str) and pool and pool != "__pending__":
                    pools.add((mint, pool))
                    try:
                        stamp = datetime.fromisoformat(row["fetched_at"].replace("Z", "+00:00"))
                        if stamp.tzinfo is not None:
                            stamp = stamp.astimezone(timezone.utc)
                            groups.setdefault((mint, pool), []).append(stamp)
                            first_at = stamp if first_at is None or stamp < first_at else first_at
                            last_at = stamp if last_at is None or stamp > last_at else last_at
                    except (AttributeError, TypeError, ValueError):
                        pass
                try:
                    payload = json.loads(row["payload_json"])
                except (TypeError, ValueError):
                    malformed_payloads += 1
                    continue
                evidence = payload.get("evidence") if isinstance(payload, dict) else None
                if not isinstance(evidence, dict):
                    continue
                for key in feature_counts:
                    value = evidence.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                        feature_counts[key] += 1
            label_rows = conn.execute(
                "SELECT horizon_seconds,label_json FROM solana_research_labels "
                "WHERE owner_id=? AND workspace_id=? ORDER BY horizon_seconds",
                (owner_id, workspace_id),
            )
            label_counts: dict[str, dict[str, int]] = {}
            for row in label_rows:
                key = str(row["horizon_seconds"])
                stats = label_counts.setdefault(key, {"labels": 0, "distinct_mints": 0, "price_returns_available": 0})
                stats["labels"] += 1
                try:
                    label = json.loads(row["label_json"])
                except (TypeError, ValueError):
                    continue
                if isinstance(label, dict):
                    if (
                        isinstance(label.get("sampled_price_return_pct"), (int, float))
                        and not isinstance(label.get("sampled_price_return_pct"), bool)
                        and math.isfinite(label["sampled_price_return_pct"])
                    ):
                        stats["price_returns_available"] += 1
            # Count distinct mints with SQL, avoiding a potentially large in-memory set.
            distinct_by_horizon = conn.execute(
                "SELECT horizon_seconds,COUNT(DISTINCT mint) AS n FROM solana_research_labels "
                "WHERE owner_id=? AND workspace_id=? GROUP BY horizon_seconds",
                (owner_id, workspace_id),
            ).fetchall()
            for row in distinct_by_horizon:
                label_counts.setdefault(str(row["horizon_seconds"]), {
                    "labels": 0, "distinct_mints": 0, "price_returns_available": 0,
                })["distinct_mints"] = row["n"]
        intervals: list[float] = []
        gaps_over_10m = 0
        for stamps in groups.values():
            stamps.sort()
            for previous, current in zip(stamps, stamps[1:]):
                seconds = (current - previous).total_seconds()
                if seconds > 0:
                    intervals.append(seconds)
                    if seconds > 600:
                        gaps_over_10m += 1
        intervals.sort()
        median_interval = (
            intervals[len(intervals) // 2] if intervals else None
        )
        coverage = {
            key: round(count / market_count, 6) if market_count else 0.0
            for key, count in feature_counts.items()
        }
        return {
            "scope": {"owner_id": owner_id, "workspace_id": workspace_id},
            "snapshots": {
                "total": snapshot_count, "market_scans": market_count,
                "token_audits": audit_count, "distinct_mints": len(mints),
                "distinct_mint_pool_pairs": len(pools),
                "first_observation_at": first_at.isoformat().replace("+00:00", "Z") if first_at else None,
                "last_observation_at": last_at.isoformat().replace("+00:00", "Z") if last_at else None,
                "observation_span_seconds": round((last_at - first_at).total_seconds(), 3) if first_at and last_at else 0,
                "median_sample_interval_seconds": median_interval,
                "sample_gaps_over_10_minutes": gaps_over_10m,
                "malformed_payloads": malformed_payloads,
            },
            "feature_coverage": {
                key: {"count": feature_counts[key], "fraction": coverage[key]}
                for key in feature_counts
            },
            "labels_by_horizon_seconds": label_counts,
            "training_gate": {
                "ready": False,
                "status": "RESEARCH_DATA_COLLECTION",
                "reason": "No independently approved sample-size, coverage, regime-diversity, execution-cost, and out-of-sample acceptance thresholds have been configured.",
            },
            "limitations": [
                "Provider samples are not OHLCV candles or executable quotes.",
                "Reported rolling 24-hour volume is not interval volume.",
                "Missing observations during outages are not backfilled.",
                "A dataset report is descriptive and does not establish predictive performance.",
            ],
        }

    def record_solana_execution_cost_evidence(
        self, owner_id: str, workspace_id: str, label_id: str,
        mint: str, pool_address: str, evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Append immutable owner/workspace-scoped cost evidence for one exact label."""
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        label_id = _required_text(label_id, "label_id", 128)
        mint = _required_text(mint, "mint", 64)
        pool_address = _required_text(pool_address, "pool_address", 128)
        if len(mint) < 32:
            raise ValueError("mint must be a valid-length Solana address")
        if not isinstance(evidence, dict):
            raise ValueError("evidence must be an object")
        # Identity is stored in the envelope, not trusted from nested component values.
        envelope = dict(evidence)
        for key, expected in (("label_id", label_id), ("mint", mint), ("pool_address", pool_address)):
            supplied = envelope.get(key)
            if supplied != expected:
                raise ValueError(f"evidence {key} must match the requested label identity")
        encoded = _encode_json(envelope)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        now = _utc_now()
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            label = conn.execute(
                "SELECT mint,pool_address FROM solana_research_labels "
                "WHERE owner_id=? AND workspace_id=? AND id=?",
                (owner_id, workspace_id, label_id),
            ).fetchone()
            if label is None or label["mint"] != mint or label["pool_address"] != pool_address:
                raise LookupError("label not found")
            existing = conn.execute(
                "SELECT * FROM solana_execution_cost_evidence "
                "WHERE owner_id=? AND workspace_id=? AND label_id=? AND evidence_sha256=?",
                (owner_id, workspace_id, label_id, digest),
            ).fetchone()
            if existing is not None:
                return self._solana_cost_evidence_dict(existing, duplicate=True)
            record_id = str(uuid4())
            conn.execute(
                "INSERT INTO solana_execution_cost_evidence "
                "(id,owner_id,workspace_id,label_id,mint,pool_address,evidence_json,evidence_sha256,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (record_id, owner_id, workspace_id, label_id, mint, pool_address, encoded, digest, now),
            )
            conn.execute(
                "INSERT INTO audit_events(id,owner_id,workspace_id,event_type,payload_json,created_at) "
                "VALUES(?,?,?,?,?,?)",
                (str(uuid4()), owner_id, workspace_id, "solana.cost_evidence.recorded",
                 _encode_json({"evidence_id": record_id, "label_id": label_id,
                               "mint": mint, "pool_address": pool_address,
                               "evidence_sha256": digest}), now),
            )
            row = conn.execute(
                "SELECT * FROM solana_execution_cost_evidence WHERE id=?", (record_id,)
            ).fetchone()
        return self._solana_cost_evidence_dict(row, duplicate=False)

    @staticmethod
    def _solana_cost_evidence_dict(row: sqlite3.Row, *, duplicate: bool) -> dict[str, Any]:
        raw = row["evidence_json"]
        digest_valid = hashlib.sha256(raw.encode("utf-8")).hexdigest() == row["evidence_sha256"]
        try:
            evidence = json.loads(raw)
        except (TypeError, ValueError):
            evidence = None
            digest_valid = False
        return {
            "id": row["id"], "label_id": row["label_id"], "mint": row["mint"],
            "pool_address": row["pool_address"], "evidence": evidence,
            "evidence_sha256": row["evidence_sha256"], "integrity_valid": digest_valid,
            "created_at": row["created_at"], "duplicate": duplicate,
        }

    def list_solana_execution_cost_evidence(
        self, owner_id: str, workspace_id: str, label_id: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be from 1 to 1000")
        query = (
            "SELECT * FROM solana_execution_cost_evidence "
            "WHERE owner_id=? AND workspace_id=?"
        )
        params: list[Any] = [owner_id, workspace_id]
        if label_id is not None:
            query += " AND label_id=?"
            params.append(_required_text(label_id, "label_id", 128))
        query += " ORDER BY created_at DESC,id DESC LIMIT ?"
        params.append(limit)
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            rows = conn.execute(query, params).fetchall()
        return [self._solana_cost_evidence_dict(row, duplicate=False) for row in rows]

    def list_latest_solana_execution_cost_evidence(
        self, owner_id: str, workspace_id: str, label_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Return one newest immutable evidence record per requested label."""
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        if not isinstance(label_ids, list):
            raise ValueError("label_ids must be a list")
        normalized: list[str] = []
        seen: set[str] = set()
        for label_id in label_ids:
            value = _required_text(label_id, "label_id", 128)
            if value not in seen:
                normalized.append(value)
                seen.add(value)
        if not normalized:
            with self._connection() as conn:
                if not self._owns_workspace(conn, owner_id, workspace_id):
                    raise LookupError("workspace not found")
            return []

        latest: dict[str, dict[str, Any]] = {}
        # Keep each IN clause below SQLite's conservative bound-parameter limit.
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            for offset in range(0, len(normalized), 500):
                batch = normalized[offset:offset + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    "SELECT * FROM solana_execution_cost_evidence "
                    f"WHERE owner_id=? AND workspace_id=? AND label_id IN ({placeholders}) "
                    "ORDER BY label_id ASC,created_at DESC,id DESC",
                    [owner_id, workspace_id, *batch],
                ).fetchall()
                for row in rows:
                    if row["label_id"] not in latest:
                        latest[row["label_id"]] = self._solana_cost_evidence_dict(
                            row, duplicate=False
                        )
        return [latest[label_id] for label_id in normalized if label_id in latest]

    def record_solana_transaction_attestation(
        self, owner_id: str, workspace_id: str, label_id: str,
        mint: str, pool_address: str, wallet_address: str,
        attestation: dict[str, Any],
    ) -> dict[str, Any]:
        """Append an RPC-generated finalized receipt observation; never assert a verified swap."""
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        label_id = _required_text(label_id, "label_id", 128)
        mint = _required_text(mint, "mint", 64)
        pool_address = _required_text(pool_address, "pool_address", 128)
        wallet_address = _required_text(wallet_address, "wallet_address", 64)
        if not isinstance(attestation, dict):
            raise ValueError("attestation must be an object")
        if (attestation.get("transaction_receipt_verified") is not True
                or attestation.get("verified") is not False
                or attestation.get("status") != "FINALIZED_MINT_BALANCE_DEBIT_OBSERVED"):
            raise ValueError("only finalized receipt observations may be attested")
        signature = _required_text(attestation.get("signature"), "signature", 88)
        for key, expected in (
            ("mint", mint), ("expected_pool_address", pool_address),
            ("wallet_address", wallet_address),
        ):
            if attestation.get(key) != expected:
                raise ValueError(f"attestation {key} does not match requested identity")
        encoded = _encode_json(attestation)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        now = _utc_now()
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            label = conn.execute(
                "SELECT mint,pool_address FROM solana_research_labels "
                "WHERE owner_id=? AND workspace_id=? AND id=?",
                (owner_id, workspace_id, label_id),
            ).fetchone()
            if label is None or label["mint"] != mint or label["pool_address"] != pool_address:
                raise LookupError("label not found")
            existing = conn.execute(
                "SELECT * FROM solana_transaction_attestations WHERE owner_id=? AND workspace_id=? "
                "AND label_id=? AND signature=? AND attestation_sha256=?",
                (owner_id, workspace_id, label_id, signature, digest),
            ).fetchone()
            if existing is not None:
                return self._solana_transaction_attestation_dict(existing, duplicate=True)
            record_id = str(uuid4())
            conn.execute(
                "INSERT INTO solana_transaction_attestations "
                "(id,owner_id,workspace_id,label_id,signature,mint,pool_address,wallet_address,"
                "attestation_json,attestation_sha256,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (record_id, owner_id, workspace_id, label_id, signature, mint, pool_address,
                 wallet_address, encoded, digest, now),
            )
            conn.execute(
                "INSERT INTO audit_events(id,owner_id,workspace_id,event_type,payload_json,created_at) "
                "VALUES(?,?,?,?,?,?)",
                (str(uuid4()), owner_id, workspace_id, "solana.transaction_receipt.attested",
                 _encode_json({"attestation_id": record_id, "label_id": label_id,
                               "signature": signature, "attestation_sha256": digest}), now),
            )
            row = conn.execute(
                "SELECT * FROM solana_transaction_attestations WHERE id=?", (record_id,)
            ).fetchone()
        return self._solana_transaction_attestation_dict(row, duplicate=False)

    @staticmethod
    def _solana_transaction_attestation_dict(row: sqlite3.Row, *, duplicate: bool) -> dict[str, Any]:
        raw = row["attestation_json"]
        integrity_valid = hashlib.sha256(raw.encode("utf-8")).hexdigest() == row["attestation_sha256"]
        try:
            attestation = json.loads(raw)
        except (TypeError, ValueError):
            attestation = None
            integrity_valid = False
        return {
            "id": row["id"], "label_id": row["label_id"], "signature": row["signature"],
            "mint": row["mint"], "pool_address": row["pool_address"],
            "wallet_address": row["wallet_address"], "attestation": attestation,
            "attestation_sha256": row["attestation_sha256"], "integrity_valid": integrity_valid,
            "created_at": row["created_at"], "duplicate": duplicate,
        }

    def list_solana_transaction_attestations(
        self, owner_id: str, workspace_id: str, label_id: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be from 1 to 1000")
        query = ("SELECT * FROM solana_transaction_attestations "
                 "WHERE owner_id=? AND workspace_id=?")
        params: list[Any] = [owner_id, workspace_id]
        if label_id is not None:
            query += " AND label_id=?"
            params.append(_required_text(label_id, "label_id", 128))
        query += " ORDER BY created_at DESC,id DESC LIMIT ?"
        params.append(limit)
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            rows = conn.execute(query, params).fetchall()
        return [self._solana_transaction_attestation_dict(row, duplicate=False) for row in rows]

    def list_solana_research_labels(
        self, owner_id: str, workspace_id: str, mint: str | None = None, limit: int = 500,
    ) -> list[dict[str, Any]]:
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be from 1 to 1000")
        query = "SELECT * FROM solana_research_labels WHERE owner_id=? AND workspace_id=?"
        params: list[Any] = [owner_id, workspace_id]
        if mint is not None:
            query += " AND mint=?"
            params.append(_required_text(mint, "mint", 64))
        query += " ORDER BY created_at DESC,id DESC LIMIT ?"
        params.append(limit)
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            rows = conn.execute(query, params).fetchall()
        return [{"id": row["id"], "label": json.loads(row["label_json"]),
                 "created_at": row["created_at"]} for row in rows]

    def record_solana_research_event(
        self, owner_id: str, workspace_id: str, event_key: str,
        source: str, event_type: str, mint: str, payload: Any,
        source_timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Persist an idempotent, account-scoped Solana discovery event."""
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        event_key = _required_text(event_key, "event_key", 200)
        source = _required_text(source, "source", 40)
        mint = _required_text(mint, "mint", 64)
        if source != "pumpportal":
            raise ValueError("source must be pumpportal")
        if event_type not in {"new_token", "migration"}:
            raise ValueError("event_type must be new_token or migration")
        if len(mint) < 32:
            raise ValueError("mint must be a valid-length Solana address")
        if source_timestamp is not None:
            source_timestamp = _required_text(source_timestamp, "source_timestamp", 80)
        encoded = _encode_json(payload)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        now = _utc_now()
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            existing = conn.execute(
                "SELECT * FROM solana_research_events WHERE owner_id=? AND workspace_id=? AND event_key=?",
                (owner_id, workspace_id, event_key),
            ).fetchone()
            if existing is not None:
                return self._solana_event_dict(
                    existing, duplicate=True,
                    payload_conflict=existing["payload_sha256"] != digest,
                )
            event_id = str(uuid4())
            conn.execute(
                "INSERT INTO solana_research_events(id,owner_id,workspace_id,event_key,source,event_type,"
                "mint,source_timestamp,payload_json,payload_sha256,received_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (event_id, owner_id, workspace_id, event_key, source, event_type, mint,
                 source_timestamp, encoded, digest, now),
            )
            conn.execute(
                "INSERT INTO audit_events(id,owner_id,workspace_id,event_type,payload_json,created_at) "
                "VALUES(?,?,?,?,?,?)",
                (str(uuid4()), owner_id, workspace_id, "solana.event.ingested",
                 _encode_json({"event_id": event_id, "event_key": event_key,
                               "event_type": event_type, "mint": mint,
                               "payload_sha256": digest}), now),
            )
            # The event and its first research task commit together. A replay
            # returns above and cannot create a second scan task.
            scan_task_id = str(uuid4())
            scan_payload = {
                "mint": mint,
                "source_event_id": event_id,
                "source_event_key": event_key,
                "source_event_type": event_type,
                "scheduled_by": "eti.solana.ingest",
            }
            conn.execute(
                "INSERT INTO agent_tasks(id,owner_id,workspace_id,requester,target_capability,"
                "status,input_json,max_attempts,created_at,updated_at) "
                "VALUES(?,?,?,?,?,'queued',?,3,?,?)",
                (scan_task_id, owner_id, workspace_id, "eti.solana.ingest",
                 "solana.market.scan", _encode_json(scan_payload), now, now),
            )
            conn.execute(
                "INSERT INTO audit_events(id,owner_id,workspace_id,event_type,payload_json,created_at) "
                "VALUES(?,?,?,?,?,?)",
                (str(uuid4()), owner_id, workspace_id, "solana.scan.queued",
                 _encode_json({"event_id": event_id, "task_id": scan_task_id, "mint": mint}), now),
            )
            row = conn.execute(
                "SELECT * FROM solana_research_events WHERE id=?", (event_id,)
            ).fetchone()
        result = self._solana_event_dict(row, duplicate=False)
        result["scan_task_id"] = scan_task_id
        return result

    @staticmethod
    def _solana_event_dict(
        row: sqlite3.Row, duplicate: bool = False, payload_conflict: bool = False
    ) -> dict[str, Any]:
        return {
            "id": row["id"], "event_key": row["event_key"], "source": row["source"],
            "event_type": row["event_type"], "mint": row["mint"],
            "source_timestamp": row["source_timestamp"],
            "payload": json.loads(row["payload_json"]),
            "payload_sha256": row["payload_sha256"], "received_at": row["received_at"],
            "duplicate": duplicate, "payload_conflict": payload_conflict,
        }

    def list_solana_research_snapshots(
        self, owner_id: str, workspace_id: str, limit: int = 100,
        mint: str | None = None,
    ) -> list[dict[str, Any]]:
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be an integer from 1 to 500")
        query = ("SELECT * FROM solana_research_snapshots "
                 "WHERE owner_id=? AND workspace_id=?")
        params: list[Any] = [owner_id, workspace_id]
        if mint is not None:
            mint = _required_text(mint, "mint", 64)
            query += " AND mint=?"
            params.append(mint)
        query += " ORDER BY fetched_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            rows = conn.execute(query, params).fetchall()
        return [{
            "id": row["id"], "task_id": row["task_id"],
            "capability": row["capability"], "mint": row["mint"],
            "pool_address": row["pool_address"],
            "schema_version": row["schema_version"],
            "payload": json.loads(row["payload_json"]),
            "payload_sha256": row["payload_sha256"],
            "fetched_at": row["fetched_at"],
        } for row in rows]

    def list_solana_research_events(
        self, owner_id: str, workspace_id: str, limit: int = 100,
        mint: str | None = None,
    ) -> list[dict[str, Any]]:
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be an integer from 1 to 500")
        query = ("SELECT * FROM solana_research_events "
                 "WHERE owner_id=? AND workspace_id=?")
        params: list[Any] = [owner_id, workspace_id]
        if mint is not None:
            mint = _required_text(mint, "mint", 64)
            query += " AND mint=?"
            params.append(mint)
        query += " ORDER BY received_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            rows = conn.execute(query, params).fetchall()
        return [self._solana_event_dict(row) for row in rows]

    def export_solana_research_dataset(
        self, owner_id: str, workspace_id: str, limit: int = 5000,
        mint: str | None = None,
    ) -> dict[str, Any]:
        """Return a deterministic, bounded point-in-time export for this owner/workspace."""
        owner_id = _required_text(owner_id, "owner_id")
        workspace_id = _required_text(workspace_id, "workspace_id")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5000:
            raise ValueError("limit must be an integer from 1 to 5000")
        query = (
            "SELECT id,task_id,capability,mint,pool_address,schema_version,"
            "payload_json,payload_sha256,fetched_at FROM solana_research_snapshots "
            "WHERE owner_id=? AND workspace_id=?"
        )
        params: list[Any] = [owner_id, workspace_id]
        if mint is not None:
            mint = _required_text(mint, "mint", 64)
            query += " AND mint=?"
            params.append(mint)
        query += " ORDER BY fetched_at ASC,id ASC LIMIT ?"
        params.append(limit + 1)
        with self._connection() as conn:
            if not self._owns_workspace(conn, owner_id, workspace_id):
                raise LookupError("workspace not found")
            rows = conn.execute(query, params).fetchall()
            truncated = len(rows) > limit
            rows = rows[:limit]
            label_query = (
                "SELECT l.id,l.mint,l.pool_address,l.baseline_snapshot_id,l.target_snapshot_id,"
                "l.horizon_seconds,l.label_json,l.created_at FROM solana_research_labels l "
                "JOIN (SELECT id FROM solana_research_snapshots WHERE owner_id=? AND workspace_id=?"
            )
            label_params: list[Any] = [owner_id, workspace_id]
            if mint is not None:
                label_query += " AND mint=?"
                label_params.append(mint)
            label_query += (
                " ORDER BY fetched_at ASC,id ASC LIMIT ?) selected "
                "ON selected.id=l.baseline_snapshot_id "
                "WHERE l.owner_id=? AND l.workspace_id=? ORDER BY l.created_at ASC,l.id ASC"
            )
            label_params.extend([limit, owner_id, workspace_id])
            label_rows = conn.execute(label_query, label_params).fetchall()
            exported_snapshot_ids = {row["id"] for row in rows}
            baseline_rows = {row["id"]: row for row in rows}
            target_rows = {}
            target_ids = list(dict.fromkeys(row["target_snapshot_id"] for row in label_rows))
            for offset in range(0, len(target_ids), 500):
                batch = target_ids[offset:offset + 500]
                if not batch:
                    continue
                placeholders = ",".join("?" for _ in batch)
                referenced = conn.execute(
                    "SELECT id,mint,pool_address,fetched_at FROM solana_research_snapshots "
                    "WHERE owner_id=? AND workspace_id=? AND id IN (" + placeholders + ")",
                    [owner_id, workspace_id, *batch],
                ).fetchall()
                target_rows.update({item["id"]: item for item in referenced})
            labels = []
            malformed_label_count = 0
            label_reference_issue_count = 0
            for row in label_rows:
                try:
                    label_value = json.loads(row["label_json"])
                    label_parse_error = False
                except (TypeError, ValueError):
                    label_value = None
                    label_parse_error = True
                    malformed_label_count += 1
                reference_issues = []
                baseline = baseline_rows.get(row["baseline_snapshot_id"])
                target = target_rows.get(row["target_snapshot_id"])
                if baseline is None:
                    reference_issues.append("baseline_snapshot_missing_or_outside_export")
                if target is None:
                    reference_issues.append("target_snapshot_missing_or_outside_workspace")
                for role, snapshot in (("baseline", baseline), ("target", target)):
                    if snapshot is not None and (
                        snapshot["mint"] != row["mint"] or snapshot["pool_address"] != row["pool_address"]
                    ):
                        reference_issues.append(role + "_identity_mismatch")
                if baseline is not None and target is not None:
                    try:
                        baseline_at = datetime.fromisoformat(baseline["fetched_at"].replace("Z", "+00:00"))
                        target_at = datetime.fromisoformat(target["fetched_at"].replace("Z", "+00:00"))
                        if baseline_at.tzinfo is None or target_at.tzinfo is None:
                            raise ValueError("naive timestamp")
                        elapsed = (target_at - baseline_at).total_seconds()
                        if elapsed <= 0:
                            reference_issues.append("target_not_after_baseline")
                        if not isinstance(row["horizon_seconds"], int) or row["horizon_seconds"] <= 0 or elapsed < row["horizon_seconds"]:
                            reference_issues.append("target_before_label_horizon")
                    except (AttributeError, TypeError, ValueError):
                        reference_issues.append("invalid_reference_timestamp")
                if reference_issues:
                    label_reference_issue_count += 1
                labels.append({
                    "id": row["id"], "mint": row["mint"], "pool_address": row["pool_address"],
                    "baseline_snapshot_id": row["baseline_snapshot_id"],
                    "target_snapshot_id": row["target_snapshot_id"],
                    "baseline_snapshot_in_export": row["baseline_snapshot_id"] in exported_snapshot_ids,
                    "target_snapshot_in_export": row["target_snapshot_id"] in exported_snapshot_ids,
                    "horizon_seconds": row["horizon_seconds"],
                    "label": label_value, "label_parse_error": label_parse_error,
                    "reference_issues": reference_issues,
                    "created_at": row["created_at"],
                })
        snapshots = []
        for row in rows:
            try:
                payload_value = json.loads(row["payload_json"])
            except (TypeError, ValueError):
                payload_value = None
            snapshots.append({
                "id": row["id"], "task_id": row["task_id"], "capability": row["capability"],
                "mint": row["mint"], "pool_address": row["pool_address"],
                "schema_version": row["schema_version"], "payload": payload_value,
                "payload_sha256": row["payload_sha256"], "fetched_at": row["fetched_at"],
            })
        from .solana_dataset import validate_snapshot_rows
        validation = validate_snapshot_rows(snapshots)
        validation["malformed_label_count"] = malformed_label_count
        validation["label_reference_issue_count"] = label_reference_issue_count
        return {
            "schema": "eti.solana.research_dataset.v1",
            "scope": {"workspace_id": workspace_id},
            "truncated_at_limit": truncated,
            "snapshot_count": len(snapshots),
            "label_count": len(labels),
            "snapshots": snapshots,
            "labels": labels,
            "validation": validation,
            "limitations": [
                "Observations are provider-sampled snapshots, not OHLCV candles or executable quotes.",
                "Labels reference future observations; target_snapshot_in_export is false when the bounded snapshot export omits the target row.",
                "Use baseline snapshot features only for prediction inputs; target snapshot fields are outcomes and must never leak into features.",
                "This export is research data and does not authorize model training or live execution.",
            ],
        }

