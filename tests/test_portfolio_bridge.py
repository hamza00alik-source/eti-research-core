from datetime import datetime, timedelta, timezone

import pytest

from eti.portfolio_bridge import allocate_from_persisted_snapshot
from eti.storage import MemoryStore
from eti.worker import AgentWorker


def _account(store):
    workspace = store.create_workspace("owner-a", "portfolio bridge")
    account = store.create_paper_portfolio_account(
        "owner-a", workspace["id"], "Main", "USD", 10000,
    )
    return workspace, account


def _snapshot(store, workspace, account, *, as_of=None, cash=8000):
    stamp = as_of or datetime.now(timezone.utc).isoformat()
    return store.record_paper_portfolio_snapshot(
        "owner-a", workspace["id"], account["id"], stamp, 10000, cash,
        [{"symbol": "BTC/USD", "notional": 2000, "correlation_group": "crypto"}],
    )


def test_bridge_derives_equity_and_exposures_from_persisted_snapshot(tmp_path):
    store = MemoryStore(tmp_path / "bridge.sqlite3")
    workspace, account = _account(store)
    snapshot = _snapshot(store, workspace, account)
    result = allocate_from_persisted_snapshot(
        store, "owner-a", workspace["id"],
        {"portfolio_account_id": account["id"], "max_snapshot_age_seconds": 300,
         "candidates": []},
    )
    assert result["equity"] == 10000
    assert result["portfolio_provenance"]["snapshot_id"] == snapshot["id"]
    assert result["portfolio_provenance"]["exchange_verified"] is False
    assert result["paper_only"] is True
    assert result["execution_enabled"] is False
    assert result["orders_submitted"] is False


def test_bridge_rejects_stale_and_future_snapshots(tmp_path):
    store = MemoryStore(tmp_path / "stale.sqlite3")
    workspace, account = _account(store)
    now = datetime.now(timezone.utc)
    _snapshot(store, workspace, account, as_of=now - timedelta(hours=2))
    with pytest.raises(ValueError, match="stale"):
        allocate_from_persisted_snapshot(
            store, "owner-a", workspace["id"],
            {"portfolio_account_id": account["id"], "max_snapshot_age_seconds": 300,
             "candidates": []}, now=now,
        )


def test_bridge_rejects_future_snapshot(tmp_path):
    store = MemoryStore(tmp_path / "future.sqlite3")
    workspace, account = _account(store)
    now = datetime.now(timezone.utc)
    _snapshot(store, workspace, account, as_of=now + timedelta(minutes=1))
    with pytest.raises(ValueError, match="future"):
        allocate_from_persisted_snapshot(
            store, "owner-a", workspace["id"],
            {"portfolio_account_id": account["id"], "max_snapshot_age_seconds": 300,
             "candidates": []}, now=now,
        )


def test_bridge_enforces_owner_workspace_scope(tmp_path):
    store = MemoryStore(tmp_path / "scope.sqlite3")
    workspace, account = _account(store)
    _snapshot(store, workspace, account)
    with pytest.raises(LookupError, match="not found"):
        allocate_from_persisted_snapshot(
            store, "owner-b", workspace["id"],
            {"portfolio_account_id": account["id"], "max_snapshot_age_seconds": 300,
             "candidates": []},
        )


def test_worker_injects_task_owner_and_workspace_for_allocation(tmp_path):
    store = MemoryStore(tmp_path / "worker-bridge.sqlite3")
    workspace, account = _account(store)
    _snapshot(store, workspace, account)
    task = store.create_agent_task(
        "owner-a", workspace["id"], "owner-a", "portfolio.allocate_from_account",
        {"portfolio_account_id": account["id"], "max_snapshot_age_seconds": 300,
         "candidates": [], "__owner_id": "attacker", "__workspace_id": "attacker"},
    )
    result = AgentWorker(store).run_once()
    assert result["id"] == task["id"]
    assert result["status"] == "completed"
    assert result["result"]["portfolio_provenance"]["account_id"] == account["id"]
    assert result["result"]["portfolio_provenance"]["snapshot_source"] == "owner_attested"
    assert result["result"]["execution_enabled"] is False
