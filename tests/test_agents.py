import pytest

from eti.agents import AgentCapabilityRegistry, market_snapshot
from eti.storage import MemoryStore
from eti.worker import AgentWorker


def test_worker_executes_only_registered_capability_and_persists_message(tmp_path):
    store = MemoryStore(tmp_path / "agent-worker.sqlite3")
    workspace = store.create_workspace("owner", "research")
    task = store.create_agent_task(
        "owner", workspace["id"], "user", "test.echo", {"symbol": "BTC/USDT"}
    )
    registry = AgentCapabilityRegistry()
    registry.register("test.echo", lambda payload: {"received": payload})
    worker = AgentWorker(store, registry)
    result = worker.run_once()

    assert result["id"] == task["id"]
    assert result["status"] == "completed"
    assert result["attempt_count"] == 1
    assert result["result"] == {"received": {"symbol": "BTC/USDT"}}
    messages = store.list_agent_messages("owner", workspace["id"], task["id"])
    assert len(messages) == 1
    assert messages[0]["message_type"] == "task.completed"
    assert messages[0]["payload"]["status"] == "completed"


def test_unknown_capability_fails_closed_without_executing_user_input(tmp_path):
    store = MemoryStore(tmp_path / "agent-worker.sqlite3")
    workspace = store.create_workspace("owner", "research")
    task = store.create_agent_task(
        "owner", workspace["id"], "user", "shell.execute",
        {"command": "echo this must never run"}
    )
    result = AgentWorker(store, AgentCapabilityRegistry()).run_once()
    assert result["id"] == task["id"]
    assert result["status"] == "failed"
    assert result["result"]["error"]["type"] == "LookupError"
    assert "unsupported agent capability" in result["result"]["error"]["message"]


def test_market_snapshot_rejects_unsupported_inputs_before_network_access():
    with pytest.raises(ValueError, match="venue"):
        market_snapshot({"venue": "unknown", "symbol": "BTC/USDT"})
    with pytest.raises(ValueError, match="symbol"):
        market_snapshot({"venue": "binance", "symbol": "BTCUSDT"})
    with pytest.raises(ValueError, match="timeframe"):
        market_snapshot({"venue": "binance", "symbol": "BTC/USDT", "timeframe": ["1h"]})
    with pytest.raises(ValueError, match="limit"):
        market_snapshot({"venue": "binance", "symbol": "BTC/USDT", "limit": 10000})


def test_worker_does_not_claim_when_queue_is_empty(tmp_path):
    store = MemoryStore(tmp_path / "empty-worker.sqlite3")
    result = AgentWorker(store, AgentCapabilityRegistry()).run_once()
    assert result is None
