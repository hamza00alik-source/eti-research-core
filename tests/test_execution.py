import base64
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from eti.features import FEATURE_COLUMNS
from eti.storage import MemoryStore


class _LiveFakeModel:
    def has_valid_calibration(self):
        return True

    def predict_proba(self, frame):
        return np.array([[0.10, 0.10, 0.80]])


def _live_store(tmp_path):
    store = MemoryStore(tmp_path / "live.sqlite3")
    store.create_workspace("owner", "live")
    store.register_model_version(
        "eti-binance-btc-usdt-1h", "live-test",
        "artifact://sha256/" + "a" * 64, "a" * 64,
        {"features": FEATURE_COLUMNS}, ["crypto_spot"], "live-test-data",
        {"passed": True}, "calibrated", "trainer",
    )
    store.approve_model_version("eti-binance-btc-usdt-1h", "live-test", "reviewer")
    store.register_model_version(
        "eti-binance-eth-usdt-1h", "live-test",
        "artifact://sha256/" + "b" * 64, "b" * 64,
        {"features": FEATURE_COLUMNS}, ["crypto_spot"], "live-test-data",
        {"passed": True}, "calibrated", "trainer",
    )
    store.approve_model_version("eti-binance-eth-usdt-1h", "live-test", "reviewer")
    store.upsert_exchange_credential("owner", "binance", "encrypted-key", "encrypted-secret", "1234")
    store.save_live_execution_profile(
        "owner", "binance", ["BTC/USDT", "ETH/USDT"], "1h", 1000.0,
        {"max_risk_per_trade": 0.005, "max_position_fraction": 0.10,
         "max_gross_exposure": 0.25, "max_daily_loss": 0.02,
         "max_drawdown": 0.08, "confidence_threshold": 0.60},
    )
    store.authorize_live_execution("owner")
    return store


def test_live_execution_is_disabled_until_owner_authorizes(tmp_path):
    store = MemoryStore(tmp_path / "disabled.sqlite3")
    store.create_workspace("owner", "live")
    store.save_live_execution_profile("owner", "binance", ["BTC/USDT"], "1h", 1000.0, {})
    from eti.execution import run_live_tick
    result = run_live_tick(
        {"owner_id": "owner", "venue": "binance", "symbol": "BTC/USDT", "timeframe": "1h"},
        store,
    )
    assert result["status"] == "disabled"
    assert store.list_live_orders("owner") == []


def test_live_profile_enforces_hard_risk_ceilings(tmp_path):
    import pytest
    store = MemoryStore(tmp_path / "limits.sqlite3")
    store.create_workspace("owner", "live")
    with pytest.raises(ValueError, match="hard live risk ceiling"):
        store.save_live_execution_profile(
            "owner", "binance", ["BTC/USDT"], "1h", 1000.0,
            {"max_risk_per_trade": 0.02},
        )


def test_live_order_intent_is_idempotent_and_owner_scoped(tmp_path):
    store = MemoryStore(tmp_path / "orders.sqlite3")
    first, created = store.create_live_order_intent(
        "owner", "binance", "BTC/USDT", "buy", 0.01, "eti-client-order-001"
    )
    second, created_again = store.create_live_order_intent(
        "owner", "binance", "BTC/USDT", "buy", 0.01, "eti-client-order-001"
    )
    assert created is True
    assert created_again is False
    assert first["id"] == second["id"]
    assert store.list_live_orders("other-owner") == []


def test_authorized_live_tick_places_one_risk_gated_order_and_is_candle_idempotent(
    tmp_path, monkeypatch
):
    import eti.execution as execution
    store = _live_store(tmp_path)
    calls = {"orders": 0}
    now = pd.Timestamp.now(tz="UTC").floor("h")
    timestamps = [int((now - pd.Timedelta(hours=80 - i)).timestamp() * 1000) for i in range(80)]
    rows = [[ts, 100.0, 102.0, 99.0, 100.0, 1000.0] for ts in timestamps]

    class FakeExchange:
        def __init__(self, config):
            assert config["apiKey"] == "private-key"
            assert config["secret"] == "private-secret"
        def fetch_balance(self):
            return {"free": {"USDT": 1000.0, "BTC": 0.0},
                    "total": {"USDT": 1000.0, "BTC": 0.0}}
        def fetch_ohlcv(self, symbol, timeframe, limit):
            return rows
        def parse_timeframe(self, timeframe):
            return 3600
        def fetch_ticker(self, symbol):
            return {"last": 100.0}
        def load_markets(self):
            return None
        def amount_to_precision(self, symbol, amount):
            return f"{amount:.6f}"
        def market(self, symbol):
            return {"limits": {"cost": {"min": 5.0}}}
        def create_order(self, symbol, order_type, side, amount, price, params):
            calls["orders"] += 1
            assert order_type == "market" and side == "buy"
            assert params["newClientOrderId"].startswith("eti")
            return {"id": "exchange-order-1", "status": "closed",
                    "filled": amount, "average": 100.0, "cost": amount * 100.0}
        def close(self):
            return None

    class FakeVault:
        def decrypt(self, value, owner_id, venue, field):
            return "private-key" if field == "api_key" else "private-secret"

    class FakeArtifacts:
        def load_model(self, uri, digest):
            return _LiveFakeModel()

    monkeypatch.setattr(execution, "CredentialVault", FakeVault)
    monkeypatch.setattr(execution, "ModelArtifactStore", FakeArtifacts)
    monkeypatch.setattr(execution, "build_features", lambda frame: pd.DataFrame(
        [[0.0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS, index=[len(frame) - 1]
    ))
    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(binance=FakeExchange))

    payload = {"owner_id": "owner", "venue": "binance", "symbol": "BTC/USDT",
               "timeframe": "1h", "task_id": "12345678-1234-1234-1234-123456789abc"}
    first = execution.run_live_tick(payload, store)
    assert first["status"] == "closed"
    assert first["side"] == "buy"
    assert first["filled"] > 0
    assert calls["orders"] == 1
    state = store.get_live_execution_state("owner", "binance", "BTC/USDT", "1h")
    assert state["position_qty"] > 0

    waiting = execution.run_live_tick(
        {"owner_id": "owner", "venue": "binance", "symbol": "ETH/USDT",
         "timeframe": "1h", "task_id": "22222222-1234-1234-1234-123456789abc"},
        store,
    )
    assert waiting["status"] == "waiting_for_existing_position"
    assert calls["orders"] == 1

    duplicate = execution.run_live_tick(payload, store)
    assert duplicate["status"] == "already_processed"
    assert calls["orders"] == 1


def test_uncertain_exchange_order_response_triggers_kill_switch(tmp_path, monkeypatch):
    import eti.execution as execution
    store = _live_store(tmp_path)
    now = pd.Timestamp.now(tz="UTC").floor("h")
    timestamps = [int((now - pd.Timedelta(hours=80 - i)).timestamp() * 1000) for i in range(80)]
    rows = [[ts, 100.0, 102.0, 99.0, 100.0, 1000.0] for ts in timestamps]

    class FakeExchange:
        def __init__(self, config):
            pass
        def fetch_balance(self):
            return {"free": {"USDT": 1000.0, "BTC": 0.0},
                    "total": {"USDT": 1000.0, "BTC": 0.0}}
        def fetch_ohlcv(self, symbol, timeframe, limit):
            return rows
        def parse_timeframe(self, timeframe):
            return 3600
        def fetch_ticker(self, symbol):
            return {"last": 100.0}
        def load_markets(self):
            return None
        def amount_to_precision(self, symbol, amount):
            return f"{amount:.6f}"
        def market(self, symbol):
            return {"limits": {"cost": {"min": 5.0}}}
        def create_order(self, *args, **kwargs):
            raise TimeoutError("uncertain network result")
        def close(self):
            return None

    class FakeVault:
        def decrypt(self, value, owner_id, venue, field):
            return "private"
    class FakeArtifacts:
        def load_model(self, uri, digest):
            return _LiveFakeModel()

    monkeypatch.setattr(execution, "CredentialVault", FakeVault)
    monkeypatch.setattr(execution, "ModelArtifactStore", FakeArtifacts)
    monkeypatch.setattr(execution, "build_features", lambda frame: pd.DataFrame(
        [[0.0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS, index=[len(frame) - 1]
    ))
    monkeypatch.setitem(sys.modules, "ccxt", SimpleNamespace(binance=FakeExchange))
    result = execution.run_live_tick(
        {"owner_id": "owner", "venue": "binance", "symbol": "BTC/USDT",
         "timeframe": "1h", "task_id": "abcdef12-1234-1234-1234-123456789abc"},
        store,
    )
    assert result["status"] == "stopped"
    assert result["reason"] == "order_submission_uncertain"
    assert store.get_live_execution_profile("owner")["enabled"] is False
    assert store.list_live_orders("owner")[0]["status"] == "unknown"


def test_live_supervisor_schedules_only_owner_authorized_profile_tasks(tmp_path):
    from datetime import datetime, timezone
    store = _live_store(tmp_path)
    scheduled = store.enqueue_due_live_execution_tasks(datetime.now(timezone.utc))
    assert scheduled == 1
    workspace = store.list_workspaces("owner")[0]
    tasks = store.list_agent_tasks("owner", workspace["id"], status="queued")
    live_tasks = [task for task in tasks if task["target_capability"] == "live.tick"]
    assert len(live_tasks) == 1
    assert live_tasks[0]["requester"] == "eti.live.supervisor"
    assert live_tasks[0]["input"]["scheduled_by"] == "eti.live.supervisor"
    assert live_tasks[0]["input"]["task_id"] == live_tasks[0]["id"]
    assert store.enqueue_due_live_execution_tasks(datetime.now(timezone.utc)) == 0


def test_live_account_risk_state_persists_across_symbols_and_days(tmp_path):
    store = _live_store(tmp_path)
    profile = store.update_live_execution_account_state("owner", "2026-11-13", 1000.0, 1050.0)
    assert profile["daily_date"] == "2026-11-13"
    assert profile["daily_start_equity"] == 1000.0
    assert profile["peak_equity"] == 1050.0
    states = store.list_live_execution_states("owner", "binance", "1h")
    assert states == []


def test_live_settings_cannot_orphan_an_open_position(tmp_path):
    import pytest
    store = _live_store(tmp_path)
    store.apply_live_execution_state(
        "owner", "binance", "BTC/USDT", "1h", "2026-11-13T10:00:00+00:00",
        {"position_qty": 0.01, "entry_price": 100.0, "daily_date": "2026-11-13",
         "daily_start_equity": 1000.0, "peak_equity": 1000.0},
    )
    store.disable_live_execution("owner")
    with pytest.raises(ValueError, match="close ETI-managed positions"):
        store.save_live_execution_profile(
            "owner", "binance", ["ETH/USDT"], "1h", 1000.0, {},
        )
