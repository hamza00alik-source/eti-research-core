import pytest
from fastapi import HTTPException

from eti import api


def _account():
    return {"id": "owner", "username": "owner"}


def _confirmed_payload():
    return api.LiveExecutionAuthorizationPayload(
        confirm_live_trading=True,
        confirm_spot_trading_permission=True,
        confirm_withdrawals_disabled=True,
        confirm_ip_restrictions=True,
    )


def test_execution_status_reports_implemented_but_hard_disabled_order_path(monkeypatch):
    monkeypatch.setattr(api, "LIVE_ORDER_EXECUTION_ENABLED", False)
    result = api.execution_status()
    assert result["enabled"] is False
    assert result["live_orders_implemented"] is True
    assert result["hard_kill_switch_active"] is True
    assert "code-level hard kill switch" in result["message"]


def test_owner_authorization_cannot_override_global_hard_kill(monkeypatch):
    class Store:
        called = False

        def authorize_live_execution(self, owner_id):
            self.called = True
            raise AssertionError("storage authorization must not run")

    store = Store()
    monkeypatch.setattr(api, "_require_model_admin", lambda account: None)
    monkeypatch.setattr(api, "_memory_store", lambda: store)
    monkeypatch.setattr(api, "LIVE_ORDER_EXECUTION_ENABLED", False)

    with pytest.raises(HTTPException) as error:
        api.authorize_live_execution(_confirmed_payload(), _account())

    assert error.value.status_code == 409
    assert "code-level hard kill switch" in error.value.detail
    assert store.called is False


def test_profile_does_not_report_live_mode_while_global_hard_kill_is_active(monkeypatch):
    class Store:
        def get_live_execution_profile(self, owner_id):
            return {"enabled": True, "venue": "binance"}

    monkeypatch.setattr(api, "_require_model_admin", lambda account: None)
    monkeypatch.setattr(api, "_memory_store", Store)
    monkeypatch.setattr(api, "LIVE_ORDER_EXECUTION_ENABLED", False)

    result = api.get_live_execution_profile(_account())

    assert result["mode"] == "disabled"
    assert result["live_execution_available"] is False
    assert result["live_order_execution_globally_disabled"] is True


def test_kill_switch_explains_existing_exchange_orders_are_not_canceled(monkeypatch):
    class Store:
        def disable_live_execution(self, owner_id, reason):
            return {"enabled": False}

    monkeypatch.setattr(api, "_require_model_admin", lambda account: None)
    monkeypatch.setattr(api, "_memory_store", Store)

    result = api.live_execution_kill_switch(_account())

    assert result["mode"] == "disabled"
    assert "not automatically canceled or liquidated" in result["message"]
