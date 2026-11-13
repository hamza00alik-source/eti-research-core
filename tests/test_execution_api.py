import pytest
from fastapi import HTTPException

from eti import api


def test_live_authorization_requires_ip_restriction_confirmation(monkeypatch):
    monkeypatch.setattr(api, "_require_model_admin", lambda account: None)

    class Store:
        def authorize_live_execution(self, owner_id):
            raise AssertionError("authorization must not reach storage without all confirmations")

    monkeypatch.setattr(api, "_memory_store", lambda: Store())
    payload = api.LiveExecutionAuthorizationPayload(
        confirm_live_trading=True,
        confirm_spot_trading_permission=True,
        confirm_withdrawals_disabled=True,
        confirm_ip_restrictions=False,
    )
    with pytest.raises(HTTPException) as error:
        api.authorize_live_execution(payload, {"id": "owner", "username": "owner"})
    assert error.value.status_code == 422
    assert "IP-restriction" in error.value.detail


def test_live_authorization_proceeds_only_after_all_four_confirmations(monkeypatch):
    monkeypatch.setattr(api, "_require_model_admin", lambda account: None)

    class Store:
        def authorize_live_execution(self, owner_id):
            assert owner_id == "owner"
            return {"enabled": True, "venue": "binance"}

    monkeypatch.setattr(api, "_memory_store", lambda: Store())
    payload = api.LiveExecutionAuthorizationPayload(
        confirm_live_trading=True,
        confirm_spot_trading_permission=True,
        confirm_withdrawals_disabled=True,
        confirm_ip_restrictions=True,
    )
    result = api.authorize_live_execution(payload, {"id": "owner", "username": "owner"})
    assert result["mode"] == "live"
    assert result["profile"]["enabled"] is True


def test_execution_status_reports_disabled_by_default_live_feature():
    result = api.execution_status()
    assert result["enabled"] is False
    assert result["live_orders_implemented"] is True
    assert result["owner_authorization_required"] is True
    assert "disabled by default" in result["message"]


def test_kill_switch_response_warns_existing_orders_and_positions_remain(monkeypatch):
    monkeypatch.setattr(api, "_require_model_admin", lambda account: None)

    class Store:
        def disable_live_execution(self, owner_id, reason):
            assert owner_id == "owner"
            assert reason == "owner_kill_switch"
            return {"enabled": False}

    monkeypatch.setattr(api, "_memory_store", lambda: Store())
    result = api.live_execution_kill_switch({"id": "owner", "username": "owner"})
    assert result["mode"] == "disabled"
    assert "not automatically canceled or liquidated" in result["message"]
