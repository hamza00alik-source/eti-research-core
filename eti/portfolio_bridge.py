"""Owner-scoped paper portfolio adapter for evidence-gated allocation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .allocation import plan_paper_allocations


def allocate_from_persisted_snapshot(
    store: Any,
    owner_id: str,
    workspace_id: str,
    payload: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a paper proposal using only the caller's persisted account snapshot.

    Owner/workspace identity is injected by the worker from the claimed task,
    never accepted from task input. The owner-attested snapshot is explicitly
    not treated as broker-verified truth.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    account_id = payload.get("portfolio_account_id")
    if not isinstance(account_id, str) or not account_id.strip():
        raise ValueError("portfolio_account_id is required")
    max_age = payload.get("max_snapshot_age_seconds")
    if isinstance(max_age, bool) or not isinstance(max_age, int) or not 60 <= max_age <= 86400:
        raise ValueError("max_snapshot_age_seconds must be an integer from 60 to 86400")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")
    account = store.get_paper_portfolio_account(owner_id, workspace_id, account_id)
    if account is None:
        raise LookupError("paper portfolio account not found")
    if account.get("status") != "active":
        raise ValueError("paper portfolio account is not active")
    snapshot = store.get_latest_paper_portfolio_snapshot(owner_id, workspace_id, account_id)
    if snapshot is None:
        raise ValueError("paper portfolio account has no persisted snapshot")
    if snapshot.get("source") != "owner_attested":
        raise ValueError("unsupported portfolio snapshot source")
    if snapshot.get("owner_id") != owner_id or snapshot.get("workspace_id") != workspace_id:
        raise LookupError("paper portfolio snapshot scope mismatch")
    try:
        as_of = datetime.fromisoformat(str(snapshot["as_of"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("portfolio snapshot timestamp is invalid") from exc
    if as_of.tzinfo is None:
        raise ValueError("portfolio snapshot timestamp must include a timezone")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    age = (current.astimezone(timezone.utc) - as_of.astimezone(timezone.utc)).total_seconds()
    if age < 0:
        raise ValueError("portfolio snapshot timestamp is in the future")
    if age > max_age:
        raise ValueError("portfolio snapshot is stale")

    positions = snapshot.get("positions")
    if not isinstance(positions, list):
        raise ValueError("portfolio snapshot positions are invalid")
    current_positions: dict[str, float] = {}
    group_exposure: dict[str, float] = {}
    for position in positions:
        if not isinstance(position, dict):
            raise ValueError("portfolio snapshot contains an invalid position")
        symbol = position.get("symbol")
        group = position.get("correlation_group")
        notional = position.get("notional")
        if (not isinstance(symbol, str) or not symbol
                or not isinstance(group, str) or not group
                or isinstance(notional, bool) or not isinstance(notional, (int, float))
                or not 0 < float(notional) < float("inf")):
            raise ValueError("portfolio snapshot contains an invalid position")
        current_positions[symbol] = float(notional)
        group_exposure[group] = group_exposure.get(group, 0.0) + float(notional)

    # Never accept caller-supplied equity or exposure maps. Reserve at least
    # the cash deficit reported by the stored snapshot, even if a lower buffer
    # was requested in task input.
    limits = {
        key: payload[key] for key in (
            "cash_buffer_fraction", "max_gross_exposure_fraction",
            "max_single_asset_fraction", "max_correlated_group_fraction",
            "max_risk_per_trade", "max_positions",
        ) if key in payload
    }
    requested_buffer = limits.get("cash_buffer_fraction", 0.20)
    if (isinstance(requested_buffer, bool) or not isinstance(requested_buffer, (int, float))
            or not 0 <= float(requested_buffer) <= 0.95):
        raise ValueError("cash_buffer_fraction must be between 0 and 0.95")
    actual_cash_floor = 1.0 - float(snapshot["cash"]) / float(snapshot["equity"])
    limits["cash_buffer_fraction"] = max(float(requested_buffer), actual_cash_floor)
    plan = plan_paper_allocations({
        **limits,
        "equity": snapshot["equity"],
        "current_positions": current_positions,
        "correlated_group_exposure": group_exposure,
        "candidates": candidates,
    })
    plan["portfolio_provenance"] = {
        "account_id": account_id,
        "snapshot_id": snapshot["id"],
        "snapshot_as_of": snapshot["as_of"],
        "snapshot_age_seconds": round(age, 3),
        "snapshot_source": "owner_attested",
        "base_currency": account["base_currency"],
        "exchange_verified": False,
    }
    plan["paper_only"] = True
    plan["execution_enabled"] = False
    plan["orders_submitted"] = False
    return plan
