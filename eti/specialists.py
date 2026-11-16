"""Typed, fail-closed specialist services used by the Queen's allowlisted workers.

These services reuse independently tested ETI components. They do not grant
execution authority, invent provider data, or promote candidate models.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import math
from typing import Any

import pandas as pd

from .economic_disposition import assess_economic_viability
from .macro_data import InMemoryMacroProvider, MacroDataHub, MacroObservation
from .regimes import classify_market_regime
from .risk import RiskEngine, RiskRequest
from .evolution import select_candidate
from .transaction_costs import validate_transaction_cost_model


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _close_series(payload: dict[str, Any], *, minimum: int) -> pd.Series:
    values = payload.get("close")
    if not isinstance(values, list) or len(values) < minimum:
        raise ValueError(f"close must contain at least {minimum} observations")
    numeric = [_finite_number(value, f"close[{index}]", positive=True)
               for index, value in enumerate(values)]
    return pd.Series(numeric, dtype=float, name="close")


def market_regime_specialist(payload: dict[str, Any]) -> dict[str, Any]:
    """Return causal regime context from a supplied, chronological close series."""
    close = _close_series(payload, minimum=122)
    report = classify_market_regime(pd.DataFrame({"close": close}))
    latest = report.iloc[-1]
    return {
        "kind": "market_regime_specialist",
        "regime": str(latest["regime"]),
        "realized_volatility": (
            float(latest["realized_volatility"])
            if pd.notna(latest["realized_volatility"]) else None
        ),
        "historical_volatility_cutoff": (
            float(latest["historical_volatility_cutoff"])
            if pd.notna(latest["historical_volatility_cutoff"]) else None
        ),
        "trend_strength": float(latest["trend_strength"]) if pd.notna(latest["trend_strength"]) else None,
        "observations": len(close),
        "decision_mode": "descriptive_causal_only",
        "execution_enabled": False,
    }


def strategy_selection_specialist(payload: dict[str, Any]) -> dict[str, Any]:
    """Select among fixed strategies on the caller-supplied validation window only."""
    close = _close_series(payload, minimum=100)
    fee_bps = _finite_number(payload.get("fee_bps", 6.0), "fee_bps")
    slippage_bps = _finite_number(payload.get("slippage_bps", 2.0), "slippage_bps")
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("fee_bps and slippage_bps cannot be negative")
    selected = select_candidate(close)
    return {
        "kind": "strategy_selection_specialist",
        "candidate": selected["candidate"],
        "parameters": selected["parameters"],
        "validation_selection_score": float(selected["selection_score"]),
        "validation_return": float(selected["total_return"]),
        "validation_max_drawdown": float(selected["max_drawdown"]),
        "validation_trade_count": int(selected["trade_count"]),
        "observations": len(close),
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "test_evidence": "not_assessed",
        "execution_enabled": False,
        "limitations": [
            "Selection uses only the supplied validation slice; the caller must preserve a separate untouched test set.",
            "The candidate is a research suggestion, not an approved strategy or trade instruction.",
        ],
    }


def risk_assessment_specialist(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply the deterministic risk engine; model predictions cannot override it."""
    request_data = payload.get("request")
    if not isinstance(request_data, dict):
        raise ValueError("request must be an object")
    try:
        request = RiskRequest(**request_data)
    except TypeError as exc:
        raise ValueError(f"invalid risk request fields: {exc}") from exc
    decision = RiskEngine().assess(request)
    return {
        "kind": "risk_assessment_specialist",
        **asdict(decision),
        "authority": "deterministic_risk_gate",
        "model_override_allowed": False,
        "execution_enabled": False,
    }


def macro_as_of_specialist(payload: dict[str, Any]) -> dict[str, Any]:
    """Select only explicitly supplied macro observations available at decision time."""
    series_id = payload.get("series_id")
    provider_name = payload.get("provider", "operator_supplied")
    decision_value = payload.get("decision_time")
    raw = payload.get("observations")
    if not isinstance(series_id, str) or not series_id.strip():
        raise ValueError("series_id is required")
    if not isinstance(provider_name, str) or not provider_name.strip():
        raise ValueError("provider must be a non-empty string")
    if not isinstance(decision_value, str):
        raise ValueError("decision_time must be an ISO-8601 timestamp")
    try:
        decision_time = datetime.fromisoformat(decision_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("decision_time must be an ISO-8601 timestamp") from exc
    if decision_time.tzinfo is None or decision_time.utcoffset() is None:
        raise ValueError("decision_time must include a timezone")
    if not isinstance(raw, list):
        raise ValueError("observations must be a list")
    observations = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"observations[{index}] must be an object")
        try:
            observations.append(MacroObservation(
                series_id=item["series_id"],
                period=datetime.fromisoformat(item["period"].replace("Z", "+00:00")),
                value=item["value"],
                unit=item["unit"],
                released_at=datetime.fromisoformat(item["released_at"].replace("Z", "+00:00")),
                vintage_at=datetime.fromisoformat(item["vintage_at"].replace("Z", "+00:00")),
                source=item["source"],
                license_status=item["license_status"],
                provenance=item["provenance"],
            ))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"invalid observations[{index}]: {exc}") from exc
    provider = InMemoryMacroProvider(provider_name, observations)
    hub = MacroDataHub([provider])
    snapshot = hub.fetch_as_of(provider_name, series_id, decision_time)
    return {
        "kind": "macro_as_of_specialist",
        "series_id": snapshot.series_id,
        "decision_time": snapshot.decision_time.isoformat(),
        "provider": snapshot.provider,
        "observations": [asdict(item) | {"available_at": item.available_at.isoformat()}
                         for item in snapshot.observations],
        "license_statuses": list(snapshot.license_statuses),
        "lookahead_prevented": True,
        "execution_enabled": False,
    }


def execution_cost_specialist(payload: dict[str, Any]) -> dict[str, Any]:
    """Estimate evidenced transaction costs; this function never places an order."""
    identity = payload.get("instrument_identity")
    model = payload.get("cost_model")
    side = payload.get("side", "long")
    notional = _finite_number(payload.get("notional"), "notional", positive=True)
    bars = payload.get("holding_bars", 1)
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("instrument_identity is required")
    if side not in {"long", "short"}:
        raise ValueError("side must be long or short")
    if isinstance(bars, bool) or not isinstance(bars, int) or not 0 <= bars <= 100000:
        raise ValueError("holding_bars must be an integer from 0 to 100000")
    validated = validate_transaction_cost_model(model, identity=identity)
    one_way_bps = (
        validated["fee_bps"] + validated["spread_bps"] / 2.0
        + validated["slippage_bps"] + validated["market_impact_bps"]
    )
    funding_per_bar = validated["funding_bps_per_bar"] * (1 if side == "long" else -1)
    funding_bps = funding_per_bar * bars
    borrow_bps = validated["borrow_bps_per_bar"] * bars if side == "short" else 0.0
    total_bps = 2.0 * one_way_bps + funding_bps + borrow_bps
    return {
        "kind": "execution_cost_estimate",
        "instrument_identity": identity,
        "side": side,
        "notional": notional,
        "quote_currency": validated["quote_currency"],
        "one_way_entry_or_exit_bps": one_way_bps,
        "round_trip_trading_cost_bps": 2.0 * one_way_bps,
        "funding_cost_bps": funding_bps,
        "borrow_cost_bps": borrow_bps,
        "estimated_total_cost_bps": total_bps,
        "estimated_total_cost_quote": notional * total_bps / 10000.0,
        "evidence_validated": True,
        "orders_submitted": False,
        "execution_enabled": False,
        "limitations": [
            "This is an evidence-backed estimate, not a fill guarantee.",
            "Market impact remains a supplied model assumption and must be validated against realized fills.",
        ],
    }


def learning_economic_assessment_specialist(payload: dict[str, Any]) -> dict[str, Any]:
    """Classify completed research economics without promoting or executing a model."""
    report = payload.get("report")
    assessment = assess_economic_viability(report)
    return {
        "kind": "learning_economic_assessment",
        **assessment,
        "model_promotion_authority": False,
        "execution_enabled": False,
    }
