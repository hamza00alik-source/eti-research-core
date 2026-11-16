"""Shared intelligence architecture and department-aware specialist routing.

The Queen coordinates bounded, allowlisted capabilities. Specialist models are
independent, versioned candidates; a single model is never assumed to serve
every department or market. This module describes and routes existing systems;
it does not train models, grant permissions, or execute orders.
"""
from __future__ import annotations

from typing import Any


DEPARTMENTS: dict[str, dict[str, Any]] = {
    "market_data": {
        "name": "Market Data Intelligence",
        "mode": "deterministic_services",
        "purpose": "Discover instruments and validate source, provenance, freshness and completeness.",
        "capabilities": ("market.coverage", "market.discover"),
        "model_required": False,
        "model_roles": (),
    },
    "market_intelligence": {
        "name": "Market Intelligence",
        "mode": "shared_features_plus_specialists",
        "purpose": "Describe regimes, volatility, trends, correlations and derivatives context.",
        "capabilities": ("market.snapshot", "derivatives.context"),
        "model_required": True,
        "model_roles": ("regime_classifier", "volatility_estimator", "cross_asset_context"),
    },
    "prediction": {
        "name": "Prediction Intelligence",
        "mode": "specialist_models",
        "purpose": "Estimate calibrated, task-specific outcome probabilities.",
        "capabilities": ("model.train_research",),
        "model_required": True,
        "model_roles": ("direction_classifier", "return_distribution", "volatility_forecast"),
    },
    "strategy": {
        "name": "Strategy Intelligence",
        "mode": "rules_plus_specialist_models",
        "purpose": "Generate, compare and select strategies using locked, cost-adjusted evidence.",
        "capabilities": ("strategy.walk_forward", "paper.tick"),
        "model_required": True,
        "model_roles": ("strategy_selector", "portfolio_optimizer"),
    },
    "risk": {
        "name": "Risk Intelligence",
        "mode": "deterministic_controls_plus_analytics",
        "purpose": "Enforce position, exposure, leverage, drawdown and loss limits independently.",
        "capabilities": (),
        "model_required": False,
        "model_roles": ("risk_forecast_optional",),
    },
    "macro": {
        "name": "Macro Intelligence",
        "mode": "point_in_time_data_plus_specialists",
        "purpose": "Interpret releases, revisions, policy events and cross-market macro context.",
        "capabilities": (),
        "model_required": True,
        "model_roles": ("macro_regime", "event_impact"),
    },
    "execution": {
        "name": "Execution Intelligence",
        "mode": "cost_models_plus_deterministic_controls",
        "purpose": "Estimate fees, spread, slippage, impact and order quality for eligible venues.",
        "capabilities": ("live.tick",),
        "model_required": False,
        "model_roles": ("fill_probability_optional", "slippage_estimator"),
    },
    "learning": {
        "name": "Learning and Evaluation Intelligence",
        "mode": "evidence_and_lifecycle_services",
        "purpose": "Monitor drift, evaluate outcomes and govern champion/challenger candidates.",
        "capabilities": (),
        "model_required": False,
        "model_roles": ("drift_detector_optional",),
    },
}

_CAPABILITY_DEPARTMENT = {
    capability: department
    for department, spec in DEPARTMENTS.items()
    for capability in spec["capabilities"]
}


def department_for_capability(capability: str) -> str:
    """Resolve only explicitly allowlisted capability-to-department mappings."""
    if not isinstance(capability, str) or not capability.strip():
        raise ValueError("capability must be a non-empty string")
    try:
        return _CAPABILITY_DEPARTMENT[capability]
    except KeyError as exc:
        raise LookupError(f"capability has no intelligence department: {capability}") from exc


def _declared_department(model: dict[str, Any]) -> str | None:
    """Read explicit department metadata; never infer from a model name."""
    schema = model.get("input_schema")
    report = model.get("evaluation_report")
    values = []
    if isinstance(schema, dict) and "department" in schema:
        values.append(schema["department"])
    if isinstance(report, dict) and "department" in report:
        values.append(report["department"])
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("model department declarations must be non-empty strings")
    declared = set(values)
    if len(declared) > 1:
        raise ValueError("model has conflicting department declarations")
    if not declared:
        return None
    department = next(iter(declared))
    if department not in DEPARTMENTS:
        raise ValueError(f"unknown model department: {department}")
    return department


def select_specialist_models(
    model_versions: list[dict[str, Any]],
    department: str,
    *,
    asset_class: str | None = None,
    market_type: str | None = None,
    timeframe: str | None = None,
) -> list[dict[str, Any]]:
    """Return approved, explicitly assigned models compatible with the request.

    Candidate/retired models and legacy models without department metadata are
    deliberately excluded. A missing specialist never falls back to another
    department's model.
    """
    if department not in DEPARTMENTS:
        raise ValueError("unknown department")
    if not isinstance(model_versions, list):
        raise ValueError("model_versions must be a list")
    expected_class = (
        f"{asset_class}_{market_type}"
        if asset_class is not None and market_type is not None else None
    )
    selected = []
    for model in model_versions:
        if not isinstance(model, dict):
            continue
        if model.get("lifecycle") != "approved":
            continue
        if _declared_department(model) != department:
            continue
        classes = model.get("supported_classes")
        if not isinstance(classes, list) or not classes:
            continue
        if expected_class is not None and expected_class not in classes:
            continue
        schema = model.get("input_schema")
        schema = schema if isinstance(schema, dict) else {}
        market_types = schema.get("market_types")
        timeframes = schema.get("timeframes")
        if market_type is not None and isinstance(market_types, list) and market_type not in market_types:
            continue
        if timeframe is not None and isinstance(timeframes, list) and timeframe not in timeframes:
            continue
        selected.append(model)
    return sorted(selected, key=lambda item: (str(item.get("model_id", "")), str(item.get("version", ""))))


def build_intelligence_manifest(
    model_versions: list[dict[str, Any]] | None = None,
    *,
    capabilities: list[str] | None = None,
) -> dict[str, Any]:
    """Describe department readiness without implying model quality or authority."""
    versions = model_versions or []
    if not isinstance(versions, list):
        raise ValueError("model_versions must be a list")
    registered = []
    unassigned = 0
    for model in versions:
        if not isinstance(model, dict):
            continue
        department = _declared_department(model)
        if department is None:
            unassigned += 1
        else:
            registered.append((department, model))

    departments = {}
    for department, spec in DEPARTMENTS.items():
        items = [model for key, model in registered if key == department]
        approved = sum(model.get("lifecycle") == "approved" for model in items)
        candidates = sum(model.get("lifecycle") == "candidate" for model in items)
        departments[department] = {
            **spec,
            "capabilities": list(spec["capabilities"]),
            "model_roles": list(spec["model_roles"]),
            "registered_models": len(items),
            "approved_models": approved,
            "candidate_models": candidates,
            "status": (
                "ready" if approved else
                "candidate_review" if candidates else
                "not_configured" if spec["model_required"] else
                "deterministic_foundation_available"
            ),
        }

    capability_list = sorted(set(capabilities or []))
    return {
        "schema": "eti.shared_intelligence_architecture.v1",
        "queen_role": "central_coordinator",
        "shared_foundation": {
            "data_contracts": True,
            "feature_contracts": True,
            "model_version_registry": True,
            "evidence_and_calibration": True,
            "shared_memory_and_audit": True,
            "department_isolation": True,
        },
        "departments": departments,
        "registered_model_versions": len(versions),
        "unassigned_legacy_model_versions": unassigned,
        "available_allowlisted_capabilities": capability_list,
        "limitations": [
            "Department declarations must be explicit in input_schema.department or evaluation_report.department.",
            "Legacy model versions are not assigned by guessing from their names.",
            "A registered or approved model is not proof of profitability or live-trading suitability.",
            "Departments without implemented capabilities are architecture contracts, not active autonomous agents.",
            "The Queen cannot change risk limits, approve models, expand budgets or enable execution.",
        ],
        "model_promotion_authority": False,
        "budget_authority": False,
        "execution_authority": False,
    }


def route_department_task(capability: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Create a transparent routing description for an existing server capability."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    department = department_for_capability(capability)
    return {
        "department": department,
        "department_name": DEPARTMENTS[department]["name"],
        "capability": capability,
        "payload": payload,
        "routing_mode": "allowlisted_capability",
        "new_permissions_granted": False,
        "model_selected": False,
        "execution_enabled": False,
    }
