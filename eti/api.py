"""Research API and PWA host. No exchange order routes exist."""
from __future__ import annotations
from pathlib import Path
import math
import os
import sqlite3
from typing import Any

import pandas as pd

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from .runtime import runtime_lifespan
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agents import default_capabilities
from .artifacts import ModelArtifactError, ModelArtifactStore
from .auth import AuthManager, AuthenticationError, RateLimitExceeded, RegistrationDisabled
from .config import AppConfig
from .decision import build_research_decision
from .portfolio import assess_portfolio
from .risk import RiskEngine, RiskRequest
from .storage import MemoryStore

app = FastAPI(title=AppConfig().app_name, version=AppConfig().version, lifespan=runtime_lifespan)
risk_engine = RiskEngine()


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; manifest-src 'self'; "
        "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
    )
    if request.url.path.startswith((
        "/v1/", "/health", "/risk", "/markets", "/providers",
        "/decision", "/portfolio", "/execution"
    )):
        response.headers["Cache-Control"] = "no-store"
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


class RegisterPayload(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=12, max_length=1024)
    invite_code: str = Field(min_length=1, max_length=256)


class LoginPayload(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=12, max_length=1024)


class WorkspacePayload(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class MemoryPayload(BaseModel):
    value: Any


class AgentTaskPayload(BaseModel):
    target_capability: str = Field(min_length=1, max_length=120)
    input: Any
    deadline_at: str | None = Field(default=None, max_length=40)
    max_attempts: int = Field(default=3, ge=1, le=10)


class AgentTaskUpdatePayload(BaseModel):
    status: str = Field(min_length=1, max_length=20)
    result: Any = None


class AgentMessagePayload(BaseModel):
    recipient: str = Field(min_length=1, max_length=80)
    message_type: str = Field(min_length=1, max_length=120)
    payload: Any
    task_id: str | None = Field(default=None, max_length=128)


def _auth_manager() -> AuthManager:
    try:
        return AuthManager()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Authentication is not configured") from exc


def _memory_store() -> MemoryStore:
    return MemoryStore()


def _client_subject(request) -> str:
    # Use the ASGI peer address only; do not trust arbitrary X-Forwarded-For headers.
    return request.client.host if request.client and request.client.host else "unknown-client"


def _apply_auth_rate_limits(manager: AuthManager, request, action: str, username: str | None = None):
    ip = _client_subject(request)
    if action == "login":
        manager.consume_rate_limit("login-ip", ip, limit=20, window_seconds=900)
        if username:
            manager.consume_rate_limit("login-user", username.strip().lower(), limit=10,
                                       window_seconds=900)
    elif action == "register":
        manager.consume_rate_limit("register-ip", ip, limit=5, window_seconds=3600)


def current_account(authorization: str | None = Header(default=None)) -> dict[str, str]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required",
                            headers={"WWW-Authenticate": "Bearer"})
    token = authorization[7:].strip()
    try:
        return _auth_manager().verify_token(token)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc),
                            headers={"WWW-Authenticate": "Bearer"}) from exc


def _require_workspace(store: MemoryStore, account: dict[str, str], workspace_id: str) -> None:
    if not any(item["id"] == workspace_id for item in store.list_workspaces(account["id"])):
        raise HTTPException(status_code=404, detail="workspace not found")


@app.post("/v1/auth/register", status_code=201)
def register(payload: RegisterPayload, request: Request):
    manager = _auth_manager()
    try:
        _apply_auth_rate_limits(manager, request, "register")
        account = manager.register(payload.username, payload.password, payload.invite_code)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": str(exc.retry_after)}) from exc
    except RegistrationDisabled as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"account": account, "access_token": manager.issue_token(account),
            "token_type": "bearer", "expires_in": manager.token_ttl_seconds}


@app.post("/v1/auth/token")
def login(payload: LoginPayload, request: Request):
    manager = _auth_manager()
    try:
        _apply_auth_rate_limits(manager, request, "login", payload.username)
        account = manager.authenticate(payload.username, payload.password)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc),
                            headers={"Retry-After": str(exc.retry_after)}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="invalid username or password",
                            headers={"WWW-Authenticate": "Bearer"}) from exc
    return {"account": account, "access_token": manager.issue_token(account),
            "token_type": "bearer", "expires_in": manager.token_ttl_seconds}


@app.get("/v1/profile")
def profile(account: dict[str, str] = Depends(current_account)):
    return {"account": account}


@app.get("/v1/account/export")
def export_account_data(account: dict[str, str] = Depends(current_account)):
    """Download the caller's account-scoped research data as JSON."""
    return _memory_store().export_account_data(account["id"])


@app.get("/v1/workspaces")
def list_workspaces(account: dict[str, str] = Depends(current_account)):
    return {"items": _memory_store().list_workspaces(account["id"])}


@app.post("/v1/workspaces", status_code=201)
def create_workspace(payload: WorkspacePayload,
                     account: dict[str, str] = Depends(current_account)):
    try:
        workspace = _memory_store().create_workspace(account["id"], payload.name)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="workspace name already exists") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return workspace


@app.delete("/v1/workspaces/{workspace_id}", status_code=204)
def delete_workspace(workspace_id: str,
                     account: dict[str, str] = Depends(current_account)):
    if not _memory_store().delete_workspace(account["id"], workspace_id):
        raise HTTPException(status_code=404, detail="workspace not found")
    return Response(status_code=204)


@app.get("/v1/workspaces/{workspace_id}/memory")
def list_memory(workspace_id: str, namespace: str | None = Query(default=None, max_length=80),
                account: dict[str, str] = Depends(current_account)):
    store = _memory_store()
    _require_workspace(store, account, workspace_id)
    try:
        return {"items": store.list_memory(account["id"], workspace_id, namespace)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.put("/v1/workspaces/{workspace_id}/memory/{namespace}/{key}")
def put_memory(workspace_id: str, namespace: str, key: str, payload: MemoryPayload,
               account: dict[str, str] = Depends(current_account)):
    store = _memory_store()
    _require_workspace(store, account, workspace_id)
    try:
        return store.put_memory(account["id"], workspace_id, namespace, key, payload.value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="workspace not found") from exc


@app.get("/v1/workspaces/{workspace_id}/memory/{namespace}/{key}")
def get_memory(workspace_id: str, namespace: str, key: str,
               account: dict[str, str] = Depends(current_account)):
    store = _memory_store()
    _require_workspace(store, account, workspace_id)
    item = store.get_memory(account["id"], workspace_id, namespace, key)
    if item is None:
        raise HTTPException(status_code=404, detail="memory item not found")
    return item


@app.delete("/v1/workspaces/{workspace_id}/memory/{namespace}/{key}", status_code=204)
def delete_memory(workspace_id: str, namespace: str, key: str,
                  account: dict[str, str] = Depends(current_account)):
    store = _memory_store()
    _require_workspace(store, account, workspace_id)
    if not store.delete_memory(account["id"], workspace_id, namespace, key):
        raise HTTPException(status_code=404, detail="memory item not found")
    return Response(status_code=204)


@app.get("/v1/agents/capabilities")
def list_agent_capabilities(account: dict[str, str] = Depends(current_account)):
    return {"items": default_capabilities().capabilities()}


@app.get("/v1/agents/status")
def agent_worker_status(account: dict[str, str] = Depends(current_account)):
    items = _memory_store().list_agent_workers(online_window_seconds=60)
    return {"items": items, "online_count": sum(1 for item in items if item["online"])}


@app.get("/v1/workspaces/{workspace_id}/agents/tasks")
def list_agent_tasks(workspace_id: str, status: str | None = None,
                     limit: int = Query(default=100, ge=1, le=500),
                     account: dict[str, str] = Depends(current_account)):
    store = _memory_store()
    _require_workspace(store, account, workspace_id)
    try:
        return {"items": store.list_agent_tasks(account["id"], workspace_id, status, limit)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/workspaces/{workspace_id}/agents/tasks", status_code=201)
def create_agent_task(workspace_id: str, payload: AgentTaskPayload,
                      account: dict[str, str] = Depends(current_account)):
    try:
        return _memory_store().create_agent_task(
            account["id"], workspace_id, account["username"],
            payload.target_capability, payload.input, payload.deadline_at, payload.max_attempts)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="workspace not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v1/workspaces/{workspace_id}/agents/tasks/{task_id}")
def get_agent_task(workspace_id: str, task_id: str,
                   account: dict[str, str] = Depends(current_account)):
    task = _memory_store().get_agent_task(account["id"], workspace_id, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@app.patch("/v1/workspaces/{workspace_id}/agents/tasks/{task_id}")
def update_agent_task(workspace_id: str, task_id: str, payload: AgentTaskUpdatePayload,
                      account: dict[str, str] = Depends(current_account)):
    if payload.status != "cancelled" or payload.result is not None:
        raise HTTPException(status_code=403,
                            detail="Accounts may cancel queued tasks; only the worker may complete or fail tasks")
    try:
        return _memory_store().cancel_agent_task(account["id"], workspace_id, task_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="task not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/v1/workspaces/{workspace_id}/agents/messages")
def list_agent_messages(workspace_id: str, task_id: str | None = None,
                        limit: int = Query(default=100, ge=1, le=500),
                        account: dict[str, str] = Depends(current_account)):
    store = _memory_store()
    _require_workspace(store, account, workspace_id)
    try:
        return {"items": store.list_agent_messages(account["id"], workspace_id, task_id, limit)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/workspaces/{workspace_id}/agents/messages", status_code=201)
def create_agent_message(workspace_id: str, payload: AgentMessagePayload,
                          account: dict[str, str] = Depends(current_account)):
    try:
        return _memory_store().send_agent_message(
            account["id"], workspace_id, account["username"], payload.recipient,
            payload.message_type, payload.payload, payload.task_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="workspace or task not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class ModelInferencePayload(BaseModel):
    asset_class: str = Field(min_length=1, max_length=40)
    features: dict[str, float] = Field(min_length=1, max_length=128)


class ModelVersionPayload(BaseModel):
    model_id: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=80)
    artifact_uri: str = Field(min_length=1, max_length=500)
    artifact_sha256: str = Field(min_length=64, max_length=64)
    input_schema: dict[str, Any]
    supported_classes: list[str] = Field(min_length=1, max_length=20)
    training_data_ref: str = Field(min_length=1, max_length=300)
    evaluation_report: dict[str, Any]
    calibration_status: str = Field(min_length=1, max_length=20)


def _model_admin_usernames() -> set[str]:
    return {
        item.strip().lower()
        for item in os.environ.get("ETI_ADMIN_USERNAMES", "").split(",")
        if item.strip()
    }


def _model_has_verified_calibration(model: Any) -> bool:
    checker = getattr(model, "has_valid_calibration", None)
    if callable(checker):
        return bool(checker())
    report = getattr(model, "calibration_report", None)
    temperature = report.get("temperature") if isinstance(report, dict) else None
    sample_count = report.get("sample_count", 0) if isinstance(report, dict) else 0
    class_counts = report.get("class_counts") if isinstance(report, dict) else None
    nll_before = report.get("nll_before") if isinstance(report, dict) else None
    nll_after = report.get("nll_after") if isinstance(report, dict) else None
    return (
        getattr(model, "calibration_status", "uncalibrated") == "calibrated"
        and isinstance(report, dict)
        and report.get("passed") is True
        and report.get("method") == "temperature_scaling"
        and isinstance(sample_count, int) and not isinstance(sample_count, bool) and sample_count >= 30
        and isinstance(class_counts, dict)
        and all(
            isinstance(class_counts.get(str(class_id)), int)
            and not isinstance(class_counts.get(str(class_id)), bool)
            and class_counts.get(str(class_id)) >= 5
            for class_id in (0, 1, 2)
        )
        and not isinstance(temperature, bool)
        and isinstance(temperature, (int, float)) and math.isfinite(temperature)
        and temperature > 0
        and not isinstance(nll_before, bool) and isinstance(nll_before, (int, float)) and math.isfinite(nll_before)
        and not isinstance(nll_after, bool) and isinstance(nll_after, (int, float)) and math.isfinite(nll_after)
        and nll_after <= nll_before + 1e-9
    )


def _model_feature_schema_matches(model: Any, registered_features: Any) -> bool:
    if not isinstance(registered_features, list) or not registered_features:
        return False
    estimator = getattr(model, "estimator", None)
    trained_features = getattr(estimator, "feature_names_in_", None)
    if trained_features is None:
        return False
    return [str(name) for name in trained_features] == registered_features


def _require_model_admin(account: dict[str, str]) -> None:
    if account["username"].lower() not in _model_admin_usernames():
        raise HTTPException(status_code=403, detail="model registry administrator access required")


@app.get("/v1/models")
def list_models(lifecycle: str | None = None,
                account: dict[str, str] = Depends(current_account)):
    is_admin = account["username"].lower() in _model_admin_usernames()
    selected_lifecycle = lifecycle if is_admin else "approved"
    if not is_admin and lifecycle not in (None, "approved"):
        raise HTTPException(status_code=404, detail="model version not found")
    try:
        return {"items": _memory_store().list_model_versions(selected_lifecycle)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v1/models/{model_id}/{version}")
def get_model(model_id: str, version: str,
              account: dict[str, str] = Depends(current_account)):
    model = _memory_store().get_model_version(model_id, version)
    if model is None or (
        model["lifecycle"] != "approved"
        and account["username"].lower() not in _model_admin_usernames()
    ):
        raise HTTPException(status_code=404, detail="model version not found")
    return model


@app.post("/v1/inference/{model_id}/{version}")
def infer_model(model_id: str, version: str, payload: ModelInferencePayload,
                account: dict[str, str] = Depends(current_account)):
    manager = _auth_manager()
    try:
        manager.consume_rate_limit(
            "inference", account["id"], limit=60, window_seconds=60
        )
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429, detail="Inference request limit reached",
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc

    metadata = _memory_store().get_model_version(model_id, version)
    if metadata is None or metadata["lifecycle"] != "approved":
        raise HTTPException(status_code=404, detail="approved model version not found")
    if payload.asset_class not in metadata["supported_classes"]:
        raise HTTPException(status_code=422, detail="model does not support this asset class")
    expected = metadata["input_schema"].get("features")
    if (not isinstance(expected, list) or not expected
            or any(not isinstance(name, str) or not name for name in expected)
            or len(expected) != len(set(expected))):
        raise HTTPException(status_code=503, detail="registered model input schema is invalid")
    if set(payload.features) != set(expected):
        raise HTTPException(
            status_code=422,
            detail={"message": "feature names must exactly match the registered schema",
                    "expected": expected},
        )
    values = [float(payload.features[name]) for name in expected]
    if not all(math.isfinite(value) for value in values):
        raise HTTPException(status_code=422, detail="all model features must be finite numbers")
    try:
        model = ModelArtifactStore().load_model(
            metadata["artifact_uri"], metadata["artifact_sha256"]
        )
    except ModelArtifactError as exc:
        raise HTTPException(status_code=503, detail="approved model artifact is unavailable or invalid") from exc
    if not _model_feature_schema_matches(model, expected):
        raise HTTPException(status_code=503, detail="approved model feature schema does not match its artifact")
    if not _model_has_verified_calibration(model):
        raise HTTPException(status_code=503, detail="approved model lacks a valid calibration report")
    frame = pd.DataFrame([values], columns=expected)
    try:
        probabilities = [float(value) for value in model.predict_proba(frame)[0]]
        probability_total = math.fsum(probabilities)
        if (len(probabilities) != 3 or not all(math.isfinite(value) and value >= 0 for value in probabilities)
                or not math.isclose(probability_total, 1.0, rel_tol=1e-5, abs_tol=1e-6)):
            raise ValueError("model returned invalid class probabilities")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="approved model returned invalid inference output") from exc
    class_names = {0: "short", 1: "flat", 2: "long"}
    scores = {class_names[index]: probabilities[index] for index in range(3)}
    predicted = max(scores, key=scores.get)
    return {
        "model_id": model_id,
        "version": version,
        "artifact_sha256": metadata["artifact_sha256"],
        "asset_class": payload.asset_class,
        "prediction": predicted,
        "probabilities": scores,
        "warning": "Model inference only. This is not a trade recommendation or order instruction.",
    }


@app.post("/v1/models", status_code=201)
def register_model(payload: ModelVersionPayload,
                   account: dict[str, str] = Depends(current_account)):
    _require_model_admin(account)
    try:
        return _memory_store().register_model_version(
            payload.model_id, payload.version, payload.artifact_uri,
            payload.artifact_sha256, payload.input_schema, payload.supported_classes,
            payload.training_data_ref, payload.evaluation_report,
            payload.calibration_status, account["username"])
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="model version already exists") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/models/{model_id}/{version}/approve")
def approve_model(model_id: str, version: str,
                  account: dict[str, str] = Depends(current_account)):
    _require_model_admin(account)
    store = _memory_store()
    candidate = store.get_model_version(model_id, version)
    if candidate is None:
        raise HTTPException(status_code=404, detail="model version not found")
    try:
        # Approval is allowed only when the server-owned artifact exists, matches
        # the registered digest and deserializes as a fitted ETI BaselineModel.
        artifact_model = ModelArtifactStore().load_model(
            candidate["artifact_uri"], candidate["artifact_sha256"]
        )
        if not _model_feature_schema_matches(
            artifact_model, candidate["input_schema"].get("features")
        ):
            raise HTTPException(
                status_code=409,
                detail="registered feature schema must exactly match the trained model",
            )
        if not _model_has_verified_calibration(artifact_model):
            raise HTTPException(
                status_code=409,
                detail="model artifact must contain a passing temperature-calibration report",
            )
    except ModelArtifactError as exc:
        raise HTTPException(
            status_code=409,
            detail="model artifact must exist and pass integrity checks before approval",
        ) from exc
    try:
        return store.approve_model_version(model_id, version, account["username"])
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="model version not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/v1/models/{model_id}/{version}/retire")
def retire_model(model_id: str, version: str,
                 account: dict[str, str] = Depends(current_account)):
    _require_model_admin(account)
    try:
        return _memory_store().retire_model_version(model_id, version)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="model version not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


class PortfolioPayload(BaseModel):
    equity: float = Field(gt=0)
    current_positions: dict[str, float] = Field(default_factory=dict)
    proposed_symbol: str = Field(min_length=1)
    proposed_notional: float = Field(gt=0)
    correlated_group_exposure: float = Field(default=0, ge=0)


class RiskPayload(BaseModel):
    equity: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    stop_price: float = Field(gt=0)
    requested_notional: float = Field(gt=0)
    gross_exposure: float = Field(default=0, ge=0)
    daily_pnl_fraction: float = 0
    current_drawdown: float = Field(default=0, ge=0)
    leverage: float = Field(default=1, ge=1)
    side: str = "long"


class DecisionPayload(BaseModel):
    probabilities: dict[str, float]
    equity: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    stop_price: float = Field(gt=0)
    requested_notional: float = Field(gt=0)
    gross_exposure: float = Field(default=0, ge=0)
    daily_pnl_fraction: float = 0
    current_drawdown: float = Field(default=0, ge=0)
    leverage: float = Field(default=1, ge=1)
    current_positions: dict[str, float] = Field(default_factory=dict)
    symbol: str = Field(min_length=1)
    correlated_group_exposure: float = Field(default=0, ge=0)
    confidence_threshold: float = Field(default=0.55, ge=0.5, le=1)


@app.get("/v1/audit/events")
def list_audit_events(limit: int = Query(default=100, ge=1, le=500),
                      account: dict[str, str] = Depends(current_account)):
    try:
        return {"items": _memory_store().list_audit_events(account["id"], limit)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok", "version": AppConfig().version, "mode": "research_only"}


@app.get("/markets/supported")
def supported_markets():
    return {"venues": ["binance", "mexc"], "asset_classes": ["crypto"],
            "market_types": ["spot"], "other_asset_classes": "CSV historical data only"}


@app.get("/providers/capabilities")
def provider_capabilities():
    return {"live_public_provider": ["crypto spot via Binance/MEXC"],
            "historical_csv_adapter": ["crypto", "forex", "equity", "commodity", "index"],
            "not_connected": ["forex broker feeds", "equity broker feeds", "commodity feeds"]}


@app.get("/risk/config")
def risk_config():
    return risk_engine.snapshot()


@app.post("/risk/assess")
def assess_risk(payload: RiskPayload):
    result = risk_engine.assess(RiskRequest(**payload.model_dump()))
    return {"approved": result.approved, "approved_notional": result.approved_notional,
            "quantity": result.quantity, "reasons": result.reasons}


@app.post("/portfolio/assess")
def portfolio_assess(payload: PortfolioPayload):
    result = assess_portfolio(payload.equity, payload.current_positions,
                              payload.proposed_symbol, payload.proposed_notional,
                              payload.correlated_group_exposure)
    return {"approved": result.approved, "reasons": result.reasons,
            "resulting_gross_fraction": result.resulting_gross_fraction}


@app.post("/decision/assess")
def decision_assess(payload: DecisionPayload):
    data = payload.model_dump()
    probabilities = data.pop("probabilities")
    equity, entry, stop = data.pop("equity"), data.pop("entry_price"), data.pop("stop_price")
    notional, gross = data.pop("requested_notional"), data.pop("gross_exposure")
    daily, drawdown, leverage = data.pop("daily_pnl_fraction"), data.pop("current_drawdown"), data.pop("leverage")
    positions, symbol = data.pop("current_positions"), data.pop("symbol")
    correlation, threshold = data.pop("correlated_group_exposure"), data.pop("confidence_threshold")
    result = build_research_decision(
        probabilities, RiskRequest(equity, entry, stop, notional, gross,
                                   daily, drawdown, leverage), risk_engine=risk_engine,
        confidence_threshold=threshold, current_positions=positions,
        proposed_symbol=symbol, correlated_group_exposure=correlation)
    return result


@app.get("/execution/status")
def execution_status():
    return {"enabled": False, "live_orders_implemented": False,
            "message": "ETI is research-only. No order execution endpoints are available."}


web_dir = Path(__file__).resolve().parent.parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
