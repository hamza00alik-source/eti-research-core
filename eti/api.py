"""Research API and PWA host. No exchange order routes exist."""
from __future__ import annotations
from pathlib import Path
import sqlite3
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .auth import AuthManager, AuthenticationError, RegistrationDisabled
from .config import AppConfig
from .decision import build_research_decision
from .portfolio import assess_portfolio
from .risk import RiskEngine, RiskRequest
from .storage import MemoryStore

app = FastAPI(title=AppConfig().app_name, version=AppConfig().version)
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
def register(payload: RegisterPayload):
    manager = _auth_manager()
    try:
        account = manager.register(payload.username, payload.password, payload.invite_code)
    except RegistrationDisabled as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"account": account, "access_token": manager.issue_token(account),
            "token_type": "bearer", "expires_in": manager.token_ttl_seconds}


@app.post("/v1/auth/token")
def login(payload: LoginPayload):
    manager = _auth_manager()
    try:
        account = manager.authenticate(payload.username, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="invalid username or password",
                            headers={"WWW-Authenticate": "Bearer"}) from exc
    return {"account": account, "access_token": manager.issue_token(account),
            "token_type": "bearer", "expires_in": manager.token_ttl_seconds}


@app.get("/v1/profile")
def profile(account: dict[str, str] = Depends(current_account)):
    return {"account": account}


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
            payload.target_capability, payload.input, payload.deadline_at)
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
    try:
        return _memory_store().update_agent_task(
            account["id"], workspace_id, task_id, payload.status, payload.result)
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
