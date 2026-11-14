"""Research API and PWA host. No live order placement routes exist."""
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
from pydantic import BaseModel, Field, SecretStr

from .agents import default_capabilities, derivatives_market_context, discover_market_universe, market_coverage_report
from .artifacts import ModelArtifactError, ModelArtifactStore
from .auth import AuthManager, AuthenticationError, RateLimitExceeded, RegistrationDisabled
from .config import AppConfig
from .credentials import CredentialVault, CredentialVaultError, verify_exchange_credentials
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
        "img-src 'self' data:; connect-src 'self' https://api.binance.com wss://stream.binance.com:9443; manifest-src 'self'; "
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


class AutonomyConfigPayload(BaseModel):
    enabled: bool = True
    workspace_id: str | None = Field(default=None, min_length=1, max_length=128)
    venues: list[str] = Field(default_factory=lambda: ["mexc"], min_length=1, max_length=2)
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"], min_length=1, max_length=20)
    timeframes: list[str] = Field(default_factory=lambda: ["1h"], min_length=1, max_length=6)
    market_types: list[str] = Field(default_factory=lambda: ["spot", "swap"], min_length=1, max_length=3)
    interval_seconds: int = Field(default=900, ge=300, le=86400)
    model_training_enabled: bool = False
    model_training_interval_seconds: int = Field(default=86400, ge=86400, le=604800)


class ExchangeCredentialsPayload(BaseModel):
    venue: str = Field(pattern="^(binance|mexc)$")
    api_key: SecretStr = Field(min_length=8, max_length=256)
    api_secret: SecretStr = Field(min_length=8, max_length=256)


class LiveExecutionSettingsPayload(BaseModel):
    venue: str = Field(default="binance", pattern="^(binance|mexc)$")
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"], min_length=1, max_length=8)
    timeframe: str = Field(default="1h", pattern="^(1h|4h|1d)$")
    capital_allocation_usdt: float = Field(default=100.0, ge=20, le=1000000)
    max_risk_per_trade: float = Field(default=0.005, gt=0, le=0.005)
    max_position_fraction: float = Field(default=0.10, gt=0, le=0.10)
    max_gross_exposure: float = Field(default=0.25, gt=0, le=0.25)
    max_daily_loss: float = Field(default=0.02, gt=0, le=0.02)
    max_drawdown: float = Field(default=0.08, gt=0, le=0.08)
    confidence_threshold: float = Field(default=0.60, ge=0.55, le=0.95)


class LiveExecutionAuthorizationPayload(BaseModel):
    confirm_live_trading: bool = False
    confirm_spot_trading_permission: bool = False
    confirm_withdrawals_disabled: bool = False
    confirm_ip_restrictions: bool = False


class MemoryPayload(BaseModel):
    value: Any


class AgentTaskPayload(BaseModel):
    target_capability: str = Field(min_length=1, max_length=120)
    input: Any
    deadline_at: str | None = Field(default=None, max_length=40)
    max_attempts: int = Field(default=3, ge=1, le=10)

class SolanaResearchEventPayload(BaseModel):
    event_key: str = Field(min_length=1, max_length=200)
    source: str = Field(default="pumpportal", min_length=1, max_length=40)
    event_type: str = Field(min_length=1, max_length=20)
    mint: str = Field(min_length=32, max_length=64)
    source_timestamp: str | None = Field(default=None, max_length=80)
    payload: dict[str, Any]



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




@app.get("/v1/autonomy")
def get_autonomy_status(account: dict[str, str] = Depends(current_account)):
    profile = _memory_store().get_autonomy_profile(account["id"])
    return {"profile": profile, "mode": "research_only",
            "live_execution_enabled": False,
            "note": "Autonomy currently schedules public market research only; it cannot place orders."}


@app.put("/v1/autonomy")
def set_autonomy_config(payload: AutonomyConfigPayload,
                        account: dict[str, str] = Depends(current_account)):
    store = _memory_store()
    workspaces = store.list_workspaces(account["id"])
    workspace_id = payload.workspace_id
    if workspace_id is None:
        workspace_id = workspaces[0]["id"] if workspaces else store.create_workspace(
            account["id"], "ETI Autonomous Research"
        )["id"]
    try:
        profile = store.configure_autonomy(
            account["id"], workspace_id, payload.enabled, payload.venues,
            payload.symbols, payload.timeframes, payload.interval_seconds,
            payload.model_training_enabled, payload.model_training_interval_seconds,
            market_types=payload.market_types
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="workspace not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"profile": profile, "mode": "research_only",
            "live_execution_enabled": False,
            "note": "The worker schedules public research scans and, when explicitly enabled, bounded daily model training. Training artifacts remain unapproved candidates. No exchange credentials or orders are used."}





@app.get("/v1/exchanges/credentials")
def list_exchange_credentials(account: dict[str, str] = Depends(current_account)):
    _require_model_admin(account)
    return {"items": _memory_store().list_exchange_credentials(account["id"]),
            "note": "Credential secrets are encrypted at rest and are never returned by this API. Trading permission is not verified by the read-only connection test."}


@app.post("/v1/exchanges/credentials", status_code=201)
def save_exchange_credentials(payload: ExchangeCredentialsPayload,
                              account: dict[str, str] = Depends(current_account)):
    _require_model_admin(account)
    try:
        vault = CredentialVault()
    except CredentialVaultError as exc:
        raise HTTPException(status_code=503, detail="Server credential encryption is not configured") from exc
    api_key = payload.api_key.get_secret_value()
    api_secret = payload.api_secret.get_secret_value()
    try:
        verification = verify_exchange_credentials(payload.venue, api_key, api_secret)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except CredentialVaultError as exc:
        raise HTTPException(status_code=422, detail="Exchange rejected the credentials or private read permission check") from exc
    stored = _memory_store().upsert_exchange_credential(
        account["id"], payload.venue,
        vault.encrypt(api_key, account["id"], payload.venue, "api_key"),
        vault.encrypt(api_secret, account["id"], payload.venue, "api_secret"),
        api_key[-4:],
    )
    return {"credential": stored, "verification": verification,
            "note": "Read access verified only. ETI has not verified trading permission and cannot place orders from this connection yet."}


@app.delete("/v1/exchanges/credentials/{venue}")
def delete_exchange_credentials(venue: str,
                                account: dict[str, str] = Depends(current_account)):
    _require_model_admin(account)
    if venue not in {"binance", "mexc"}:
        raise HTTPException(status_code=422, detail="unsupported exchange venue")
    deleted = _memory_store().delete_exchange_credential(account["id"], venue)
    if not deleted:
        raise HTTPException(status_code=404, detail="exchange credentials not found")
    return {"deleted": True, "venue": venue}




@app.get("/v1/execution/profile")
def get_live_execution_profile(account: dict[str, str] = Depends(current_account)):
    _require_model_admin(account)
    profile = _memory_store().get_live_execution_profile(account["id"])
    return {"profile": profile, "mode": "live" if profile and profile["enabled"] else "disabled",
            "per_trade_approval": False, "live_execution_available": True}


@app.put("/v1/execution/profile")
def save_live_execution_profile(payload: LiveExecutionSettingsPayload,
                                account: dict[str, str] = Depends(current_account)):
    _require_model_admin(account)
    risk_config = {
        "max_risk_per_trade": payload.max_risk_per_trade,
        "max_position_fraction": payload.max_position_fraction,
        "max_gross_exposure": payload.max_gross_exposure,
        "max_daily_loss": payload.max_daily_loss,
        "max_drawdown": payload.max_drawdown,
        "confidence_threshold": payload.confidence_threshold,
    }
    try:
        profile = _memory_store().save_live_execution_profile(
            account["id"], payload.venue, payload.symbols, payload.timeframe,
            payload.capital_allocation_usdt, risk_config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"profile": profile, "mode": "disabled",
            "note": "Settings are saved but live trading is still disabled. Authorize once to let ETI execute within these hard limits."}


@app.post("/v1/execution/authorize")
def authorize_live_execution(payload: LiveExecutionAuthorizationPayload,
                             account: dict[str, str] = Depends(current_account)):
    _require_model_admin(account)
    if not all((payload.confirm_live_trading, payload.confirm_spot_trading_permission,
                payload.confirm_withdrawals_disabled, payload.confirm_ip_restrictions)):
        raise HTTPException(status_code=422, detail="All one-time live trading and API-key safety confirmations are required")
    try:
        profile = _memory_store().authorize_live_execution(account["id"])
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="live execution settings not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"profile": profile, "mode": "live", "per_trade_approval": False,
            "note": "ETI may execute eligible spot orders automatically within the saved capital allocation and hard risk limits. The kill switch remains owner-controlled."}


@app.post("/v1/execution/kill-switch")
def live_execution_kill_switch(account: dict[str, str] = Depends(current_account)):
    _require_model_admin(account)
    profile = _memory_store().disable_live_execution(account["id"], "owner_kill_switch")
    if profile is None:
        raise HTTPException(status_code=404, detail="live execution profile not found")
    return {"profile": profile, "mode": "disabled", "message": "Live order scheduling is disabled."}


@app.get("/v1/execution/orders")
def list_live_execution_orders(limit: int = Query(default=100, ge=1, le=500),
                               account: dict[str, str] = Depends(current_account)):
    _require_model_admin(account)
    return {"items": _memory_store().list_live_orders(account["id"], limit),
            "per_trade_approval": False}


@app.get("/v1/paper/sessions")
def list_paper_sessions(workspace_id: str | None = None,
                        account: dict[str, str] = Depends(current_account)):
    workspaces = _memory_store().list_workspaces(account["id"])
    selected = workspace_id or (workspaces[0]["id"] if workspaces else None)
    if selected is None:
        return {"items": []}
    return {"items": _memory_store().list_paper_sessions(account["id"], selected),
            "mode": "paper_only", "live_execution_enabled": False}


@app.post("/v1/paper/sessions", status_code=201)
def create_paper_session(payload: PaperSessionPayload,
                         account: dict[str, str] = Depends(current_account)):
    store = _memory_store()
    workspaces = store.list_workspaces(account["id"])
    workspace_id = payload.workspace_id or (workspaces[0]["id"] if workspaces else None)
    if workspace_id is None:
        workspace_id = store.create_workspace(account["id"], "ETI Paper Trading")["id"]
    risk_config = {
        "max_risk_per_trade": payload.max_risk_per_trade,
        "max_position_fraction": payload.max_position_fraction,
        "max_gross_exposure": payload.max_gross_exposure,
        "max_daily_loss": payload.max_daily_loss,
        "max_drawdown": payload.max_drawdown,
        "fee_bps": payload.fee_bps,
        "slippage_bps": payload.slippage_bps,
        "confidence_threshold": payload.confidence_threshold,
        "allow_short": False,
        "allow_live_orders": False,
    }
    try:
        session = store.create_paper_session(
            account["id"], workspace_id, payload.model_id, payload.model_version,
            payload.venue, payload.symbol, payload.timeframe, payload.initial_cash,
            risk_config,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="workspace not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"session": session, "mode": "paper_only", "live_execution_enabled": False,
            "note": "This uses simulated funds and public market data only. It cannot place exchange orders."}


@app.post("/v1/paper/sessions/{session_id}/start")
def start_paper_session(session_id: str,
                        account: dict[str, str] = Depends(current_account)):
    try:
        session = _memory_store().set_paper_session_status(account["id"], session_id, "running")
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="paper session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"session": session, "mode": "paper_only", "live_execution_enabled": False}


@app.post("/v1/paper/sessions/{session_id}/pause")
def pause_paper_session(session_id: str,
                        account: dict[str, str] = Depends(current_account)):
    try:
        session = _memory_store().set_paper_session_status(account["id"], session_id, "paused")
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="paper session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"session": session, "mode": "paper_only", "live_execution_enabled": False}


@app.get("/v1/paper/sessions/{session_id}/trades")
def list_paper_trades(session_id: str, limit: int = Query(default=200, ge=1, le=1000),
                      account: dict[str, str] = Depends(current_account)):
    session = _memory_store().get_paper_session(account["id"], session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="paper session not found")
    return {"items": _memory_store().list_paper_trades(account["id"], session_id, limit),
            "mode": "paper_only", "live_execution_enabled": False}


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


@app.get("/v1/workspaces/{workspace_id}/solana/events")
def list_solana_research_events(
    workspace_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    mint: str | None = Query(default=None, min_length=32, max_length=64),
    account: dict[str, str] = Depends(current_account),
):
    try:
        items = _memory_store().list_solana_research_events(
            account["id"], workspace_id, limit=limit, mint=mint
        )
        return {"items": items}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="workspace not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v1/workspaces/{workspace_id}/solana/snapshots")
def list_solana_research_snapshots(
    workspace_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    mint: str | None = Query(default=None, min_length=32, max_length=64),
    account: dict[str, str] = Depends(current_account),
):
    try:
        items = _memory_store().list_solana_research_snapshots(
            account["id"], workspace_id, limit=limit, mint=mint
        )
        return {"items": items}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="workspace not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/workspaces/{workspace_id}/solana/events", status_code=201)
def record_solana_research_event(
    workspace_id: str,
    payload: SolanaResearchEventPayload,
    account: dict[str, str] = Depends(current_account),
):
    try:
        return _memory_store().record_solana_research_event(
            account["id"], workspace_id, payload.event_key, payload.source,
            payload.event_type, payload.mint, payload.payload, payload.source_timestamp,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="workspace not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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

class PaperSessionPayload(BaseModel):
    model_id: str = Field(min_length=1, max_length=120)
    model_version: str = Field(min_length=1, max_length=80)
    venue: str = Field(default="mexc", pattern="^(binance|mexc)$")
    symbol: str = Field(min_length=3, max_length=40)
    timeframe: str = Field(default="1h", pattern="^(1h|4h|1d)$")
    initial_cash: float = Field(gt=0, le=1000000000)
    workspace_id: str | None = Field(default=None, min_length=1, max_length=128)
    max_risk_per_trade: float = Field(default=0.005, gt=0, le=0.005)
    max_position_fraction: float = Field(default=0.10, gt=0, le=0.10)
    max_gross_exposure: float = Field(default=0.25, gt=0, le=0.25)
    max_daily_loss: float = Field(default=0.02, gt=0, le=0.02)
    max_drawdown: float = Field(default=0.08, gt=0, le=0.08)
    fee_bps: float = Field(default=6.0, ge=0, le=100)
    slippage_bps: float = Field(default=8.0, ge=0, le=100)
    confidence_threshold: float = Field(default=0.55, ge=0.50, le=0.95)


class PaperSessionStatusPayload(BaseModel):
    status: str = Field(pattern="^(running|paused)$")


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
    return {
        "venues": ["binance", "mexc"],
        "asset_classes": ["crypto", "forex", "equity", "commodity", "index"],
        "crypto_public_market_types": ["spot", "swap", "future"],
        "market_types": ["spot", "swap", "future"],
        "catalog_market_types": ["spot", "swap", "future"],
        "not_implemented_market_types": ["margin", "option", "cfd"],
        "data_coverage": {
            "crypto": "Public CCXT OHLCV for supported spot, perpetual swap and dated-future symbols",
            "forex_equity_commodity_index": "Yahoo Finance public OHLCV plus validated CSV fallback; authenticated broker feeds are not connected",
        },
        "execution_market_types": ["spot"],
        "live_execution_enabled": False,
    }


@app.get("/v1/markets/coverage")
def get_market_coverage(account: dict[str, str] = Depends(current_account)):
    """Return the implemented research and live-execution capability matrix."""
    return market_coverage_report({})


@app.get("/v1/markets/discover")
def discover_markets(
    asset_class: str = Query(default="crypto", pattern="^(crypto|forex|equity|commodity|index)$"),
    venue: str = Query(default="binance", pattern="^(binance|mexc)$"),
    market_type: str = Query(default="spot", pattern="^(spot|swap|future)$"),
    limit: int = Query(default=100, ge=1, le=500),
    account: dict[str, str] = Depends(current_account),
):
    try:
        return discover_market_universe({
            "asset_class": asset_class, "venue": venue, "market_type": market_type, "limit": limit,
        })
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Public exchange market discovery is temporarily unavailable") from exc


@app.get("/v1/markets/derivatives/context")
def get_derivatives_context(
    venue: str = Query(default="binance", pattern="^(binance|mexc)$"),
    market_type: str = Query(default="swap", pattern="^(swap|future)$"),
    symbol: str = Query(min_length=3, max_length=40),
    timeframe: str = Query(default="1h", pattern="^(1h|4h|1d)$"),
    limit: int = Query(default=100, ge=1, le=1000),
    account: dict[str, str] = Depends(current_account),
):
    try:
        return derivatives_market_context({
            "venue": venue, "market_type": market_type, "symbol": symbol,
            "timeframe": timeframe, "limit": limit,
        })
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Public derivatives data is temporarily unavailable") from exc


@app.get("/providers/capabilities")
def provider_capabilities():
    return {
        "public_market_data_provider": [
            "crypto spot via Binance/MEXC where listed",
            "crypto perpetual swaps via Binance/MEXC where listed",
            "crypto dated futures via Binance/MEXC where listed",
            "forex, equities, commodities and indices via Yahoo Finance public OHLCV",
        ],
        "historical_csv_adapter": ["crypto", "forex", "equity", "commodity", "index"],
        "autonomous_reference_universe": {
            "forex": ["EURUSD=X"], "equity": ["AAPL"],
            "commodity": ["GC=F"], "index": ["^GSPC"],
            "crypto": "Owner-configured Binance/MEXC spot and perpetual-swap watchlist",
        },
        "daily_model_training_budget": 10,
        "not_connected": ["authenticated forex broker feeds", "authenticated equity broker feeds", "authenticated commodity broker feeds", "authenticated index broker feeds"],
        "data_quality_notes": ["Yahoo Finance coverage and history vary by instrument and interval", "derivatives funding and open-interest history are queried where CCXT reports support", "liquidation history is not yet incorporated into model features", "exchange-specific margin tiers and side-aware funding settlement remain incomplete"],
        "discovery": "Authenticated discovery covers active Binance/MEXC crypto spot, swap and dated-future listings plus curated Yahoo Finance forex, equity, commodity and index reference universes",
        "derivatives_research": ["historical funding rate features", "open-interest percentage-change features", "conservative absolute funding drag in nested evaluation"],
        "live_execution": "spot only; derivatives execution is not implemented",
    }


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
