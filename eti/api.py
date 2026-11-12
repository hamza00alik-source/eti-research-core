"""Minimal research API. All endpoints are read-only; no exchange order routes exist."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import AppConfig
from .risk import RiskEngine, RiskRequest

app = FastAPI(title=AppConfig().app_name, version=AppConfig().version)
risk_engine = RiskEngine()


class RiskPayload(BaseModel):
    equity: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    stop_price: float = Field(gt=0)
    requested_notional: float = Field(gt=0)
    gross_exposure: float = Field(default=0, ge=0)
    daily_pnl_fraction: float = 0
    current_drawdown: float = Field(default=0, ge=0)
    leverage: float = Field(default=1, gt=0)
    side: str = "long"


@app.get("/health")
def health():
    return {"status": "ok", "version": AppConfig().version, "mode": "research_only"}


@app.get("/markets/supported")
def supported_markets():
    return {"venues": ["binance", "mexc"], "asset_classes": ["crypto"], "market_types": ["spot"]}


@app.get("/risk/config")
def risk_config():
    return risk_engine.snapshot()


@app.post("/risk/assess")
def assess_risk(payload: RiskPayload):
    request = RiskRequest(**payload.model_dump())
    result = risk_engine.assess(request)
    return {"approved": result.approved, "approved_notional": result.approved_notional,
            "quantity": result.quantity, "reasons": result.reasons}


@app.get("/execution/status")
def execution_status():
    return {"enabled": False, "live_orders_implemented": False,
            "message": "ETI is research-only. No order execution endpoints are available."}
