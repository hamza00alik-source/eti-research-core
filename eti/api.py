"""Research API and PWA host. No exchange order routes exist."""
from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .config import AppConfig
from .decision import build_research_decision
from .risk import RiskEngine, RiskRequest
from .portfolio import assess_portfolio

app = FastAPI(title=AppConfig().app_name, version=AppConfig().version)
risk_engine = RiskEngine()

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
                                   daily, drawdown, leverage),
        risk_engine=risk_engine, confidence_threshold=threshold,
        current_positions=positions, proposed_symbol=symbol,
        correlated_group_exposure=correlation)
    return result

@app.get("/execution/status")
def execution_status():
    return {"enabled": False, "live_orders_implemented": False,
            "message": "ETI is research-only. No order execution endpoints are available."}

web_dir = Path(__file__).resolve().parent.parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
