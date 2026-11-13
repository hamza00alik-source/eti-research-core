"""Owner-controlled, research-only paper trading on public crypto spot data."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .artifacts import ModelArtifactStore
from .config import RiskConfig
from .features import build_features
from .portfolio import PortfolioLimits, assess_portfolio
from .providers import CcxtPublicProvider
from .risk import RiskEngine, RiskRequest


def run_paper_tick(payload: dict[str, Any], store) -> dict[str, Any]:
    owner_id = payload.get("owner_id")
    session_id = payload.get("session_id")
    if not isinstance(owner_id, str) or not owner_id or not isinstance(session_id, str) or not session_id:
        raise ValueError("owner_id and session_id are required")
    session = store.get_paper_session(owner_id, session_id)
    if session is None:
        raise LookupError("paper session not found")
    if session["status"] != "running":
        return {"status": session["status"], "processed": False, "reason": "session_not_running"}

    metadata = store.get_model_version(session["model_id"], session["model_version"])
    if metadata is None or metadata["lifecycle"] != "approved":
        store.set_paper_session_status(owner_id, session_id, "stopped")
        return {"status": "stopped", "processed": False, "reason": "model_not_approved"}

    frame = CcxtPublicProvider(session["venue"]).fetch_ohlcv(
        session["symbol"], session["timeframe"], 200
    )
    features = build_features(frame).dropna()
    if features.empty:
        raise ValueError("no completed candle has a full feature set")
    schema = metadata["input_schema"].get("features")
    if not isinstance(schema, list) or not schema or list(features.columns) != schema:
        raise ValueError("approved model feature schema does not match current ETI features")
    latest = features.iloc[[-1]]
    probabilities = np.asarray(
        ModelArtifactStore().load_model(
            metadata["artifact_uri"], metadata["artifact_sha256"]
        ).predict_proba(latest),
        dtype=float,
    )
    if probabilities.shape != (1, 3) or not np.isfinite(probabilities).all():
        raise ValueError("approved model returned invalid class probabilities")
    probabilities = probabilities[0]
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])
    price = float(frame.iloc[-1]["close"])
    candle_at = frame.iloc[-1]["timestamp"].isoformat()
    if session["last_processed_candle"] and candle_at <= session["last_processed_candle"]:
        return {"status": session["status"], "processed": False, "reason": "candle_already_processed"}

    risk_values = session["risk_config"]
    risk_config = RiskConfig(
        max_risk_per_trade=float(risk_values["max_risk_per_trade"]),
        max_position_fraction=float(risk_values["max_position_fraction"]),
        max_gross_exposure=float(risk_values["max_gross_exposure"]),
        max_daily_loss=float(risk_values["max_daily_loss"]),
        max_drawdown=float(risk_values["max_drawdown"]),
        max_leverage=1.0,
        allow_short=False,
        allow_live_orders=False,
    )
    fee_bps = float(risk_values.get("fee_bps", 6.0))
    slippage_bps = float(risk_values.get("slippage_bps", 8.0))
    confidence_threshold = float(risk_values.get("confidence_threshold", 0.55))
    if (not np.isfinite([fee_bps, slippage_bps, confidence_threshold]).all()
            or not 0 <= fee_bps <= 100 or not 0 <= slippage_bps <= 100
            or not 0.50 <= confidence_threshold <= 0.95):
        raise ValueError("paper fee, slippage, or confidence settings are outside safe bounds")
    fee_rate, slippage_rate = fee_bps / 10000.0, slippage_bps / 10000.0

    cash = float(session["cash"])
    quantity = float(session["position_qty"])
    entry_price = session["entry_price"]
    position_cost = float(session["position_cost"])
    realized_pnl = float(session["realized_pnl"])
    peak_equity = float(session["peak_equity"])
    daily_date = candle_at[:10]
    daily_start_equity = float(session["daily_start_equity"])
    if session["daily_date"] != daily_date:
        daily_date, daily_start_equity = daily_date, float(session["equity"])

    mark_equity = cash + quantity * price
    peak_equity = max(peak_equity, mark_equity)
    drawdown = max(0.0, 1.0 - mark_equity / peak_equity) if peak_equity > 0 else 1.0
    daily_pnl_fraction = (
        (mark_equity - daily_start_equity) / daily_start_equity
        if daily_start_equity > 0 else -1.0
    )
    events: list[dict[str, Any]] = []
    decision_detail: dict[str, Any] = {
        "approved": False,
        "reason": "below_confidence_threshold_or_flat",
        "prediction": prediction,
        "confidence": confidence,
        "live_execution": False,
    }

    risk_breach = (
        daily_pnl_fraction <= -risk_config.max_daily_loss
        or drawdown >= risk_config.max_drawdown
    )
    should_exit = quantity > 0 and (
        prediction != 2 or confidence < confidence_threshold or risk_breach
    )
    if should_exit:
        fill_price = price * (1.0 - slippage_rate)
        notional = quantity * fill_price
        fee = notional * fee_rate
        pnl = notional - fee - position_cost
        cash += notional - fee
        realized_pnl += pnl
        events.append({
            "event_type": "risk_stop" if risk_breach else "exit",
            "quantity": quantity, "price": fill_price, "notional": notional,
            "fee": fee, "slippage_bps": slippage_bps, "realized_pnl": pnl,
            "prediction": prediction, "confidence": confidence,
            "risk_decision": {"reason": "daily_loss_or_drawdown_limit" if risk_breach else "model_signal_exit",
                              "live_execution": False},
        })
        quantity, entry_price, position_cost = 0.0, None, 0.0

    if quantity <= 0 and prediction == 2 and confidence >= confidence_threshold and not risk_breach:
        requested = mark_equity * risk_config.max_position_fraction
        stop_price = price * 0.95
        risk_decision = RiskEngine(risk_config).assess(RiskRequest(
            equity=max(mark_equity, 1e-9),
            entry_price=price,
            stop_price=stop_price,
            requested_notional=requested,
            gross_exposure=0.0,
            daily_pnl_fraction=daily_pnl_fraction,
            current_drawdown=drawdown,
            leverage=1.0,
            side="long",
        ))
        portfolio = assess_portfolio(
            equity=max(mark_equity, 1e-9),
            current_positions={},
            proposed_symbol=session["symbol"],
            proposed_notional=risk_decision.approved_notional,
            limits=PortfolioLimits(
                max_positions=1,
                max_gross_exposure_fraction=risk_config.max_gross_exposure,
                max_single_asset_fraction=risk_config.max_position_fraction,
                max_correlated_group_fraction=risk_config.max_position_fraction,
            ),
        ) if risk_decision.approved else None
        allowed_notional = min(
            risk_decision.approved_notional if risk_decision.approved else 0.0,
            cash / (1.0 + fee_rate) if cash > 0 else 0.0,
        )
        if risk_decision.approved and portfolio and portfolio.approved and allowed_notional > 0:
            fill_price = price * (1.0 + slippage_rate)
            fee = allowed_notional * fee_rate
            quantity = allowed_notional / fill_price
            cash -= allowed_notional + fee
            entry_price = fill_price
            position_cost = allowed_notional + fee
            decision_detail = {
                "approved": True, "approved_notional": allowed_notional,
                "quantity": quantity, "risk_reasons": list(risk_decision.reasons),
                "portfolio_approved": portfolio.approved, "live_execution": False,
            }
            events.append({
                "event_type": "entry", "quantity": quantity, "price": fill_price,
                "notional": allowed_notional, "fee": fee, "slippage_bps": slippage_bps,
                "realized_pnl": 0.0, "prediction": prediction, "confidence": confidence,
                "risk_decision": decision_detail,
            })
        else:
            decision_detail = {
                "approved": False,
                "risk_reasons": list(risk_decision.reasons),
                "portfolio_reasons": list(portfolio.reasons) if portfolio else [],
                "live_execution": False,
            }

    equity = cash + quantity * price
    peak_equity = max(peak_equity, equity)
    max_drawdown = max(float(session["max_drawdown"]),
                       max(0.0, 1.0 - equity / peak_equity) if peak_equity > 0 else 1.0)
    daily_pnl_fraction = (
        (equity - daily_start_equity) / daily_start_equity
        if daily_start_equity > 0 else -1.0
    )
    status = "stopped" if (
        daily_pnl_fraction <= -risk_config.max_daily_loss
        or max_drawdown >= risk_config.max_drawdown
    ) else "running"
    updated = store.apply_paper_tick(
        owner_id, session_id, candle_at,
        {
            "cash": cash, "position_qty": quantity, "entry_price": entry_price,
            "position_cost": position_cost, "status": status, "equity": equity,
            "peak_equity": peak_equity, "realized_pnl": realized_pnl,
            "max_drawdown": max_drawdown, "daily_date": daily_date,
            "daily_start_equity": daily_start_equity,
        },
        events,
    )
    return {
        "status": updated["status"], "processed": True,
        "session_id": session_id, "symbol": session["symbol"],
        "candle_at": candle_at, "price": price,
        "prediction": prediction, "probabilities": {
            "short": float(probabilities[0]), "flat": float(probabilities[1]),
            "long": float(probabilities[2]),
        },
        "confidence": confidence, "decision": decision_detail,
        "equity": updated["equity"], "cash": updated["cash"],
        "position_qty": updated["position_qty"],
        "realized_pnl": updated["realized_pnl"],
        "max_drawdown": updated["max_drawdown"],
        "events": events, "live_execution": False,
    }
