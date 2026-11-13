"""Autonomous, owner-authorized Binance/MEXC spot order execution.

This module is called only by the durable worker. It never exposes an HTTP
endpoint that accepts arbitrary order instructions. Every order is gated by
the persisted owner authorization, an approved calibrated model, current
account state, deterministic risk checks and an idempotent order intent.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from .artifacts import ModelArtifactStore
from .config import RiskConfig
from .credentials import CredentialVault, CredentialVaultError
from .data import drop_incomplete_candles
from .features import build_features
from .portfolio import PortfolioLimits, assess_portfolio
from .risk import RiskEngine, RiskRequest


def _total_balance(balance: dict[str, Any], currency: str) -> float:
    total = balance.get("total") or {}
    if currency in total and total[currency] is not None:
        return max(0.0, float(total[currency]))
    entry = balance.get(currency) or {}
    return max(0.0, float(entry.get("total") or 0.0))


def _free_balance(balance: dict[str, Any], currency: str) -> float:
    free = balance.get("free") or {}
    if currency in free and free[currency] is not None:
        return max(0.0, float(free[currency]))
    entry = balance.get(currency) or {}
    return max(0.0, float(entry.get("free") or 0.0))


def _approved_model(store, venue: str, symbol: str, timeframe: str) -> dict[str, Any] | None:
    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    model_id = f"eti-{venue}-{safe_symbol}-{timeframe}".lower()
    versions = [
        item for item in store.list_model_versions("approved")
        if item.get("model_id") == model_id
    ]
    return max(versions, key=lambda item: item.get("updated_at", ""), default=None)


def _validated_probabilities(model: Any, features: pd.DataFrame, expected: list[str]) -> np.ndarray:
    if list(features.columns) != expected:
        raise ValueError("approved model feature schema does not match current ETI features")
    checker = getattr(model, "has_valid_calibration", None)
    calibrated = bool(checker()) if callable(checker) else getattr(model, "calibration_status", None) == "calibrated"
    if not calibrated:
        raise ValueError("approved model calibration is not verified")
    values = np.asarray(model.predict_proba(features.iloc[[-1]]), dtype=float)
    if (values.shape != (1, 3) or not np.isfinite(values).all()
            or np.any(values < 0)
            or not np.isclose(float(values.sum()), 1.0, rtol=1e-5, atol=1e-6)):
        raise ValueError("approved model returned invalid class probabilities")
    return values[0]


def _position_after_fill(state: dict[str, Any], order: dict[str, Any]) -> dict[str, Any]:
    next_state = dict(state)
    filled = max(0.0, float(order.get("filled") or 0.0))
    average = order.get("average") or order.get("result", {}).get("price")
    if order["side"] == "buy":
        old_qty = float(state.get("position_qty") or 0.0)
        old_entry = state.get("entry_price")
        new_qty = old_qty + filled
        if filled > 0 and average and new_qty > 0:
            next_state["entry_price"] = (
                (old_qty * float(old_entry or average) + filled * float(average)) / new_qty
            )
        next_state["position_qty"] = new_qty
    else:
        next_state["position_qty"] = max(0.0, float(state.get("position_qty") or 0.0) - filled)
        if next_state["position_qty"] <= 1e-12:
            next_state["position_qty"] = 0.0
            next_state["entry_price"] = None
    return next_state


def _reconcile_pending_orders(exchange, store, owner_id: str, state_for):
    pending = [
        item for item in store.list_live_orders(owner_id, 100)
        if item["status"] in {"submitting", "open", "unknown"}
    ]
    for item in pending:
        if item["status"] == "unknown" or not item.get("exchange_order_id"):
            store.disable_live_execution(owner_id, "order_state_uncertain")
            return {"blocked": True, "reason": "order_state_uncertain", "order_id": item["id"]}
        try:
            remote = exchange.fetch_order(item["exchange_order_id"], item["symbol"])
        except Exception:
            store.disable_live_execution(owner_id, "order_reconciliation_failed")
            return {"blocked": True, "reason": "order_reconciliation_failed", "order_id": item["id"]}
        status = remote.get("status")
        if status not in {"closed", "canceled", "rejected"}:
            store.update_live_order(
                owner_id, item["client_order_id"], "open",
                exchange_order_id=str(remote.get("id") or item["exchange_order_id"]),
                filled=float(remote.get("filled") or 0.0),
                average=remote.get("average"), cost=remote.get("cost"),
                result={"candle_at": item["result"].get("candle_at"),
                        "timeframe": item["result"].get("timeframe", "1h"),
                        "price": item["result"].get("price")},
            )
            return {"blocked": True, "reason": "order_still_open", "order_id": item["id"]}
        normalized = "closed" if status == "closed" else ("canceled" if status == "canceled" else "rejected")
        updated = store.update_live_order(
            owner_id, item["client_order_id"], normalized,
            exchange_order_id=str(remote.get("id") or item["exchange_order_id"]),
            filled=float(remote.get("filled") or 0.0),
            average=remote.get("average"), cost=remote.get("cost"),
            result={"candle_at": item["result"].get("candle_at"),
                    "timeframe": item["result"].get("timeframe", "1h"),
                    "price": item["result"].get("price")},
        )
        candle_at = item["result"].get("candle_at")
        if candle_at:
            state = state_for(item["venue"], item["symbol"])
            next_state = _position_after_fill(state, updated)
            store.apply_live_execution_state(
                owner_id, item["venue"], item["symbol"], item["result"].get("timeframe", "1h"),
                candle_at, next_state,
            )
    return {"blocked": False}


def run_live_tick(payload: dict[str, Any], store) -> dict[str, Any]:
    """Process one completed candle for an authorized live spot profile."""
    owner_id, venue = payload.get("owner_id"), payload.get("venue")
    symbol, timeframe = payload.get("symbol"), payload.get("timeframe")
    task_id = payload.get("task_id")
    if (not isinstance(owner_id, str) or not owner_id
            or venue not in {"binance", "mexc"}
            or not isinstance(symbol, str) or "/" not in symbol
            or timeframe not in {"1h", "4h", "1d"}):
        raise ValueError("live execution task input is invalid")
    profile = store.get_live_execution_profile(owner_id)
    if (not profile or not profile["enabled"] or profile["venue"] != venue
            or symbol not in profile["symbols"] or profile["timeframe"] != timeframe):
        return {"status": "disabled", "processed": False, "live_execution": False}
    metadata = _approved_model(store, venue, symbol, timeframe)
    if metadata is None:
        return {"status": "no_approved_model", "processed": False, "live_execution": False}

    credential = store.get_exchange_credential(owner_id, venue)
    if credential is None:
        store.disable_live_execution(owner_id, "exchange_credentials_missing")
        return {"status": "stopped", "reason": "exchange_credentials_missing", "live_execution": False}
    try:
        vault = CredentialVault()
        api_key = vault.decrypt(credential["api_key_ciphertext"], owner_id, venue, "api_key")
        api_secret = vault.decrypt(credential["api_secret_ciphertext"], owner_id, venue, "api_secret")
    except CredentialVaultError:
        store.disable_live_execution(owner_id, "credential_decryption_failed")
        return {"status": "stopped", "reason": "credential_decryption_failed", "live_execution": False}

    import ccxt
    exchange = getattr(ccxt, venue)({
        "apiKey": api_key, "secret": api_secret, "enableRateLimit": True,
        "timeout": 20000, "options": {"defaultType": "spot"},
    })
    try:
        pending = _reconcile_pending_orders(
            exchange, store, owner_id,
            lambda v, s: store.get_live_execution_state(owner_id, v, s, timeframe),
        )
        if pending["blocked"]:
            return {"status": "blocked", "reason": pending["reason"],
                    "order_id": pending.get("order_id"), "live_execution": True}

        balance = exchange.fetch_balance()
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=220)
        if not rows:
            raise ValueError("exchange returned no OHLCV candles")
        frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        timeframe_ms = int(exchange.parse_timeframe(timeframe) * 1000)
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        frame = drop_incomplete_candles(frame, timeframe_ms, now_ms)
        if len(frame) < 40:
            raise ValueError("exchange returned insufficient completed candles")
        last_open_ms = int(frame.iloc[-1]["timestamp"].timestamp() * 1000)
        if now_ms - (last_open_ms + timeframe_ms) > timeframe_ms * 2:
            raise ValueError("market candles are stale")
        features = build_features(frame).dropna()
        if features.empty:
            raise ValueError("no completed candle has a full feature set")
        expected = metadata["input_schema"].get("features")
        if not isinstance(expected, list) or not expected:
            raise ValueError("approved model input schema is invalid")
        model = ModelArtifactStore().load_model(metadata["artifact_uri"], metadata["artifact_sha256"])
        probabilities = _validated_probabilities(model, features, expected)
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])
        ticker = exchange.fetch_ticker(symbol)
        price = float(ticker.get("last") or frame.iloc[-1]["close"])
        if not np.isfinite(price) or price <= 0:
            raise ValueError("exchange returned an invalid live price")
        candle_at = frame.iloc[-1]["timestamp"].isoformat()
        state = store.get_live_execution_state(owner_id, venue, symbol, timeframe)
        if state["last_processed_candle"] and candle_at <= state["last_processed_candle"]:
            return {"status": "already_processed", "processed": False, "live_execution": True}
        other_positions = [
            item for item in store.list_live_execution_states(owner_id, venue, timeframe)
            if item["symbol"] != symbol and float(item["position_qty"]) > 1e-12
        ]
        if other_positions and float(state["position_qty"]) <= 1e-12:
            return {"status": "waiting_for_existing_position",
                    "active_symbol": other_positions[0]["symbol"], "live_execution": True}

        base_currency, quote_currency = symbol.split("/", 1)
        quote_total = _total_balance(balance, quote_currency)
        quote_free = _free_balance(balance, quote_currency)
        base_free = _free_balance(balance, base_currency)
        position_qty = float(state["position_qty"])
        allocation = float(profile["capital_allocation_usdt"])
        equity = min(allocation, quote_total + position_qty * price)
        if equity <= 0:
            return {"status": "no_allocated_equity", "processed": False, "live_execution": True}

        daily_date = candle_at[:10]
        daily_start = float(profile["daily_start_equity"] or equity)
        if profile["daily_date"] != daily_date:
            daily_start = equity
        peak_equity = max(float(profile["peak_equity"] or equity), equity)
        daily_pnl = (equity - daily_start) / daily_start if daily_start > 0 else -1.0
        drawdown = max(0.0, 1.0 - equity / peak_equity) if peak_equity > 0 else 1.0

        risk_values = profile["risk_config"]
        risk_config = RiskConfig(
            max_risk_per_trade=float(risk_values["max_risk_per_trade"]),
            max_position_fraction=float(risk_values["max_position_fraction"]),
            max_gross_exposure=float(risk_values["max_gross_exposure"]),
            max_daily_loss=float(risk_values["max_daily_loss"]),
            max_drawdown=float(risk_values["max_drawdown"]),
            max_leverage=1.0, allow_short=False, allow_live_orders=False,
        )
        confidence_threshold = float(risk_values["confidence_threshold"])
        stop_breach = position_qty > 0 and state["entry_price"] is not None and price <= float(state["entry_price"]) * 0.95
        risk_breach = daily_pnl <= -risk_config.max_daily_loss or drawdown >= risk_config.max_drawdown
        should_sell = position_qty > 0 and (
            prediction != 2 or confidence < confidence_threshold or stop_breach or risk_breach
        )
        should_buy = (
            position_qty <= 1e-12 and prediction == 2 and confidence >= confidence_threshold
            and not risk_breach
        )
        side, amount, risk_reasons = None, 0.0, []
        if should_sell:
            if base_free + 1e-10 < position_qty:
                store.disable_live_execution(owner_id, "tracked_position_balance_mismatch")
                return {"status": "stopped", "reason": "tracked_position_balance_mismatch", "live_execution": True}
            side = "sell"
            amount = position_qty
        elif should_buy:
            requested = min(
                equity * risk_config.max_position_fraction,
                allocation * risk_config.max_position_fraction,
                quote_free * 0.995,
            )
            decision = RiskEngine(risk_config).assess(RiskRequest(
                equity=equity, entry_price=price, stop_price=price * 0.95,
                requested_notional=requested, gross_exposure=position_qty * price,
                daily_pnl_fraction=daily_pnl, current_drawdown=drawdown,
                leverage=1.0, side="long",
            ))
            risk_reasons = list(decision.reasons)
            portfolio = assess_portfolio(
                equity=equity, current_positions={symbol: position_qty * price} if position_qty > 0 else {},
                proposed_symbol=symbol, proposed_notional=decision.approved_notional,
                limits=PortfolioLimits(
                    max_positions=1,
                    max_gross_exposure_fraction=risk_config.max_gross_exposure,
                    max_single_asset_fraction=risk_config.max_position_fraction,
                    max_correlated_group_fraction=risk_config.max_position_fraction,
                ),
            ) if decision.approved else None
            if decision.approved and portfolio and portfolio.approved:
                side = "buy"
                amount = decision.approved_notional / price
            else:
                risk_reasons.extend(list(portfolio.reasons) if portfolio else [])

        if risk_breach:
            store.disable_live_execution(owner_id, "daily_loss_or_drawdown_limit")
            if not should_sell:
                store.update_live_execution_account_state(owner_id, daily_date, daily_start, peak_equity)
                store.apply_live_execution_state(
                    owner_id, venue, symbol, timeframe, candle_at,
                    {"position_qty": position_qty, "entry_price": state["entry_price"],
                     "daily_date": daily_date, "daily_start_equity": daily_start,
                     "peak_equity": peak_equity},
                )
                return {"status": "stopped", "reason": "daily_loss_or_drawdown_limit", "live_execution": True}

        if side is None or amount <= 0:
            store.update_live_execution_account_state(owner_id, daily_date, daily_start, peak_equity)
            store.apply_live_execution_state(
                owner_id, venue, symbol, timeframe, candle_at,
                {"position_qty": position_qty, "entry_price": state["entry_price"],
                 "daily_date": daily_date, "daily_start_equity": daily_start,
                 "peak_equity": peak_equity},
            )
            return {"status": "no_order", "processed": True, "symbol": symbol,
                    "prediction": prediction, "confidence": confidence,
                    "risk_reasons": risk_reasons, "live_execution": True}

        exchange.load_markets()
        amount = float(exchange.amount_to_precision(symbol, amount))
        market = exchange.market(symbol)
        minimum_cost = ((market.get("limits") or {}).get("cost") or {}).get("min") or 0.0
        if amount <= 0 or amount * price < float(minimum_cost):
            store.update_live_execution_account_state(owner_id, daily_date, daily_start, peak_equity)
            store.apply_live_execution_state(
                owner_id, venue, symbol, timeframe, candle_at,
                {"position_qty": position_qty, "entry_price": state["entry_price"],
                 "daily_date": daily_date, "daily_start_equity": daily_start,
                 "peak_equity": peak_equity},
            )
            return {"status": "below_exchange_minimum", "processed": True, "live_execution": True}

        store.update_live_execution_account_state(owner_id, daily_date, daily_start, peak_equity)
        task_id = task_id if isinstance(task_id, str) and task_id else str(uuid4())
        client_order_id = "eti" + task_id.replace("-", "")[:29]
        intent, created = store.create_live_order_intent(
            owner_id, venue, symbol, side, amount, client_order_id,
        )
        if not created:
            return {"status": "duplicate_order_blocked", "order_id": intent["id"], "live_execution": True}

        try:
            result = exchange.create_order(
                symbol, "market", side, amount, None,
                {"newClientOrderId": client_order_id},
            )
        except Exception as exc:
            store.update_live_order(
                owner_id, client_order_id, "unknown",
                result={"error_type": type(exc).__name__, "candle_at": candle_at,
                        "timeframe": timeframe, "price": price},
            )
            store.disable_live_execution(owner_id, "order_submission_uncertain")
            return {"status": "stopped", "reason": "order_submission_uncertain",
                    "order_id": intent["id"], "live_execution": True}

        remote_status = result.get("status")
        normalized_status = remote_status if remote_status in {"open", "closed", "canceled", "rejected"} else "unknown"
        filled = float(result.get("filled") or 0.0)
        order = store.update_live_order(
            owner_id, client_order_id, normalized_status,
            exchange_order_id=str(result.get("id")) if result.get("id") is not None else None,
            filled=filled, average=result.get("average"), cost=result.get("cost"),
            result={"candle_at": candle_at, "timeframe": timeframe, "price": price},
        )
        if normalized_status in {"closed", "canceled", "rejected"}:
            next_state = _position_after_fill(state, order)
            next_state.update({
                "daily_date": daily_date, "daily_start_equity": daily_start,
                "peak_equity": peak_equity,
            })
            store.apply_live_execution_state(owner_id, venue, symbol, timeframe, candle_at, next_state)
            store.update_live_execution_account_state(owner_id, daily_date, daily_start, peak_equity)
        elif normalized_status == "unknown":
            store.disable_live_execution(owner_id, "exchange_order_response_uncertain")
        return {
            "status": normalized_status, "processed": True, "symbol": symbol,
            "side": side, "amount": amount, "filled": filled,
            "exchange_order_id": order["exchange_order_id"],
            "client_order_id": client_order_id, "prediction": prediction,
            "confidence": confidence, "live_execution": True,
        }
    finally:
        close = getattr(exchange, "close", None)
        if callable(close):
            close()
