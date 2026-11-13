"""Allowlisted ETI agent capabilities.

Handlers are explicit server-side functions. User task payloads are data only and
are never interpreted as Python, shell commands, prompts or executable code.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .features import build_features
from .providers import CcxtPublicProvider, YahooFinanceProvider
from .opportunities import rank_spot_opportunity
from .strategy_evaluation import run_strategy_walk_forward


AgentHandler = Callable[[dict[str, Any]], dict[str, Any]]


class AgentCapabilityRegistry:
    """In-process allowlist of named, server-implemented capabilities."""

    def __init__(self):
        self._handlers: dict[str, AgentHandler] = {}

    def register(self, name: str, handler: AgentHandler) -> None:
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ValueError("capability name must contain 1-120 characters")
        if not callable(handler):
            raise TypeError("capability handler must be callable")
        if name in self._handlers:
            raise ValueError(f"capability already registered: {name}")
        self._handlers[name] = handler

    def execute(self, name: str, payload: Any) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            raise LookupError(f"unsupported agent capability: {name}")
        if not isinstance(payload, dict):
            raise ValueError("agent task input must be a JSON object")
        result = handler(payload)
        if not isinstance(result, dict):
            raise TypeError("agent capability must return a JSON object")
        # Reuse strict JSON encoding rules to reject NaN, infinity and oversized results.
        from .storage import _encode_json
        _encode_json(result)
        return result

    def capabilities(self) -> list[str]:
        return sorted(self._handlers)


def _market_provider(payload: dict[str, Any]):
    """Resolve an explicit public-data adapter without implying execution access."""
    asset_class = payload.get("asset_class", "crypto")
    market_type = payload.get("market_type", "spot")
    if not isinstance(asset_class, str) or asset_class not in {"crypto", "forex", "equity", "commodity", "index"}:
        raise ValueError("asset_class must be crypto, forex, equity, commodity, or index")
    if not isinstance(market_type, str):
        raise ValueError("market_type must be a string")
    if asset_class == "crypto":
        venue = payload.get("venue", "mexc")
        if not isinstance(venue, str) or venue not in {"binance", "mexc"}:
            raise ValueError("crypto venue must be binance or mexc")
        if market_type not in {"spot", "swap", "future"}:
            raise ValueError("crypto market_type must be spot, swap, or future")
        return asset_class, market_type, venue, CcxtPublicProvider(venue, market_type)
    if market_type not in {"spot", "future"}:
        raise ValueError("Yahoo Finance market_type must be spot or future")
    return asset_class, market_type, "yahoo", YahooFinanceProvider(asset_class)


def market_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Fetch completed public crypto spot or derivatives candles and summarize indicators."""
    asset_class, market_type, venue, provider = _market_provider(payload)
    symbol = payload.get("symbol")
    timeframe = payload.get("timeframe", "1h")
    limit = payload.get("limit", 200)
    if not isinstance(symbol, str) or not 3 <= len(symbol) <= 40 or "/" not in symbol:
        raise ValueError("symbol must be a valid CCXT market symbol such as BTC/USDT or BTC/USDT:USDT")
    if not isinstance(timeframe, str) or timeframe not in {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}:
        raise ValueError("unsupported timeframe")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 50 <= limit <= 500:
        raise ValueError("limit must be an integer from 50 to 500")

    frame = provider.fetch_ohlcv(symbol, timeframe, limit)
    if len(frame) < 40:
        raise ValueError("provider returned too few completed candles for indicator calculation")
    features = build_features(frame)
    latest = features.iloc[-1]
    indicator_names = (
        "ret_1", "ret_3", "ret_6", "volatility_20",
        "sma_gap_10", "sma_gap_30", "rsi_14", "range_pct", "atr_pct",
    )
    indicators: dict[str, float] = {}
    for name in indicator_names:
        value = float(latest[name])
        if not np.isfinite(value):
            raise ValueError(f"latest indicator is unavailable: {name}")
        indicators[name] = round(value, 8)
    short_gap, long_gap = indicators["sma_gap_10"], indicators["sma_gap_30"]
    trend = "bullish" if short_gap > 0 and long_gap > 0 else (
        "bearish" if short_gap < 0 and long_gap < 0 else "mixed"
    )
    return {
        "kind": "public_market_snapshot",
        "venue": venue,
        "asset_class": asset_class,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "completed_candles": len(frame),
        "latest_completed_candle": frame.iloc[-1]["timestamp"].isoformat(),
        "last_close": round(float(frame.iloc[-1]["close"]), 10),
        "trend_context": trend,
        "indicators": indicators,
        "research_ranking": rank_spot_opportunity(indicators) if asset_class == "crypto" and market_type == "spot" else None,
        "limitations": [
            "Descriptive market research and a transparent heuristic ranking only; not a forecast or trade signal.",
            "Public market OHLCV only. Source coverage, latency and history limits vary by instrument and provider.",
            "The ranking is not a trained AI model; model training is performed separately.",
        ],
    }


def strategy_walk_forward(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate fixed rule candidates on public candles; report is stored as task result."""
    asset_class, market_type, venue, provider = _market_provider(payload)
    symbol = payload.get("symbol")
    timeframe = payload.get("timeframe", "1h")
    limit = payload.get("limit", 500)
    if not isinstance(symbol, str) or not 3 <= len(symbol) <= 40 or "/" not in symbol:
        raise ValueError("symbol must be a valid CCXT market symbol such as BTC/USDT or BTC/USDT:USDT")
    if not isinstance(timeframe, str) or timeframe not in {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}:
        raise ValueError("unsupported timeframe")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 300 <= limit <= 500:
        raise ValueError("strategy walk-forward limit must be from 300 to 500")
    options = {
        "min_history": payload.get("min_history", 120),
        "validation_size": payload.get("validation_size", 80),
        "test_size": payload.get("test_size", 80),
        "step_size": payload.get("step_size", 80),
        "purge_bars": payload.get("purge_bars", 2),
    }
    for name, value in options.items():
        minimum = 0 if name == "purge_bars" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} has an invalid value")
    fee_bps, slippage_bps = payload.get("fee_bps", 6.0), payload.get("slippage_bps", 2.0)
    frame = provider.fetch_ohlcv(symbol, timeframe, limit)
    if len(frame) < options["min_history"] + options["validation_size"] + options["test_size"] + 2 * options["purge_bars"]:
        raise ValueError("provider returned insufficient completed candles for walk-forward evaluation")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any() or timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("provider timestamps must be valid, unique, and chronological")
    close = pd.Series(pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float),
                      index=pd.DatetimeIndex(timestamps), name="close")
    report = run_strategy_walk_forward(close, **options, fee_bps=fee_bps, slippage_bps=slippage_bps)
    return {
        "kind": "strategy_walk_forward_report", "venue": venue, "asset_class": asset_class,
        "market_type": market_type, "symbol": symbol, "timeframe": timeframe,
        "requested_candles": limit, "available_candles": len(frame),
        "data_start": timestamps.iloc[0].isoformat(), "data_end": timestamps.iloc[-1].isoformat(),
        **report,
        "limitations": [
            "Fixed rule-based moving-average and momentum candidates; not a trained AI model.",
            "Validation data selects candidates; test windows are held out from candidate selection.",
            "Historical results do not establish future profitability and are not trade signals.",
            "Price-only rule evaluation; provider-specific financing, corporate actions, futures rolls and contract costs are not modeled.",
        ],
    }


def model_training_research(payload: dict[str, Any]) -> dict[str, Any]:
    """Train a crypto spot/derivatives research candidate; approval remains gated by data quality."""
    asset_class, market_type, venue, provider = _market_provider(payload)
    symbol = payload.get("symbol")
    timeframe = payload.get("timeframe", "1h")
    limit = payload.get("limit", 3000)
    horizon = payload.get("horizon", 6)
    if not isinstance(symbol, str) or not 3 <= len(symbol) <= 40 or "/" not in symbol:
        raise ValueError("symbol must be a valid CCXT market symbol such as BTC/USDT or BTC/USDT:USDT")
    if not isinstance(timeframe, str) or timeframe not in {"1h", "4h", "1d"}:
        raise ValueError("timeframe must be one of: 1h, 4h, or 1d")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1500 <= limit <= 5000:
        raise ValueError("training limit must be an integer from 1500 to 5000")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 2 <= horizon <= 24:
        raise ValueError("horizon must be an integer from 2 to 24")

    frame = provider.fetch_ohlcv(symbol, timeframe, limit)
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any() or timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("provider timestamps must be valid, unique, and chronological")
    if len(frame) < 1500:
        raise ValueError("provider returned fewer than 1500 completed candles")

    from .pipeline import fit_deployable_model, run_nested_experiment
    evaluation = run_nested_experiment(
        frame, horizon=horizon, min_train=500, validation_size=100,
        test_size=100, step_size=250, timeframe=timeframe,
        allow_short=(market_type in {"swap", "future"} or asset_class in {"forex", "index"}),
    )
    model, schema = fit_deployable_model(frame, horizon=horizon)
    from .artifacts import ModelArtifactStore
    artifact = ModelArtifactStore().save_model(model)

    trading = evaluation.get("trading_performance", {})
    trading_summary = trading.get("summary", {})
    fold_count = int(evaluation.get("fold_count", 0))
    selected_model_folds = sum(
        fold.get("selected_candidate") == "hist_gradient_boosting"
        for fold in evaluation.get("folds", [])
    )
    mean_score = evaluation.get("mean_test_balanced_accuracy")
    score_ci = evaluation.get("test_balanced_accuracy_ci") or {}
    calibration_ok = (
        model.calibration_status == "calibrated"
        and isinstance(model.calibration_report, dict)
        and model.calibration_report.get("passed") is True
    )
    quality_checks = {
        "at_least_three_test_folds": fold_count >= 3,
        "derivative_funding_history_available": not (asset_class == "crypto" and market_type in {"swap", "future"}),
        "balanced_accuracy_at_least_0_52": isinstance(mean_score, (int, float)) and mean_score >= 0.52,
        "confidence_interval_lower_above_chance": isinstance(score_ci.get("lower"), (int, float)) and score_ci["lower"] > 0.50,
        "model_selected_in_majority_of_folds": selected_model_folds > fold_count / 2 if fold_count else False,
        "positive_mean_net_return": isinstance(trading_summary.get("mean_net_return"), (int, float)) and trading_summary["mean_net_return"] > 0,
        "beats_no_trade_in_two_thirds_of_folds": int(trading_summary.get("folds_beating_no_trade", 0)) >= max(2, int(__import__("math").ceil(fold_count * 2 / 3))),
        "calibration_passed": calibration_ok,
    }
    evaluation_passed = bool(quality_checks) and all(quality_checks.values())
    training_data_ref = (
        f"{venue}:{asset_class}:{market_type}:{symbol}:{timeframe}:{len(frame)}:"
        f"{schema.get('dataset_sha256', 'unknown')}"
    )
    trading_folds = {item.get("fold"): item for item in trading.get("folds", [])}
    fold_diagnostics = []
    for fold in evaluation.get("folds", []):
        trading_fold = trading_folds.get(fold.get("fold"), {})
        fold_diagnostics.append({
            "fold": fold.get("fold"),
            "selected_candidate": fold.get("selected_candidate"),
            "validation_scores": fold.get("validation_scores"),
            "test_accuracy": fold.get("test_accuracy"),
            "test_balanced_accuracy": fold.get("test_balanced_accuracy"),
            "boundaries": fold.get("boundaries"),
            "market_regimes": trading_fold.get("market_regimes"),
            "trading_metrics": trading_fold.get("trading", {}),
        })
    return {
        "kind": "trained_model_candidate",
        "venue": venue,
        "asset_class": "crypto",
        "asset_class": asset_class,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "requested_candles": limit,
        "available_candles": len(frame),
        "data_start": timestamps.iloc[0].isoformat(),
        "data_end": timestamps.iloc[-1].isoformat(),
        "model_family": "HistGradientBoostingClassifier via ETI BaselineModel",
        "training_method": "from_scratch_purged_nested_walk_forward_v1",
        "training_performed": True,
        "evaluation": {
            "fold_count": evaluation.get("fold_count", 0),
            "passed": evaluation_passed,
            "quality_checks": quality_checks,
            "mean_test_balanced_accuracy": evaluation.get("mean_test_balanced_accuracy"),
            "test_balanced_accuracy_ci": evaluation.get("test_balanced_accuracy_ci"),
            "trading_summary": trading.get("summary", {}),
            "folds": fold_diagnostics,
            "model_note": evaluation.get("model_note"),
            "warning": evaluation.get("warning"),
        },
        "candidate_artifact": artifact,
        "input_schema": schema,
        "training_data_ref": training_data_ref,
        "calibration_status": model.calibration_status,
        "calibration_report": model.calibration_report,
        "lifecycle": "unregistered_candidate",
        "approval_status": "not_approved",
        "evaluation_passed": evaluation_passed,
        "quality_checks": quality_checks,
        "limitations": [
            "This is a research candidate artifact; it is not registered, approved, or enabled for inference.",
            "Nested test folds are held out from candidate selection; final artifact calibration uses a separate chronological holdout.",
            "Calibration failure blocks approval; no automatic promotion occurs.",
            "Single-market evaluation is not a portfolio backtest and does not establish future profitability.",
            "Derivative candidates use a long/short price-only simulation; financing, funding, roll, corporate-action and contract-specific costs are not fully modeled. Crypto derivative funding-history gate blocks approval.",
        ],
    }




def default_capabilities() -> AgentCapabilityRegistry:
    registry = AgentCapabilityRegistry()
    registry.register("market.snapshot", market_snapshot)
    registry.register("strategy.walk_forward", strategy_walk_forward)
    registry.register("model.train_research", model_training_research)
    return registry
