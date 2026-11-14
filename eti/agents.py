"""Allowlisted ETI agent capabilities.

Handlers are explicit server-side functions. User task payloads are data only and
are never interpreted as Python, shell commands, prompts or executable code.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .data import assess_ohlcv_quality

from .features import build_features
from .providers import CcxtPublicProvider, YahooFinanceProvider, enrich_ohlcv_with_derivatives
from .opportunities import rank_spot_opportunity
from .strategy_evaluation import run_strategy_walk_forward
from .solana_research import DexScreenerClient, SolanaRpcClient, RugCheckClient, scan_pair, audit_mint


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


def _ohlcv_fetch_diagnostics(provider: Any, frame: pd.DataFrame, timeframe: str, requested: int) -> dict[str, Any]:
    """Describe provider response size without confusing request fulfillment with market coverage."""
    count = len(frame)
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce") if count else pd.Series(dtype="datetime64[ns, UTC]")
    valid = timestamps.dropna()
    return {
        "provider": getattr(provider, "provider_name", type(provider).__name__),
        "provider_class": type(provider).__name__,
        "venue": getattr(provider, "venue", "yahoo" if isinstance(provider, YahooFinanceProvider) else None),
        "timeframe": timeframe,
        "requested_candles": requested,
        "returned_candles": count,
        "request_shortfall_candles": max(0, requested - count),
        "request_fulfillment_ratio": round(min(count / requested, 1.0), 6) if requested else None,
        "first_returned_candle": valid.iloc[0].isoformat() if len(valid) else None,
        "latest_returned_candle": valid.iloc[-1].isoformat() if len(valid) else None,
        "note": "Request fulfillment compares returned rows with the requested limit; it is not a measure of exchange-session coverage or missing-candle completeness.",
    }


def market_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Fetch completed public multi-asset OHLCV candles and summarize causal indicators."""
    asset_class, market_type, venue, provider = _market_provider(payload)
    symbol = payload.get("symbol")
    timeframe = payload.get("timeframe", "1h")
    limit = payload.get("limit", 200)
    if not isinstance(symbol, str) or not 1 <= len(symbol) <= 40 or (asset_class == "crypto" and "/" not in symbol):
        raise ValueError("symbol must be a valid CCXT symbol for crypto or a Yahoo Finance ticker for other asset classes")
    if not isinstance(timeframe, str) or timeframe not in {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}:
        raise ValueError("unsupported timeframe")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 50 <= limit <= 500:
        raise ValueError("limit must be an integer from 50 to 500")

    frame = provider.fetch_ohlcv(symbol, timeframe, limit)
    provider_diagnostics = _ohlcv_fetch_diagnostics(provider, frame, timeframe, limit)
    data_quality = assess_ohlcv_quality(frame, timeframe)
    derivative_context = None
    if asset_class == "crypto" and market_type in {"swap", "future"}:
        derivative_context = provider.fetch_derivatives_context(symbol, timeframe, min(100, limit))
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
    limitations = [
        "Descriptive market research and a transparent heuristic ranking only; not a forecast or trade signal.",
        "Public market OHLCV only. Source coverage, latency and history limits vary by instrument and provider.",
        "The ranking is not a trained AI model; model training is performed separately.",
    ]
    if asset_class == "crypto" and market_type in {"swap", "future"}:
        limitations.append("Current funding and open-interest availability is summarized below; historical funding/open-interest features enter derivative model training only when observations are supplied. Liquidations and validated contract-specific execution costs remain incomplete.")
    return {
        "kind": "public_market_snapshot",
        "venue": venue,
        "asset_class": asset_class,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "completed_candles": len(frame),
        "provider_diagnostics": provider_diagnostics,
        "data_quality": data_quality,
        "latest_completed_candle": frame.iloc[-1]["timestamp"].isoformat(),
        "last_close": round(float(frame.iloc[-1]["close"]), 10),
        "trend_context": trend,
        "indicators": indicators,
        "research_ranking": rank_spot_opportunity(indicators) if asset_class == "crypto" and market_type == "spot" else None,
        "derivatives_context": ({
            "current_funding": derivative_context.get("current_funding"),
            "current_open_interest": derivative_context.get("current_open_interest"),
            "history_coverage": derivative_context.get("history_coverage"),
            "data_quality": derivative_context.get("data_quality"),
            "contract_metadata": derivative_context.get("contract_metadata"),
            "liquidation_observations": len(derivative_context.get("liquidation_history") or []),
            "availability": derivative_context.get("availability"),
        } if derivative_context else None),
        "limitations": limitations,
    }


def strategy_walk_forward(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate fixed rule candidates on public candles; report is stored as task result."""
    asset_class, market_type, venue, provider = _market_provider(payload)
    symbol = payload.get("symbol")
    timeframe = payload.get("timeframe", "1h")
    limit = payload.get("limit", 500)
    if not isinstance(symbol, str) or not 1 <= len(symbol) <= 40 or (asset_class == "crypto" and "/" not in symbol):
        raise ValueError("symbol must be a valid CCXT symbol for crypto or a Yahoo Finance ticker for other asset classes")
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
    provider_diagnostics = _ohlcv_fetch_diagnostics(provider, frame, timeframe, limit)
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
        "provider_diagnostics": provider_diagnostics,
        "data_start": timestamps.iloc[0].isoformat(), "data_end": timestamps.iloc[-1].isoformat(),
        **report,
        "limitations": [
            "Fixed rule-based moving-average and momentum candidates; not a trained AI model.",
            "This evaluator uses a long/flat rule set; it is not a validated short-selling or derivatives execution strategy.",
            "Validation data selects candidates; test windows are held out from candidate selection.",
            "Historical results do not establish future profitability and are not trade signals.",
            "Price-only rule evaluation; provider-specific financing, corporate actions, futures rolls and contract costs are not modeled.",
        ],
    }


def _training_data_quality_review(data_quality: dict[str, Any], asset_class: str) -> dict[str, Any]:
    """Apply strict freshness only to 24/7 crypto; session-calendar gaps stay informational."""
    is_continuous_market = asset_class == "crypto"
    return {
        "freshness_gate_applied": is_continuous_market,
        "freshness_gate_passed": data_quality["status"] == "fresh" if is_continuous_market else None,
        "calendar_gaps_blocking": False,
        "calendar_coverage_ratio": data_quality["calendar_coverage_ratio"],
        "gap_count": data_quality["gap_count"],
        "review_note": (
            "Crypto trades continuously; stale latest candles block candidate evaluation."
            if is_continuous_market
            else "Freshness and elapsed-calendar gaps are informational because provider session calendars, weekends and holidays are not modeled."
        ),
    }


def model_training_research(payload: dict[str, Any]) -> dict[str, Any]:
    """Train a multi-asset research candidate; approval remains gated by data quality."""
    asset_class, market_type, venue, provider = _market_provider(payload)
    symbol = payload.get("symbol")
    timeframe = payload.get("timeframe", "1h")
    limit = payload.get("limit", 3000)
    horizon = payload.get("horizon", 6)
    if not isinstance(symbol, str) or not 1 <= len(symbol) <= 40 or (asset_class == "crypto" and "/" not in symbol):
        raise ValueError("symbol must be a valid CCXT symbol for crypto or a Yahoo Finance ticker for other asset classes")
    if not isinstance(timeframe, str) or timeframe not in {"1h", "4h", "1d"}:
        raise ValueError("timeframe must be one of: 1h, 4h, or 1d")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1500 <= limit <= 5000:
        raise ValueError("training limit must be an integer from 1500 to 5000")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 2 <= horizon <= 24:
        raise ValueError("horizon must be an integer from 2 to 24")

    frame = provider.fetch_ohlcv(symbol, timeframe, limit)
    provider_diagnostics = _ohlcv_fetch_diagnostics(provider, frame, timeframe, limit)
    # Assess the raw OHLCV frame before derivative enrichment adds optional columns.
    data_quality = assess_ohlcv_quality(frame, timeframe)
    data_quality_review = _training_data_quality_review(data_quality, asset_class)
    derivative_context = None
    funding_bps_per_bar = 0.0
    derivative_feature_coverage = None
    if asset_class == "crypto" and market_type in {"swap", "future"}:
        derivative_context = provider.fetch_derivatives_context(symbol, timeframe, min(1000, limit))
        frame = enrich_ohlcv_with_derivatives(frame, derivative_context, timeframe)
        funding_rows = derivative_context.get("funding_history") or []
        if len(funding_rows) >= 2:
            rates = pd.to_numeric(pd.Series([item["funding_rate"] for item in funding_rows]), errors="coerce").dropna()
            times = pd.to_numeric(pd.Series([item["timestamp"] for item in funding_rows]), errors="coerce").dropna().sort_values()
            if len(rates) and len(times) >= 2:
                interval_ms = float(times.diff().dropna().median())
                bar_ms = float(pd.Timedelta(timeframe).total_seconds() * 1000)
                if interval_ms > 0:
                    # Absolute average funding is used as a conservative, direction-neutral research drag.
                    funding_bps_per_bar = float(rates.abs().mean()) * 10000.0 * bar_ms / interval_ms
        derivative_feature_coverage = {
            "funding_observations": len(funding_rows),
            "open_interest_observations": len(derivative_context.get("open_interest_history") or []),
            "funding_feature_rows": int(frame.get("funding_rate", pd.Series(dtype=float)).notna().sum()),
            "open_interest_feature_rows": int(frame.get("open_interest", pd.Series(dtype=float)).notna().sum()),
            "funding_bps_per_bar_conservative": round(funding_bps_per_bar, 8),
        }
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
        funding_bps_per_bar=funding_bps_per_bar,
        fee_bps=float(payload.get("fee_bps", 8.0 if market_type in {"swap", "future"} else 6.0)),
        slippage_bps=float(payload.get("slippage_bps", 5.0 if market_type in {"swap", "future"} else 2.0)),
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
        "derivative_funding_history_available": not (asset_class == "crypto" and market_type in {"swap", "future"}) or bool(derivative_feature_coverage and derivative_feature_coverage["funding_observations"] >= 3 and derivative_feature_coverage["funding_feature_rows"] >= 500),
        "derivative_open_interest_history_available": not (asset_class == "crypto" and market_type in {"swap", "future"}) or bool(derivative_feature_coverage and derivative_feature_coverage["open_interest_observations"] >= 50 and derivative_feature_coverage["open_interest_feature_rows"] >= 500),
        "instrument_cost_model_supported": asset_class == "crypto" and market_type == "spot",
        "derivative_cost_model_independently_validated": asset_class == "crypto" and market_type == "spot",
        "balanced_accuracy_at_least_0_52": isinstance(mean_score, (int, float)) and mean_score >= 0.52,
        "confidence_interval_lower_above_chance": isinstance(score_ci.get("lower"), (int, float)) and score_ci["lower"] > 0.50,
        "model_selected_in_majority_of_folds": selected_model_folds > fold_count / 2 if fold_count else False,
        "positive_mean_net_return": isinstance(trading_summary.get("mean_net_return"), (int, float)) and trading_summary["mean_net_return"] > 0,
        "beats_no_trade_in_two_thirds_of_folds": int(trading_summary.get("folds_beating_no_trade", 0)) >= max(2, int(__import__("math").ceil(fold_count * 2 / 3))),
        "calibration_passed": calibration_ok,
    }
    if asset_class == "crypto":
        quality_checks["crypto_market_data_freshness_acceptable"] = (
            data_quality["status"] == "fresh"
        )
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
        "asset_class": asset_class,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "requested_candles": limit,
        "available_candles": len(frame),
        "data_start": timestamps.iloc[0].isoformat(),
        "data_end": timestamps.iloc[-1].isoformat(),
        "model_family": "HistGradientBoostingClassifier via ETI BaselineModel",
        "training_method": "from_scratch_purged_nested_walk_forward_v2_multimodal_market_features",
        "training_performed": True,
        "provider_diagnostics": provider_diagnostics,
        "data_quality": data_quality,
        "data_quality_review": data_quality_review,
        "derivative_feature_coverage": derivative_feature_coverage,
        "derivatives_data_quality": derivative_context.get("data_quality") if derivative_context else None,
        "derivatives_contract_metadata": derivative_context.get("contract_metadata") if derivative_context else None,
        "derivatives_context_availability": derivative_context.get("availability") if derivative_context else None,
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
            "Funding and open-interest features are used only when the exchange returns historical observations; missing observations are never synthesized.",
            "Derivative funding drag is conservatively approximated from absolute observed funding rates and median settlement intervals; it is not a substitute for side-aware exchange settlement and contract fee modelling.",
            "Yahoo Finance and non-spot instruments remain research-only until provider-specific financing, funding, roll, corporate-action and contract costs are independently validated. Unsupported cost models block approval.",
        ],
    }





def discover_market_universe(payload: dict[str, Any]) -> dict[str, Any]:
    """Discover listed crypto markets or return curated Yahoo public reference instruments."""
    asset_class = payload.get("asset_class", "crypto")
    limit = payload.get("limit", 100)
    market_type = payload.get("market_type", "spot")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise ValueError("limit must be an integer from 1 to 500")
    if asset_class == "crypto":
        venue = payload.get("venue", "binance")
        if venue not in {"binance", "mexc"}:
            raise ValueError("venue must be binance or mexc")
        if market_type not in {"spot", "swap", "future"}:
            raise ValueError("market_type must be spot, swap, or future")
        instruments = CcxtPublicProvider(venue, market_type).discover_markets(limit=limit)
        source = "CCXT public exchange market metadata"
        venue_name = venue
    elif asset_class in {"forex", "equity", "commodity", "index"}:
        if market_type not in {"spot", "future"}:
            raise ValueError("Yahoo Finance market type must be spot or future")
        provider = YahooFinanceProvider(asset_class)
        instruments = provider.discover_markets(limit=limit, market_type=market_type)
        source = "curated Yahoo Finance public reference universe"
        venue_name = "yahoo"
    else:
        raise ValueError("asset_class must be crypto, forex, equity, commodity, or index")
    return {
        "kind": "market_universe_discovery",
        "venue": venue_name, "asset_class": asset_class, "market_type": market_type,
        "instrument_count": len(instruments), "instruments": instruments,
        "discovery_source": source,
        "limitations": [
            "Crypto discovery covers active instruments listed by Binance or MEXC and ranks by public quote volume when supported.",
            "Yahoo Finance asset catalogues are curated reference lists, not exhaustive exchange listings or live screeners.",
            "Listed instruments are not automatically eligible for training, approval or live execution.",
        ],
    }


def market_coverage_report(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Describe implemented data and execution coverage without implying availability."""
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("coverage input must be a JSON object")
    return {
        "kind": "market_coverage_report",
        "coverage_as_of": "implementation_capabilities",
        "research": {
            "crypto": {"venues": ["binance", "mexc"], "market_types": ["spot", "swap", "future"],
                       "discovery": "active CCXT-listed instruments; dated futures require exact listed contract symbols",
                       "data": "public OHLCV; funding/open-interest context where endpoints support it"},
            "forex": {"venue": "yahoo", "market_types": ["spot"], "universe": "curated reference symbols"},
            "equity": {"venue": "yahoo", "market_types": ["spot"], "universe": "curated reference symbols"},
            "commodity": {"venue": "yahoo", "market_types": ["future"], "universe": "curated futures reference symbols"},
            "index": {"venue": "yahoo", "market_types": ["spot", "future"], "universe": "curated reference symbols"},
            "csv": {"asset_classes": ["crypto", "forex", "equity", "commodity", "index"],
                    "market_types": ["caller-labelled local OHLCV"]},
        },
        "live_execution": {
            "enabled_by_default": False, "venues": ["binance", "mexc"],
            "market_types": ["spot"], "derivatives_enabled": False,
            "requires": ["owner-managed credentials", "approved calibrated model",
                         "saved risk limits", "explicit owner authorization"],
        },
        "not_implemented": ["options", "CFDs", "margin execution", "live derivative execution",
                            "exhaustive global exchange catalogues"],
        "important_distinctions": [
            "Discovery does not prove historical candles are available or sufficiently complete.",
            "Research support does not imply broker connectivity or permission to execute.",
            "Yahoo Finance symbols are curated public references, not exhaustive or guaranteed real-time feeds.",
            "Funding, open-interest, fees, margin, settlement and roll coverage vary by contract and venue.",
        ],
    }

def derivatives_market_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Collect exchange-reported funding and open-interest data for one derivative."""
    venue = payload.get("venue", "binance")
    market_type = payload.get("market_type", "swap")
    symbol = payload.get("symbol")
    timeframe = payload.get("timeframe", "1h")
    limit = payload.get("limit", 100)
    if venue not in {"binance", "mexc"}:
        raise ValueError("venue must be binance or mexc")
    if market_type not in {"swap", "future"}:
        raise ValueError("market_type must be swap or future")
    if not isinstance(symbol, str) or not 1 <= len(symbol) <= 40 or "/" not in symbol:
        raise ValueError("symbol must be a valid CCXT derivative symbol")
    if timeframe not in {"1h", "4h", "1d"}:
        raise ValueError("timeframe must be 1h, 4h, or 1d")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer from 1 to 1000")
    context = CcxtPublicProvider(venue, market_type).fetch_derivatives_context(
        symbol, timeframe=timeframe, limit=limit
    )
    context["kind"] = "derivatives_market_context"
    context["timeframe"] = timeframe
    return context

def solana_market_scan(payload: dict[str, Any]) -> dict[str, Any]:
    """Fetch one Solana mint's current DexScreener pair and apply explicit market filters."""
    mint = payload.get("mint")
    if not isinstance(mint, str) or not 32 <= len(mint) <= 48:
        raise ValueError("mint must be a valid-length Solana address")
    liquidity_floor = payload.get("min_liquidity_usd", 40000.0)
    age_floor = payload.get("min_pool_age_minutes", 10.0)
    chase_ceiling = payload.get("max_h1_change_pct", 80.0)
    for name, value in (("min_liquidity_usd", liquidity_floor),
                        ("min_pool_age_minutes", age_floor),
                        ("max_h1_change_pct", chase_ceiling)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
    pairs = DexScreenerClient().fetch_pairs([mint])
    pair = DexScreenerClient.select_pair(mint, pairs)
    return scan_pair(mint, pair, min_liquidity_usd=float(liquidity_floor),
                     min_pool_age_minutes=float(age_floor),
                     max_h1_change_pct=float(chase_ceiling))


def solana_token_audit(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the read-only Solana evidence audit; incomplete critical evidence blocks."""
    mint = payload.get("mint")
    if not isinstance(mint, str) or not 32 <= len(mint) <= 48:
        raise ValueError("mint must be a valid-length Solana address")
    pairs = DexScreenerClient().fetch_pairs([mint])
    pair = DexScreenerClient.select_pair(mint, pairs)
    created_at = pair.get("pairCreatedAt") if pair else None
    created_at = int(created_at) if isinstance(created_at, (int, float)) else None
    excluded = payload.get("excluded_token_accounts", [])
    if not isinstance(excluded, list) or len(excluded) > 100:
        raise ValueError("excluded_token_accounts must be a list of at most 100 addresses")
    return audit_mint(
        mint, rpc=SolanaRpcClient(), rugcheck=RugCheckClient(),
        pool_created_at_ms=created_at,
        pool_address=pair.get("pairAddress") if pair else None,
        creator_address=payload.get("creator_hint"),
        excluded_token_accounts=excluded,
    )


def default_capabilities() -> AgentCapabilityRegistry:
    registry = AgentCapabilityRegistry()
    registry.register("market.snapshot", market_snapshot)
    registry.register("market.discover", discover_market_universe)
    registry.register("market.coverage", market_coverage_report)
    registry.register("derivatives.context", derivatives_market_context)
    registry.register("strategy.walk_forward", strategy_walk_forward)
    registry.register("model.train_research", model_training_research)
    registry.register("solana.market.scan", solana_market_scan)
    registry.register("solana.token.audit", solana_token_audit)
    return registry
