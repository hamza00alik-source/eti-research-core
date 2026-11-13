"""Autonomous rule-based strategy selection with untouched walk-forward tests.

Candidates are selected only on validation windows. If none is sufficiently
active, the evaluator explicitly selects cash. Research only; no live trading.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .backtest import backtest_positions
from .evaluation import walk_forward_folds
from .evolution import build_signal, candidate_set, evaluate_candidate

MIN_VALIDATION_EXPOSURE = 0.10
MIN_VALIDATION_POSITION_CHANGES = 2


def run_strategy_walk_forward(
    close: pd.Series,
    *,
    min_history: int = 100,
    validation_size: int = 100,
    test_size: int = 100,
    step_size: int = 100,
    purge_bars: int = 1,
    fee_bps: float = 6.0,
    slippage_bps: float = 2.0,
) -> dict[str, Any]:
    """Select on validation and evaluate frozen choice on subsequent test windows."""
    if not isinstance(close, pd.Series):
        raise ValueError("close must be a pandas Series")
    if len(close) < 2 or not close.index.is_unique or not close.index.is_monotonic_increasing:
        raise ValueError("close must have at least two unique, chronological observations")
    values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("close prices must be finite and positive")
    for name, value in (("fee_bps", fee_bps), ("slippage_bps", slippage_bps)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    folds = walk_forward_folds(
        len(close), min_train=min_history, validation_size=validation_size,
        test_size=test_size, step_size=step_size, purge_bars=purge_bars,
    )
    candidates = candidate_set()
    reports = []
    for number, fold in enumerate(folds):
        validation = close.iloc[fold.validation_start:fold.validation_end]
        validation_scores = []
        for candidate in candidates:
            result = evaluate_candidate(
                validation, candidate, fee_bps=float(fee_bps),
                slippage_bps=float(slippage_bps),
            )
            active = (
                result["exposure"] >= MIN_VALIDATION_EXPOSURE
                and result["position_changes"] >= MIN_VALIDATION_POSITION_CHANGES
            )
            validation_scores.append({
                "name": candidate.name,
                "parameters": candidate.parameters,
                "selection_score": float(result["selection_score"]),
                "validation_return": float(result["total_return"]),
                "validation_max_drawdown": float(result["max_drawdown"]),
                "validation_exposure": float(result["exposure"]),
                "validation_position_changes": int(result["position_changes"]),
                "eligible": bool(active),
            })

        eligible_indices = [i for i, item in enumerate(validation_scores) if item["eligible"]]
        if eligible_indices:
            selected_index = max(
                eligible_indices,
                key=lambda i: (validation_scores[i]["selection_score"],
                               validation_scores[i]["name"],
                               str(sorted(validation_scores[i]["parameters"].items()))),
            )
            selected = candidates[selected_index]
            selected_name, selected_parameters = selected.name, selected.parameters
        else:
            selected_name, selected_parameters = "cash", {}

        # Start each held-out fold flat and give every candidate the same
        # in-window warm-up; do not carry a stateful position from pre-test
        # history into a test fold.
        test_close = close.iloc[fold.test_start:fold.test_end]
        if selected_name == "cash":
            test_signal = pd.Series(0.0, index=test_close.index)
        else:
            from .evolution import Candidate
            test_signal = build_signal(test_close, Candidate(selected_name, selected_parameters))
        tested = backtest_positions(
            test_close, test_signal, fee_bps=float(fee_bps),
            slippage_bps=float(slippage_bps), allow_short=False,
        )
        benchmark = backtest_positions(
            test_close, pd.Series(1.0, index=test_close.index),
            fee_bps=float(fee_bps), slippage_bps=float(slippage_bps),
            allow_short=False,
        )
        reports.append({
            "fold": number,
            "selected_candidate": {"name": selected_name, "parameters": selected_parameters},
            "selection_status": "cash_fallback_no_active_candidate" if selected_name == "cash" else "active_candidate_selected",
            "validation_candidates": validation_scores,
            "test": {
                "bars": tested["bars"],
                "net_return": tested["total_return"],
                "max_drawdown": tested["max_drawdown"],
                "sharpe_per_bar": tested["sharpe_per_bar"],
                "exposure": tested["exposure"],
                "position_changes": tested["position_changes"],
                "trading_cost_total": tested["trading_cost_total"],
                "buy_and_hold_net_return": benchmark["total_return"],
                "beats_buy_and_hold": tested["total_return"] > benchmark["total_return"],
                "no_trade": tested["exposure"] == 0.0,
            },
            "boundaries": {
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "purge_bars": purge_bars,
            },
        })
    returns = [item["test"]["net_return"] for item in reports]
    drawdowns = [item["test"]["max_drawdown"] for item in reports]
    return {
        "method": "validation_selected_rule_based_walk_forward_v3",
        "candidate_count": len(candidates),
        "candidate_families": ["ma", "momentum", "breakout", "mean_reversion"],
        "fold_count": len(reports),
        "folds": reports,
        "summary": {
            "mean_test_net_return": float(np.mean(returns)),
            "median_test_net_return": float(np.median(returns)),
            "mean_test_max_drawdown": float(np.mean(drawdowns)),
            "folds_beating_buy_and_hold": int(sum(item["test"]["beats_buy_and_hold"] for item in reports)),
            "cash_fallback_folds": int(sum(item["selected_candidate"]["name"] == "cash" for item in reports)),
            "active_candidate_folds": int(sum(item["selected_candidate"]["name"] != "cash" for item in reports)),
        },
        "cost_assumptions_bps": {"fee": float(fee_bps), "slippage": float(slippage_bps)},
        "selection_rule": (
            "Among validation candidates with exposure >= 10% and at least 2 position changes, "
            "select highest validation net return plus 0.5 times validation max drawdown; "
            "otherwise explicitly select cash. Validation and test use identical configured costs."
        ),
        "minimum_activity": {
            "validation_exposure": MIN_VALIDATION_EXPOSURE,
            "validation_position_changes": MIN_VALIDATION_POSITION_CHANGES,
        },
        "safety": {
            "is_trained_model": False,
            "is_trade_signal": False,
            "places_orders": False,
            "warning": "Rule-based research only. Candidate selection uses validation windows; test windows are untouched until selection is frozen. Cash fallback is inactivity, not strategy alpha. Historical results do not establish future profitability.",
        },
    }
