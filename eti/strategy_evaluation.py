"""Autonomous rule-based strategy selection with untouched walk-forward tests.

Candidate choice is made exclusively on each chronological validation window.
The subsequent test window is evaluated once, after selection is frozen.
This is research tooling, not a live-trading signal or profitability claim.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .backtest import backtest_positions
from .evaluation import walk_forward_folds
from .evolution import candidate_set, evaluate_candidate
from .strategies import moving_average_signal, momentum_signal


def _signal(close: pd.Series, name: str, parameters: dict[str, int]) -> pd.Series:
    if name == "ma":
        return moving_average_signal(close, **parameters)
    if name == "momentum":
        return momentum_signal(close, **parameters)
    raise ValueError(f"Unsupported strategy candidate: {name}")


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
    """Select among fixed MA/momentum candidates on validation, then test out of sample.

    The first test position starts flat. Signals are causal and shifted one bar by
    the backtester, so a close-time decision cannot earn the same bar's return.
    """
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
            result = evaluate_candidate(validation, candidate)
            validation_scores.append({
                "name": candidate.name,
                "parameters": candidate.parameters,
                "selection_score": float(result["selection_score"]),
                "validation_return": float(result["total_return"]),
                "validation_max_drawdown": float(result["max_drawdown"]),
            })
        selected_index = max(
            range(len(validation_scores)),
            key=lambda i: (validation_scores[i]["selection_score"],
                           validation_scores[i]["name"],
                           str(sorted(validation_scores[i]["parameters"].items()))),
        )
        selected = candidates[selected_index]

        # Include pre-test history to warm up rolling indicators. Only the test
        # slice is passed to the backtester and reported as out-of-sample.
        through_test = close.iloc[:fold.test_end]
        test_signal = _signal(through_test, selected.name, selected.parameters).iloc[fold.test_start:fold.test_end]
        test_close = close.iloc[fold.test_start:fold.test_end]
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
            "selected_candidate": {"name": selected.name, "parameters": selected.parameters},
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
        "method": "validation_selected_rule_based_walk_forward_v1",
        "candidate_count": len(candidates),
        "fold_count": len(reports),
        "folds": reports,
        "summary": {
            "mean_test_net_return": float(np.mean(returns)),
            "median_test_net_return": float(np.median(returns)),
            "mean_test_max_drawdown": float(np.mean(drawdowns)),
            "folds_beating_buy_and_hold": int(sum(item["test"]["beats_buy_and_hold"] for item in reports)),
        },
        "cost_assumptions_bps": {"fee": float(fee_bps), "slippage": float(slippage_bps)},
        "selection_rule": "Highest validation net return plus 0.5 times validation max drawdown; deterministic tie-break.",
        "safety": {
            "is_trained_model": False,
            "is_trade_signal": False,
            "places_orders": False,
            "warning": "Rule-based research only. Candidate selection uses validation windows; test windows are untouched until selection is frozen. Historical results do not establish future profitability.",
        },
    }
