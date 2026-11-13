"""Constrained strategy search on chronological validation data; no live promotion."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import backtest_positions
from .strategies import moving_average_signal, momentum_signal


@dataclass(frozen=True)
class Candidate:
    name: str
    parameters: dict


def candidate_set() -> list[Candidate]:
    return [
        Candidate("ma", {"fast": 5, "slow": 20}),
        Candidate("ma", {"fast": 10, "slow": 30}),
        Candidate("ma", {"fast": 20, "slow": 60}),
        Candidate("momentum", {"lookback": 6}),
        Candidate("momentum", {"lookback": 12}),
        Candidate("momentum", {"lookback": 24}),
    ]


def evaluate_candidate(
    close: pd.Series,
    candidate: Candidate,
    *,
    fee_bps: float = 6.0,
    slippage_bps: float = 2.0,
) -> dict:
    if candidate.name == "ma":
        signal = moving_average_signal(close, **candidate.parameters)
    elif candidate.name == "momentum":
        signal = momentum_signal(close, **candidate.parameters)
    else:
        raise ValueError(f"Unknown candidate strategy: {candidate.name}")
    result = backtest_positions(
        close, signal, fee_bps=fee_bps, slippage_bps=slippage_bps,
    )
    # Penalize severe drawdowns and reward returns; keep the score transparent.
    result["selection_score"] = result["total_return"] + 0.5 * result["max_drawdown"]
    result["candidate"] = candidate.name
    result["parameters"] = candidate.parameters
    return result


def select_candidate(close: pd.Series) -> dict:
    """Select only on supplied validation slice; caller must keep test data separate."""
    if len(close) < 100:
        raise ValueError("At least 100 validation bars are required")
    results = [evaluate_candidate(close, c) for c in candidate_set()]
    return max(results, key=lambda item: item["selection_score"])
