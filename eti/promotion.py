"""Evidence gate for human-reviewed champion/challenger research promotion.

This module never promotes or approves an artifact. It evaluates a preregistered
paired comparison and returns eligibility for a separate human review.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Any


_HASH = re.compile(r"^[0-9a-f]{64}$")


def _hash(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _timestamp(name: str, value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _folds(report: dict[str, Any], label: str) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        raise ValueError(f"{label} report must be an object")
    if report.get("cost_adjusted") is not True:
        raise ValueError(f"{label} report must contain cost-adjusted returns")
    if report.get("evaluation_locked") is not True:
        raise ValueError(f"{label} evaluation data must be declared locked")
    _hash(f"{label}.model_sha256", report.get("model_sha256"))
    _hash(f"{label}.training_data_sha256", report.get("training_data_sha256"))
    _hash(f"{label}.evaluation_data_sha256", report.get("evaluation_data_sha256"))
    _hash(f"{label}.cost_model_sha256", report.get("cost_model_sha256"))
    folds = report.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"{label} report must contain fold-level evidence")
    normalized = []
    for index, fold in enumerate(folds):
        if not isinstance(fold, dict):
            raise ValueError(f"{label}.folds[{index}] must be an object")
        start = _timestamp(f"{label}.folds[{index}].test_start", fold.get("test_start"))
        end = _timestamp(f"{label}.folds[{index}].test_end", fold.get("test_end"))
        if end <= start:
            raise ValueError(f"{label}.folds[{index}] has an empty test interval")
        bars = fold.get("test_bars")
        if isinstance(bars, bool) or not isinstance(bars, int) or bars <= 0:
            raise ValueError(f"{label}.folds[{index}].test_bars must be a positive integer")
        net_return_bps = _finite(f"{label}.folds[{index}].net_return_bps", fold.get("net_return_bps"))
        drawdown = _finite(f"{label}.folds[{index}].max_drawdown", fold.get("max_drawdown"))
        ece = _finite(f"{label}.folds[{index}].out_of_sample_ece", fold.get("out_of_sample_ece"))
        if not 0 <= drawdown <= 1 or not 0 <= ece <= 1:
            raise ValueError(f"{label}.folds[{index}] drawdown/ECE must be in [0, 1]")
        normalized.append({
            "start": start, "end": end, "test_bars": bars,
            "net_return_bps": net_return_bps,
            "max_drawdown": drawdown, "ece": ece,
        })
    normalized.sort(key=lambda item: item["start"])
    for previous, current in zip(normalized, normalized[1:]):
        if current["start"] < previous["end"]:
            raise ValueError(f"{label} test folds overlap; independent fold comparison is not allowed")
    return normalized


def assess_challenger_promotion(
    champion: dict[str, Any],
    challenger: dict[str, Any],
    *,
    comparison_family_size: int,
    min_folds: int = 5,
    min_test_bars: int = 1000,
    alpha: float = 0.05,
    min_mean_improvement_bps: float = 0.0,
    max_mean_drawdown: float = 0.10,
    max_mean_oos_ece: float = 0.08,
    max_drawdown_worsening: float = 0.02,
    max_ece_worsening: float = 0.02,
) -> dict[str, Any]:
    """Assess a preregistered paired challenger; never changes artifact state.

    Uses a one-sided exact sign test over non-overlapping paired test folds.
    Bonferroni adjusts for the declared comparison family size. This is a
    conservative gate, not proof of independence or future profitability.
    """
    if isinstance(comparison_family_size, bool) or not isinstance(comparison_family_size, int) or comparison_family_size < 1:
        raise ValueError("comparison_family_size must be a positive integer")
    if isinstance(min_folds, bool) or not isinstance(min_folds, int) or min_folds < 5:
        raise ValueError("min_folds must be an integer >= 5")
    if isinstance(min_test_bars, bool) or not isinstance(min_test_bars, int) or min_test_bars < 1:
        raise ValueError("min_test_bars must be a positive integer")
    alpha = _finite("alpha", alpha)
    min_improvement = _finite("min_mean_improvement_bps", min_mean_improvement_bps)
    max_drawdown = _finite("max_mean_drawdown", max_mean_drawdown)
    max_ece = _finite("max_mean_oos_ece", max_mean_oos_ece)
    max_dd_worsening = _finite("max_drawdown_worsening", max_drawdown_worsening)
    max_ece_worsening_value = _finite("max_ece_worsening", max_ece_worsening)
    if (
        not 0 < alpha < 1 or min_improvement < 0
        or not 0 <= max_drawdown <= 1 or not 0 <= max_ece <= 1
        or not 0 <= max_dd_worsening <= 1
        or not 0 <= max_ece_worsening_value <= 1
    ):
        raise ValueError("promotion thresholds are outside permitted ranges")

    champ = _folds(champion, "champion")
    chall = _folds(challenger, "challenger")
    for field in ("evaluation_data_sha256", "cost_model_sha256"):
        if champion[field] != challenger[field]:
            raise ValueError(f"champion and challenger must share {field}")
    if len(champ) != len(chall):
        raise ValueError("champion and challenger must have identical paired test folds")
    for index, (left, right) in enumerate(zip(champ, chall)):
        if (left["start"], left["end"], left["test_bars"]) != (right["start"], right["end"], right["test_bars"]):
            raise ValueError(f"champion/challenger test fold {index} does not match")

    total_bars = sum(item["test_bars"] for item in champ)
    if len(champ) < min_folds or total_bars < min_test_bars:
        return {
            "promotion_eligible": False, "status": "insufficient_independent_evidence",
            "fold_count": len(champ), "test_bars": total_bars,
            "required_folds": min_folds, "required_test_bars": min_test_bars,
            "automatic_promotion": False, "requires_human_review": True,
        }

    differences = [challenger_fold["net_return_bps"] - champion_fold["net_return_bps"]
                   for champion_fold, challenger_fold in zip(champ, chall)]
    non_tied = [value for value in differences if value != 0]
    wins = sum(value > 0 for value in non_tied)
    n = len(non_tied)
    # Exact one-sided binomial sign-test tail under p(win)=0.5; ties are excluded.
    raw_p = (
        sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)
        if n else 1.0
    )
    adjusted_p = min(1.0, raw_p * comparison_family_size)
    mean_improvement = sum(differences) / len(differences)
    mean_champion_drawdown = sum(item["max_drawdown"] for item in champ) / len(champ)
    mean_drawdown = sum(item["max_drawdown"] for item in chall) / len(chall)
    mean_champion_ece = sum(item["ece"] for item in champ) / len(champ)
    mean_ece = sum(item["ece"] for item in chall) / len(chall)
    drawdown_worsening = mean_drawdown - mean_champion_drawdown
    ece_worsening = mean_ece - mean_champion_ece
    per_fold_drawdown_passed = all(item["max_drawdown"] <= max_drawdown for item in chall)
    per_fold_ece_passed = all(item["ece"] <= max_ece for item in chall)
    eligible = (
        adjusted_p <= alpha
        and mean_improvement > min_improvement
        and mean_drawdown <= max_drawdown
        and per_fold_drawdown_passed
        and mean_ece <= max_ece
        and per_fold_ece_passed
        and drawdown_worsening <= max_dd_worsening
        and ece_worsening <= max_ece_worsening_value
    )
    return {
        "promotion_eligible": bool(eligible),
        "status": "eligible_for_human_review" if eligible else "not_eligible",
        "fold_count": len(differences), "non_tied_fold_count": n, "test_bars": total_bars,
        "fold_wins": wins, "mean_net_return_improvement_bps": mean_improvement,
        "sign_test_p_value": raw_p,
        "comparison_family_size": comparison_family_size,
        "bonferroni_adjusted_p_value": adjusted_p,
        "mean_champion_drawdown": mean_champion_drawdown,
        "mean_challenger_drawdown": mean_drawdown,
        "challenger_drawdown_worsening": drawdown_worsening,
        "per_fold_drawdown_passed": per_fold_drawdown_passed,
        "mean_champion_out_of_sample_ece": mean_champion_ece,
        "mean_challenger_out_of_sample_ece": mean_ece,
        "challenger_ece_worsening": ece_worsening,
        "per_fold_ece_passed": per_fold_ece_passed,
        "automatic_promotion": False,
        "requires_human_review": True,
        "warning": (
            "The gate relies on declared locked data, cost-model hashes and fold boundaries. "
            "It cannot independently authenticate data independence, preregistration, source completeness "
            "or the declared comparison family. Passing does not establish profitability or authorize live orders."
        ),
    }
