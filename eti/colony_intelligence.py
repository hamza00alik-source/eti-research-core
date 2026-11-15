"""Evidence-bounded operational intelligence for the ETI worker colony.

This module measures task outcomes only. It does not infer strategy quality,
model profitability, or individual-worker skill from task completion.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import math
from statistics import median
from typing import Any


MIN_CAPABILITY_SAMPLE = 20
TASK_HISTORY_LIMIT = 500


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> list[float] | None:
    if trials <= 0:
        return None
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def build_colony_intelligence(
    tasks: list[dict[str, Any]], *, history_limit: int = TASK_HISTORY_LIMIT,
    min_capability_sample: int = MIN_CAPABILITY_SAMPLE,
) -> dict[str, Any]:
    """Summarize bounded task history with conservative capability-level evidence."""
    if not isinstance(tasks, list):
        raise ValueError("tasks must be a list")
    if (isinstance(history_limit, bool) or not isinstance(history_limit, int)
            or not 1 <= history_limit <= TASK_HISTORY_LIMIT):
        raise ValueError("history_limit must be from 1 to 500")
    if (isinstance(min_capability_sample, bool)
            or not isinstance(min_capability_sample, int)
            or min_capability_sample < 1):
        raise ValueError("min_capability_sample must be a positive integer")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    status_counts = {name: 0 for name in ("queued", "running", "completed", "failed", "cancelled")}
    valid_tasks = []
    for task in tasks[:history_limit]:
        if not isinstance(task, dict):
            continue
        capability = task.get("target_capability")
        status = task.get("status")
        if not isinstance(capability, str) or not capability or status not in status_counts:
            continue
        status_counts[status] += 1
        valid_tasks.append(task)
        groups[capability].append(task)

    capabilities = {}
    for capability, records in sorted(groups.items()):
        terminal = [item for item in records if item["status"] in {"completed", "failed"}]
        successes = sum(item["status"] == "completed" for item in terminal)
        failures = len(terminal) - successes
        durations = []
        retries = 0
        for item in terminal:
            attempts = item.get("attempt_count", 0)
            if isinstance(attempts, int) and not isinstance(attempts, bool) and attempts > 0:
                retries += max(0, attempts - 1)
            created = _timestamp(item.get("created_at"))
            updated = _timestamp(item.get("updated_at"))
            if created is not None and updated is not None:
                elapsed = (updated - created).total_seconds()
                if math.isfinite(elapsed) and elapsed >= 0:
                    durations.append(elapsed)

        sample_count = len(terminal)
        mature = sample_count >= min_capability_sample
        capabilities[capability] = {
            "terminal_tasks": sample_count,
            "completed": successes,
            "failed": failures,
            "success_rate": successes / sample_count if sample_count else None,
            "success_rate_wilson_95": _wilson_interval(successes, sample_count),
            "retry_count": retries,
            "median_end_to_end_seconds": median(durations) if durations else None,
            "latency_observations": len(durations),
            "evidence_status": "sufficient_operational_sample" if mature else "insufficient_sample",
            "routing_reliability_eligible": mature,
        }

    return {
        "schema": "eti.colony_intelligence.v1",
        "scope": "owner_workspace_task_history",
        "observed_tasks": len(valid_tasks),
        "history_limit": history_limit,
        "history_may_be_truncated": len(tasks) >= history_limit,
        "minimum_capability_sample": min_capability_sample,
        "status_counts": status_counts,
        "capabilities": capabilities,
        "limitations": [
            "Task completion measures operational reliability, not research or trading quality.",
            "Latency is end-to-end task age and includes queue wait; it is not worker execution time.",
            "Worker identity is not persisted in task records, so individual-worker rankings are unavailable.",
            "Capability samples below the minimum are descriptive only and must not drive reliability ranking.",
            "Overlapping tasks, retries, and changing workloads can make observations dependent.",
        ],
        "model_promotion_authority": False,
        "budget_authority": False,
        "execution_authority": False,
    }
