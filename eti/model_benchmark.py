"""Compare already-fitted inference models on identical feature rows; no fitting occurs."""
from __future__ import annotations

from typing import Any
import pandas as pd
from .device_benchmark import benchmark_inference


def compare_fitted_models(models: dict[str, Any], features: pd.DataFrame,
                          warmup: int = 3, repetitions: int = 30) -> list[dict]:
    if not models:
        raise ValueError("At least one already-fitted model is required")
    results = []
    for name, model in models.items():
        if not name:
            raise ValueError("Model names cannot be empty")
        results.append({"model": name, **benchmark_inference(
            model, features, warmup=warmup, repetitions=repetitions)})
    return sorted(results, key=lambda row: row["latency_ms_p95"])
