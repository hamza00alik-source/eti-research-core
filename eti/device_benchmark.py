"""Inference-only latency and memory benchmark for target devices; does not train models."""
from __future__ import annotations

import platform
import statistics
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd


def benchmark_inference(model: Any, features: pd.DataFrame, warmup: int = 3,
                        repetitions: int = 30) -> dict:
    if warmup < 0 or repetitions < 1 or features.empty:
        raise ValueError("Provide non-empty features, non-negative warmup and positive repetitions")
    sample = features.iloc[:min(len(features), 1)]
    for _ in range(warmup):
        model.predict(sample)
    timings = []
    tracemalloc.start()
    for _ in range(repetitions):
        start = time.perf_counter()
        model.predict(sample)
        timings.append((time.perf_counter() - start) * 1000)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)
    return {
        "device": platform.platform(),
        "python": platform.python_version(),
        "sample_rows": int(len(sample)),
        "repetitions": repetitions,
        "latency_ms_median": float(statistics.median(timings)),
        "latency_ms_p95": float(ordered[p95_index]),
        "python_tracemalloc_peak_bytes": int(peak_bytes),
        "memory_note": "Python-traced allocations only; not total process RAM or Android native memory.",
        "training_performed": False,
        "warning": "Run this benchmark on the actual target phone; desktop/Kaggle measurements do not predict Android performance.",
    }
