"""Chronological walk-forward split utilities; test folds remain untouched by selection."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int


def walk_forward_folds(
    n_rows: int,
    min_train: int = 500,
    validation_size: int = 100,
    test_size: int = 100,
    step_size: int = 100,
    purge_bars: int = 6,
) -> list[Fold]:
    """Expanding-window folds with purge gaps before validation and test windows."""
    if min(n_rows, min_train, validation_size, test_size, step_size) <= 0 or purge_bars < 0:
        raise ValueError("Row counts and window sizes must be positive; purge cannot be negative")
    folds = []
    test_start = min_train + purge_bars + validation_size + purge_bars
    while test_start + test_size <= n_rows:
        validation_end = test_start - purge_bars
        validation_start = validation_end - validation_size
        train_end = validation_start - purge_bars
        if train_end >= min_train:
            folds.append(Fold(0, train_end, validation_start, validation_end,
                              test_start, test_start + test_size))
        test_start += step_size
    if not folds:
        raise ValueError("Not enough rows to construct one purged walk-forward fold")
    return folds
