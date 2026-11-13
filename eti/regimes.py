"""Causal market-regime diagnostics shared across instrument classes.

Regimes are descriptive context, not trade instructions. Thresholds use only
historical observations available before the current bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


REGIME_LABELS = ("unknown", "high_volatility", "uptrend", "downtrend", "range")


def classify_market_regime(
    frame: pd.DataFrame,
    *,
    volatility_window: int = 20,
    trend_window: int = 20,
    percentile_window: int = 100,
    high_volatility_quantile: float = 0.80,
    trend_threshold: float = 1.5,
) -> pd.DataFrame:
    """Return causal volatility/trend features and a categorical regime per bar.

    The current bar's realized volatility is compared with a quantile computed
    from prior bars only. Trend strength is the log directional move over
    trend_window divided by trailing realized volatility scaled by sqrt(window).
    The output is instrument-price-scale independent.
    """
    missing = {"close"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for name, value in (
        ("volatility_window", volatility_window),
        ("trend_window", trend_window),
        ("percentile_window", percentile_window),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    if not 0 < high_volatility_quantile < 1:
        raise ValueError("high_volatility_quantile must be in (0, 1)")
    if not np.isfinite(trend_threshold) or trend_threshold <= 0:
        raise ValueError("trend_threshold must be finite and positive")

    close = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    if close.isna().any() or not np.isfinite(close.to_numpy()).all() or (close <= 0).any():
        raise ValueError("close prices must be finite and positive")

    log_return = np.log(close).diff()
    realized_volatility = log_return.rolling(
        volatility_window, min_periods=volatility_window
    ).std()
    historical_volatility_cutoff = realized_volatility.shift(1).rolling(
        percentile_window, min_periods=percentile_window
    ).quantile(high_volatility_quantile)

    directional_move = np.log(close / close.shift(trend_window))
    trend_scale = realized_volatility * np.sqrt(trend_window)
    trend_strength = directional_move / trend_scale.replace(0, np.nan)

    regime = pd.Series("unknown", index=frame.index, dtype="object")
    ready = historical_volatility_cutoff.notna() & trend_strength.notna()
    high_vol = ready & (realized_volatility > historical_volatility_cutoff)
    regime.loc[ready & ~high_vol] = "range"
    regime.loc[ready & ~high_vol & (trend_strength >= trend_threshold)] = "uptrend"
    regime.loc[ready & ~high_vol & (trend_strength <= -trend_threshold)] = "downtrend"
    regime.loc[high_vol] = "high_volatility"

    return pd.DataFrame(
        {
            "realized_volatility": realized_volatility,
            "historical_volatility_cutoff": historical_volatility_cutoff,
            "trend_strength": trend_strength,
            "regime": regime,
        },
        index=frame.index,
    )
