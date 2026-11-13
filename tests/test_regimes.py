import numpy as np
import pandas as pd
import pytest

from eti.regimes import REGIME_LABELS, classify_market_regime


def prices(n=260, seed=7):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.004, n)
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"close": close})


def test_regime_output_is_aligned_and_uses_known_labels():
    frame = prices()
    result = classify_market_regime(frame)
    assert result.index.equals(frame.index)
    assert set(result["regime"].unique()).issubset(REGIME_LABELS)
    assert result["regime"].iloc[:120].eq("unknown").all()
    assert result["regime"].iloc[-1] != "unknown"


def test_future_prices_do_not_change_past_regime_labels():
    frame = prices()
    baseline = classify_market_regime(frame)
    changed = frame.copy()
    changed.loc[220:, "close"] *= np.exp(np.linspace(0, 1.5, len(changed.loc[220:])))
    revised = classify_market_regime(changed)
    pd.testing.assert_series_equal(
        baseline["regime"].iloc[:220], revised["regime"].iloc[:220]
    )


def test_invalid_prices_and_parameters_fail_closed():
    frame = prices()
    frame.loc[10, "close"] = 0
    with pytest.raises(ValueError, match="finite and positive"):
        classify_market_regime(frame)
    with pytest.raises(ValueError, match="percentile_window"):
        classify_market_regime(prices(), percentile_window=1)


def test_high_volatility_regime_detects_a_large_volatility_shock():
    frame = prices(n=320)
    frame.loc[280:, "close"] = (
        frame.loc[279, "close"]
        * np.exp(np.cumsum(np.tile([0.08, -0.07], 20)))
    )
    result = classify_market_regime(frame)
    assert "high_volatility" in set(result["regime"].iloc[285:])
