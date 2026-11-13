import math

import pytest

from eti.opportunities import rank_spot_opportunity


def _indicators(**overrides):
    values = {
        "ret_3": 0.012,
        "ret_6": 0.025,
        "volatility_20": 0.018,
        "sma_gap_10": 0.02,
        "sma_gap_30": 0.04,
        "rsi_14": 56.0,
    }
    values.update(overrides)
    return values


def test_bullish_moderate_volatility_ranks_as_long_watch():
    result = rank_spot_opportunity(_indicators())
    assert result["score"] == 95
    assert result["classification"] == "long_watch"
    assert result["method"] == "transparent_heuristic_v1"
    assert result["is_predictive_model"] is False
    assert result["reasons"]


def test_bearish_trend_is_never_classified_as_long_watch():
    result = rank_spot_opportunity(_indicators(
        ret_3=-0.02, ret_6=-0.04, sma_gap_10=-0.03, sma_gap_30=-0.05,
        rsi_14=28.0, volatility_20=0.10,
    ))
    assert result["classification"] == "avoid"
    assert 0 <= result["score"] <= 100


def test_mixed_market_is_monitor_not_trade_signal():
    result = rank_spot_opportunity(_indicators(sma_gap_10=0.01, sma_gap_30=-0.01))
    assert result["classification"] == "monitor"
    assert "not a forecast" in result["disclaimer"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), None, "bad"])
def test_rejects_invalid_indicators(bad):
    with pytest.raises(ValueError):
        rank_spot_opportunity(_indicators(rsi_14=bad))


def test_rejects_missing_indicator():
    values = _indicators()
    del values["ret_3"]
    with pytest.raises(ValueError, match="ret_3"):
        rank_spot_opportunity(values)
