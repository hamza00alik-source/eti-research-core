import numpy as np
import pandas as pd
import pytest

from eti.instruments import InstrumentSpec
from eti.market_data_hub import MarketDataSnapshot
from eti.multi_market_intelligence import train_multi_market_intelligence_specialists


def make_snapshot(symbol, seed, group):
    rng = np.random.default_rng(seed)
    n = 760
    scales = np.where((np.arange(n) // 110) % 2 == 0, 0.002, 0.009)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0001, scales)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.001, 0.0003, n))
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "open": open_, "high": np.maximum(open_, close) * (1 + spread),
        "low": np.minimum(open_, close) * (1 - spread),
        "close": close, "volume": rng.uniform(10, 1000, n),
    })
    instrument = InstrumentSpec(
        symbol=symbol, venue="testvenue", asset_class="crypto",
        market_type="spot",
    )
    snapshot = MarketDataSnapshot(
        instrument=instrument, timeframe="1h", frame=frame,
        provenance={"venue": "testvenue", "symbol": symbol, "fixture": True},
        quality={"ohlcv_validated": True},
    )
    return {"snapshot": snapshot, "independence_group": group}


def run(markets, **kwargs):
    return train_multi_market_intelligence_specialists(
        markets, horizon=4, min_train=200, validation_size=40,
        test_size=40, step_size=200, max_model_fits=96, **kwargs,
    )


def test_multi_market_validation_keeps_markets_separate_and_reports_lineage():
    result = run([
        make_snapshot("AAA/USDT", 1, "AAA"),
        make_snapshot("BBB/USDT", 2, "BBB"),
        make_snapshot("CCC/USDT", 3, "CCC"),
    ])
    assert result["market_count"] == 3
    assert result["independent_market_count"] == 3
    assert result["summary"]["status"] == "cross_market_evidence_available"
    assert len({row["lineage"]["dataset_sha256"] for row in result["markets"]}) == 3
    assert all(row["status"] == "research_candidate_only" for row in result["markets"])
    assert result["compute_budget"]["preflight_completed_before_any_fit"] is True
    assert result["promotion"]["eligible"] is False
    assert result["execution_enabled"] is False


def test_duplicate_underlying_is_not_counted_as_independent_replication():
    markets = [
        make_snapshot("AAA/USDT", 1, "AAA"),
        make_snapshot("AAA/USD", 2, "AAA"),
    ]
    with pytest.raises(ValueError, match="Independence group"):
        run(markets)


def test_global_fit_budget_fails_before_any_estimator_fit(monkeypatch):
    import eti.market_intelligence_training as training

    def fail_if_called(*args, **kwargs):
        raise AssertionError("fit must not occur before global budget approval")

    monkeypatch.setattr(training.HistGradientBoostingClassifier, "fit", fail_if_called)
    monkeypatch.setattr(training.HistGradientBoostingRegressor, "fit", fail_if_called)
    with pytest.raises(ValueError, match="stopped before fitting"):
        train_multi_market_intelligence_specialists(
            [make_snapshot("AAA/USDT", 1, "AAA"),
             make_snapshot("BBB/USDT", 2, "BBB")],
            horizon=4, min_train=200, validation_size=40,
            test_size=40, step_size=200, max_model_fits=1,
        )


def test_timeframe_mismatch_fails_closed():
    first = make_snapshot("AAA/USDT", 1, "AAA")
    second = make_snapshot("BBB/USDT", 2, "BBB")
    second["snapshot"] = MarketDataSnapshot(
        instrument=second["snapshot"].instrument, timeframe="4h",
        frame=second["snapshot"].frame, provenance={}, quality={},
    )
    with pytest.raises(ValueError, match="same timeframe"):
        run([first, second])
