import numpy as np
import pandas as pd
import pytest

from eti.data import drop_incomplete_candles, validate_ohlcv
from eti.features import build_features, make_labels
from eti.pipeline import _timeframe_seconds, run_experiment
from eti.simulator import simulate_long_flat


def sample_ohlcv(n=400):
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = close * 0.003
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "open": open_, "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread, "close": close,
        "volume": rng.uniform(10, 100, n),
    })


def test_validation_sorts_deduplicates_and_drops_non_finite():
    raw = sample_ohlcv()
    raw.loc[0, "volume"] = np.inf
    raw = pd.concat([raw.iloc[[2]], raw.iloc[[0]], raw.iloc[[2]]], ignore_index=True)
    clean = validate_ohlcv(raw)
    assert len(clean) == 1
    assert clean.timestamp.is_monotonic_increasing


def test_incomplete_latest_candle_is_removed():
    raw = sample_ohlcv(3)
    now_ms = int(pd.Timestamp("2024-01-01 02:30", tz="UTC").timestamp() * 1000)
    clean = drop_incomplete_candles(raw, timeframe_ms=60 * 60 * 1000, now_ms=now_ms)
    assert clean.timestamp.tolist() == raw.loc[:1, "timestamp"].tolist()


def test_features_are_causal_at_current_row():
    raw = sample_ohlcv()
    base = build_features(raw)
    changed = raw.copy()
    changed.loc[250:, ["open", "high", "low", "close", "volume"]] *= 1.5
    altered = build_features(changed)
    pd.testing.assert_series_equal(base.loc[249], altered.loc[249], check_names=False)


def test_rsi_handles_no_loss_and_flat_series():
    rising = sample_ohlcv(80)
    rising["close"] = np.arange(80, dtype=float) + 100
    rising["open"] = rising["close"]
    rising["high"] = rising["close"] + 1
    rising["low"] = rising["close"] - 1
    rising_rsi = build_features(rising)["rsi_14"].dropna()
    assert (rising_rsi == 100).all()
    flat = rising.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    flat_rsi = build_features(flat)["rsi_14"].dropna()
    assert (flat_rsi == 50).all()


def test_labels_have_expected_classes_and_tail_nan():
    raw = sample_ohlcv()
    labels = make_labels(raw, horizon=5)
    assert labels.iloc[-5:].isna().all()
    assert set(labels.dropna().unique()).issubset({0, 1, 2})


def test_simulator_applies_signal_next_bar_and_costs():
    close = pd.Series([100.0, 110.0, 121.0, 121.0])
    predictions = pd.Series([2, 2, 1, 1])
    result = simulate_long_flat(close, predictions, fee_bps=0, slippage_bps=0, bars_per_year=24)
    assert result["exposure"] == pytest.approx(0.50)
    assert result["total_return"] == pytest.approx(0.21)
    assert result["bars_per_year"] == 24
    assert "position_changes" in result


def test_timeframe_annualization_is_not_fixed_to_hourly():
    assert _timeframe_seconds("1h") == 3600
    assert _timeframe_seconds("15m") == 900
    assert 4 * _timeframe_seconds("15m") == _timeframe_seconds("1h")


def test_end_to_end_research_pipeline_runs():
    metrics = run_experiment(sample_ohlcv(1200), horizon=6, timeframe="1h")
    assert metrics["rows"] > 200
    assert metrics["timeframe"] == "1h"
    assert metrics["paper_simulation"]["bars_per_year"] == pytest.approx(365.25 * 24)


def test_fetch_ohlcv_continues_after_exchange_returns_short_page(monkeypatch):
    import sys
    import types
    import eti.data as data_module

    now_ms = int(pd.Timestamp("2025-01-10", tz="UTC").timestamp() * 1000)
    start_ms = now_ms - 2000 * 60 * 60 * 1000
    all_rows = []
    for i in range(1200):
        ts = start_ms + i * 60 * 60 * 1000
        price = 100.0 + i * 0.01
        all_rows.append([ts, price, price + 1, price - 1, price, 10.0])

    class FakeExchange:
        has = {"fetchOHLCV": True}
        def __init__(self, config):
            self.calls = []
        def parse_timeframe(self, timeframe):
            return 3600
        def milliseconds(self):
            return now_ms
        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            self.calls.append((since, limit))
            eligible = [row for row in all_rows if row[0] >= since]
            # Simulate an exchange that caps each response at 500 rows,
            # even when the client requests 1000.
            return eligible[:min(limit, 500)]

    fake = FakeExchange({})
    monkeypatch.setattr(data_module, "time", types.SimpleNamespace(sleep=lambda _: None))
    monkeypatch.setitem(sys.modules, "ccxt", types.SimpleNamespace(fakevenue=lambda config: fake))
    result = data_module.fetch_ohlcv("fakevenue", "BTC/USDT", "1h", limit=1200)

    assert len(result) == 1200
    assert len(fake.calls) == 3
    assert result.timestamp.is_monotonic_increasing
