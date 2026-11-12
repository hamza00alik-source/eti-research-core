import numpy as np
import pandas as pd
import pytest

from eti.backtest import backtest_positions
from eti.decision import build_research_decision
from eti.device_benchmark import benchmark_inference
from eti.evaluation import run_walk_forward
from eti.portfolio import PortfolioLimits
from eti.providers import ProviderRegistry, CcxtPublicProvider


def test_decision_requires_portfolio_gate_after_trade_risk_gate():
    probabilities = {"short": 0.05, "flat": 0.05, "long": 0.90}
    from eti.risk import RiskRequest
    result = build_research_decision(
        probabilities, RiskRequest(1000, 100, 98, 100),
        current_positions={"ETH/USDT": 200}, proposed_symbol="BTC/USDT",
        portfolio_limits=PortfolioLimits(max_gross_exposure_fraction=0.20))
    assert result["action"] == "no_trade"
    assert result["reason"] == "portfolio_rejected"
    assert result["execution"] == "disabled"


def test_decision_rejects_nan_probabilities():
    from eti.risk import RiskRequest
    with pytest.raises(ValueError):
        build_research_decision({"short": 0.1, "flat": 0.1, "long": float("nan")},
                                RiskRequest(1000, 100, 98, 100))


def test_backtest_requires_aligned_finite_inputs_and_positive_equity():
    close = pd.Series([100.0, 101.0, 102.0])
    with pytest.raises(ValueError):
        backtest_positions(close, pd.Series([0.0, 1.0]))
    with pytest.raises(ValueError):
        backtest_positions(close, pd.Series([0.0, float("nan"), 1.0]))
    with pytest.raises(ValueError):
        backtest_positions(close, pd.Series([0.0, 1.0, 1.0]), initial_equity=0)


def test_walk_forward_runner_runs_callback_without_using_test_for_fit():
    frame = pd.DataFrame({"x": np.arange(300, dtype=float),
                          "label": np.tile([0, 1], 150)})
    fit_lengths = []
    def predictor(train_x, train_y, eval_x):
        fit_lengths.append(len(train_x))
        return np.zeros(len(eval_x), dtype=int)
    result = run_walk_forward(frame, ["x"], "label", predictor,
                              min_train=100, validation_size=30, test_size=30,
                              step_size=30, purge_bars=5)
    assert result["fold_count"] > 0
    assert len(fit_lengths) == result["fold_count"] * 2
    assert result["warning"].startswith("Research evaluation only")


def test_provider_registry_fails_closed_and_reports_explicit_capabilities():
    registry = ProviderRegistry()
    registry.register("crypto", CcxtPublicProvider("binance"))
    assert registry.capabilities() == {"crypto": "ccxt-public"}
    with pytest.raises(LookupError):
        registry.get("forex")


def test_device_benchmark_is_inference_only():
    class Model:
        def predict(self, frame):
            return np.zeros(len(frame))
    result = benchmark_inference(Model(), pd.DataFrame({"x": [1.0]}),
                                 warmup=1, repetitions=3)
    assert result["training_performed"] is False
    assert result["latency_ms_p95"] >= 0
