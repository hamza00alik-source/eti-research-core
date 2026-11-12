import pandas as pd
import pytest

from eti.backtest import backtest_positions
from eti.benchmark import compare_baselines
from eti.decision import build_research_decision
from eti.evolution import candidate_set, select_candidate
from eti.markets import normalize_markets
from eti.risk import RiskEngine, RiskRequest
from eti.strategies import moving_average_signal


def test_risk_engine_caps_position_and_rejects_excess_leverage():
    engine = RiskEngine()
    req = RiskRequest(1000, 100, 98, 1000)
    decision = engine.assess(req)
    assert decision.approved
    assert decision.approved_notional <= 100
    blocked = engine.assess(RiskRequest(1000, 100, 98, 100, leverage=2))
    assert not blocked.approved
    assert "leverage_limit" in blocked.reasons


def test_short_disabled_by_default():
    decision = RiskEngine().assess(RiskRequest(1000, 100, 102, 100, side="short"))
    assert not decision.approved
    assert "shorts_disabled" in decision.reasons


def test_market_normalization_only_spot():
    markets = normalize_markets("binance", [
        {"symbol":"BTC/USDT","spot":True,"active":True},
        {"symbol":"BTC/USDT:USDT","swap":True,"active":True},
    ])
    assert len(markets) == 1 and markets[0].market_type == "spot"


def test_short_backtest_requires_explicit_opt_in():
    close = pd.Series([100, 99, 98, 97], dtype=float)
    signal = pd.Series([-1, -1, -1, -1])
    blocked = backtest_positions(close, signal)
    allowed = backtest_positions(close, signal, allow_short=True)
    assert blocked["exposure"] == 0
    assert allowed["total_return"] > 0


def test_evolution_candidates_and_baseline_comparison():
    close = pd.Series([100 + i * 0.1 for i in range(180)], dtype=float)
    assert len(candidate_set()) >= 4
    selected = select_candidate(close)
    assert selected["candidate"] in {"ma", "momentum"}
    assert len(compare_baselines(close)) == 3


def test_decision_requires_risk_gate():
    req = RiskRequest(1000, 100, 98, 100)
    result = build_research_decision({"short":0.1,"flat":0.1,"long":0.8}, req)
    assert result["action"] == "long"
    assert result["execution"] == "disabled"
    blocked = build_research_decision({"short":0.1,"flat":0.1,"long":0.8},
        RiskRequest(1000,100,98,100,leverage=3))
    assert blocked["action"] == "no_trade"


def test_strategy_validates_parameters():
    with pytest.raises(ValueError):
        moving_average_signal(pd.Series([1,2,3]), fast=10, slow=5)
