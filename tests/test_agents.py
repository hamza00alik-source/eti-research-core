import pytest

from eti.agents import AgentCapabilityRegistry, market_snapshot
from eti.storage import MemoryStore
from eti.worker import AgentWorker


def test_worker_executes_only_registered_capability_and_persists_message(tmp_path):
    store = MemoryStore(tmp_path / "agent-worker.sqlite3")
    workspace = store.create_workspace("owner", "research")
    task = store.create_agent_task(
        "owner", workspace["id"], "user", "test.echo", {"symbol": "BTC/USDT"}
    )
    registry = AgentCapabilityRegistry()
    registry.register("test.echo", lambda payload: {"received": payload})
    worker = AgentWorker(store, registry)
    result = worker.run_once()

    assert result["id"] == task["id"]
    assert result["status"] == "completed"
    assert result["attempt_count"] == 1
    assert result["result"] == {"received": {"symbol": "BTC/USDT"}}
    messages = store.list_agent_messages("owner", workspace["id"], task["id"])
    assert len(messages) == 1
    assert messages[0]["message_type"] == "task.completed"
    assert messages[0]["payload"]["status"] == "completed"


def test_unknown_capability_fails_closed_without_executing_user_input(tmp_path):
    store = MemoryStore(tmp_path / "agent-worker.sqlite3")
    workspace = store.create_workspace("owner", "research")
    task = store.create_agent_task(
        "owner", workspace["id"], "user", "shell.execute",
        {"command": "echo this must never run"}
    )
    result = AgentWorker(store, AgentCapabilityRegistry()).run_once()
    assert result["id"] == task["id"]
    assert result["status"] == "failed"
    assert result["result"]["error"]["type"] == "LookupError"
    assert "unsupported agent capability" in result["result"]["error"]["message"]


def test_market_snapshot_rejects_unsupported_inputs_before_network_access():
    with pytest.raises(ValueError, match="venue"):
        market_snapshot({"venue": "unknown", "symbol": "BTC/USDT"})
    with pytest.raises(ValueError, match="symbol"):
        market_snapshot({"venue": "binance", "symbol": "BTCUSDT"})
    with pytest.raises(ValueError, match="timeframe"):
        market_snapshot({"venue": "binance", "symbol": "BTC/USDT", "timeframe": ["1h"]})
    with pytest.raises(ValueError, match="limit"):
        market_snapshot({"venue": "binance", "symbol": "BTC/USDT", "limit": 10000})


def test_worker_does_not_claim_when_queue_is_empty_but_reports_heartbeat(tmp_path):
    store = MemoryStore(tmp_path / "empty-worker.sqlite3")
    worker = AgentWorker(store, AgentCapabilityRegistry())
    result = worker.run_once()
    assert result is None
    workers = store.list_agent_workers()
    assert len(workers) == 1
    assert workers[0]["worker_id"] == worker.worker_id
    assert workers[0]["online"] is True


def test_worker_retries_transient_errors_without_leaking_exception_text(tmp_path):
    store = MemoryStore(tmp_path / "agent-retry.sqlite3")
    workspace = store.create_workspace("owner", "research")
    task = store.create_agent_task(
        "owner", workspace["id"], "user", "test.transient", {"request": 1},
        max_attempts=2,
    )
    attempts = {"count": 0}

    def handler(payload):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("private credential must not appear in task output")
        return {"ok": True, "attempt": attempts["count"]}

    registry = AgentCapabilityRegistry()
    registry.register("test.transient", handler)
    worker = AgentWorker(store, registry)

    first = worker.run_once()
    assert first["id"] == task["id"]
    assert first["status"] == "queued"
    assert first["attempt_count"] == 1
    assert "private credential" not in first["result"]["error"]["message"]

    second = worker.run_once()
    assert second["status"] == "completed"
    assert second["attempt_count"] == 2
    assert second["result"] == {"ok": True, "attempt": 2}


def test_market_snapshot_defaults_to_mexc_public_feed(monkeypatch):
    import pandas as pd
    import eti.agents as agents

    captured = {}
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=50, freq="h"),
        "open": [100.0] * 50, "high": [102.0] * 50,
        "low": [99.0] * 50, "close": [101.0] * 50,
        "volume": [10.0] * 50,
    })
    names = (
        "ret_1", "ret_3", "ret_6", "volatility_20",
        "sma_gap_10", "sma_gap_30", "rsi_14", "range_pct", "atr_pct",
    )

    class FakeProvider:
        def __init__(self, venue, market_type="spot"):
            captured["venue"] = venue
            captured["market_type"] = market_type

        def fetch_ohlcv(self, symbol, timeframe, limit):
            captured["request"] = (symbol, timeframe, limit)
            return frame

    monkeypatch.setattr(agents, "CcxtPublicProvider", FakeProvider)
    monkeypatch.setattr(
        agents, "build_features",
        lambda _frame: pd.DataFrame([{name: 0.1 for name in names}]),
    )
    result = agents.market_snapshot({"symbol": "BTC/USDT"})

    assert captured["venue"] == "mexc"
    assert captured["market_type"] == "spot"
    assert captured["request"] == ("BTC/USDT", "1h", 200)
    assert result["venue"] == "mexc"
    assert result["market_type"] == "spot"
    assert result["research_ranking"]["method"] == "transparent_heuristic_v1"
    assert result["research_ranking"]["is_predictive_model"] is False



def test_strategy_walk_forward_capability_uses_public_data_and_returns_report(monkeypatch):
    import numpy as np
    import pandas as pd
    import eti.agents as agents

    rng = np.random.default_rng(8)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.008, 500)))
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=500, freq="h", tz="UTC"),
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.full(500, 100.0),
    })
    captured = {}
    class FakeProvider:
        def __init__(self, venue, market_type="spot"):
            captured["venue"] = venue
            captured["market_type"] = market_type
        def fetch_ohlcv(self, symbol, timeframe, limit):
            captured["request"] = (symbol, timeframe, limit)
            return frame
    monkeypatch.setattr(agents, "CcxtPublicProvider", FakeProvider)
    result = agents.strategy_walk_forward({"venue": "mexc", "symbol": "BTC/USDT"})
    assert captured["venue"] == "mexc"
    assert captured["request"] == ("BTC/USDT", "1h", 500)
    assert result["kind"] == "strategy_walk_forward_report"
    assert result["fold_count"] >= 2
    assert result["safety"]["is_trained_model"] is False
    assert result["safety"]["places_orders"] is False


def test_strategy_walk_forward_rejects_invalid_request():
    from eti.agents import strategy_walk_forward
    with pytest.raises(ValueError, match="limit"):
        strategy_walk_forward({"symbol": "BTC/USDT", "limit": 200})
    with pytest.raises(ValueError, match="venue"):
        strategy_walk_forward({"venue": "private", "symbol": "BTC/USDT"})


def test_autonomous_snapshot_automatically_queues_strategy_followup(tmp_path):
    store = MemoryStore(tmp_path / "autonomous-followup.sqlite3")
    workspace = store.create_workspace("owner", "research")
    task = store.create_agent_task(
        "owner", workspace["id"], "eti.autonomy.supervisor", "market.snapshot",
        {"venue": "mexc", "symbol": "BTC/USDT", "timeframe": "1h",
         "scheduled_by": "eti.autonomy.supervisor"},
    )
    registry = AgentCapabilityRegistry()
    registry.register("market.snapshot", lambda payload: {"kind": "public_market_snapshot"})
    completed = AgentWorker(store, registry).run_once()
    assert completed["id"] == task["id"] and completed["status"] == "completed"
    queued = store.list_agent_tasks("owner", workspace["id"], status="queued")
    assert len(queued) == 1
    assert queued[0]["target_capability"] == "strategy.walk_forward"
    assert queued[0]["input"]["source_snapshot_task_id"] == task["id"]
    assert queued[0]["input"]["limit"] == 500
    assert queued[0]["input"]["market_type"] == "spot"


def test_model_training_research_creates_unapproved_candidate(monkeypatch):
    import pandas as pd
    import eti.agents as agents
    import eti.pipeline as pipeline

    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=1600, freq="h", tz="UTC"),
        "open": [100.0] * 1600, "high": [102.0] * 1600,
        "low": [99.0] * 1600, "close": [101.0] * 1600,
        "volume": [10.0] * 1600,
    })
    captured = {}

    class FakeProvider:
        def __init__(self, venue, market_type="spot"):
            captured["venue"] = venue
            captured["market_type"] = market_type
        def fetch_ohlcv(self, symbol, timeframe, limit):
            captured["request"] = (symbol, timeframe, limit)
            return frame

    class FakeModel:
        calibration_status = "calibrated"
        calibration_report = {"passed": True, "sample_count": 100}

    monkeypatch.setattr(agents, "CcxtPublicProvider", FakeProvider)
    def fake_nested(*args, **kwargs):
        captured["allow_short"] = kwargs.get("allow_short")
        return {
        "fold_count": 2, "mean_test_balanced_accuracy": 0.51,
        "test_balanced_accuracy_ci": {"lower": 0.45, "upper": 0.57},
        "folds": [{"fold": 0, "selected_candidate": "hist_gradient_boosting",
                   "validation_scores": {"hist_gradient_boosting": 0.51},
                   "test_accuracy": 0.52, "test_balanced_accuracy": 0.51,
                   "boundaries": {"test_start": 700, "test_end": 800}}],
        "trading_performance": {"summary": {"mean_net_return": -0.01}},
        "warning": "research only",
    }
    monkeypatch.setattr(pipeline, "run_nested_experiment", fake_nested)
    monkeypatch.setattr(pipeline, "fit_deployable_model", lambda *a, **k: (
        FakeModel(), {"features": ["ret_1"], "calibration_status": "calibrated"}
    ))

    class FakeArtifactStore:
        def save_model(self, model):
            return {"artifact_uri": "artifact://sha256/" + "a" * 64,
                    "artifact_sha256": "a" * 64}

    import eti.artifacts as artifacts
    monkeypatch.setattr(artifacts, "ModelArtifactStore", FakeArtifactStore)
    result = agents.model_training_research({
        "venue": "mexc", "market_type": "swap", "symbol": "BTC/USDT:USDT",
        "timeframe": "1h", "limit": 1600
    })

    assert captured["venue"] == "mexc"
    assert captured["market_type"] == "swap"
    assert captured["request"] == ("BTC/USDT:USDT", "1h", 1600)
    assert captured["allow_short"] is True
    assert result["market_type"] == "swap"
    assert result["evaluation_passed"] is False
    assert result["kind"] == "trained_model_candidate"
    assert result["training_performed"] is True
    assert result["lifecycle"] == "unregistered_candidate"
    assert result["approval_status"] == "not_approved"
    assert result["candidate_artifact"]["artifact_sha256"] == "a" * 64
    assert result["evaluation"]["fold_count"] == 2
    assert any("no credentials" in item for item in result["limitations"])


def test_model_training_research_rejects_unsafe_limits_before_network():
    from eti.agents import model_training_research
    with pytest.raises(ValueError, match="limit"):
        model_training_research({"symbol": "BTC/USDT", "limit": 500})
    with pytest.raises(ValueError, match="timeframe"):
        model_training_research({"symbol": "BTC/USDT", "timeframe": "5m"})
    with pytest.raises(ValueError, match="venue"):
        model_training_research({"symbol": "BTC/USDT", "venue": []})
    with pytest.raises(ValueError, match="timeframe"):
        model_training_research({"symbol": "BTC/USDT", "timeframe": []})


def test_default_registry_exposes_model_training_as_research_only():
    from eti.agents import default_capabilities
    registry = default_capabilities()
    assert "model.train_research" in registry.capabilities()


def test_model_training_research_runs_real_pipeline_and_writes_candidate(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    import eti.agents as agents
    from eti.artifacts import ModelArtifactStore

    rng = np.random.default_rng(20261113)
    returns = rng.normal(0.0001, 0.012, 1700)
    close = 100 * np.exp(np.cumsum(returns))
    open_price = close * (1 + rng.normal(0, 0.002, len(close)))
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=len(close), freq="h", tz="UTC"),
        "open": open_price,
        "high": np.maximum(open_price, close) * 1.003,
        "low": np.minimum(open_price, close) * 0.997,
        "close": close,
        "volume": rng.lognormal(4, 0.3, len(close)),
    })

    class FakeProvider:
        def __init__(self, venue, market_type="spot"):
            self.venue = venue
            self.market_type = market_type
        def fetch_ohlcv(self, symbol, timeframe, limit):
            return frame.tail(limit).reset_index(drop=True)

    monkeypatch.setattr(agents, "CcxtPublicProvider", FakeProvider)
    import eti.artifacts as artifacts
    monkeypatch.setattr(artifacts, "ModelArtifactStore",
                        lambda: ModelArtifactStore(tmp_path))

    result = agents.model_training_research({
        "venue": "mexc", "symbol": "BTC/USDT", "timeframe": "1h", "limit": 1700
    })

    assert result["training_performed"] is True
    assert result["model_family"] == "HistGradientBoostingClassifier via ETI BaselineModel"
    assert result["evaluation"]["fold_count"] >= 1
    assert result["evaluation"]["folds"][0]["test_balanced_accuracy"] is not None
    assert result["evaluation"]["folds"][0]["market_regimes"]["bar_count"] == 100
    assert result["candidate_artifact"]["artifact_uri"].startswith("artifact://sha256/")
    assert result["lifecycle"] == "unregistered_candidate"
    assert result["approval_status"] == "not_approved"
    assert result["calibration_status"] in {"calibrated", "uncalibrated"}


def test_market_snapshot_accepts_explicit_crypto_perpetual_market(monkeypatch):
    import pandas as pd
    import eti.agents as agents

    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=50, freq="h", tz="UTC"),
        "open": [100.0] * 50, "high": [102.0] * 50,
        "low": [99.0] * 50, "close": [101.0] * 50,
        "volume": [10.0] * 50,
    })
    captured = {}
    names = ("ret_1", "ret_3", "ret_6", "volatility_20", "sma_gap_10",
             "sma_gap_30", "rsi_14", "range_pct", "atr_pct")

    class FakeProvider:
        def __init__(self, venue, market_type="spot"):
            captured["venue"] = venue
            captured["market_type"] = market_type
        def fetch_ohlcv(self, symbol, timeframe, limit):
            captured["symbol"] = symbol
            return frame

    monkeypatch.setattr(agents, "CcxtPublicProvider", FakeProvider)
    monkeypatch.setattr(agents, "build_features",
                        lambda _frame: pd.DataFrame([{name: 0.1 for name in names}]))
    result = agents.market_snapshot({
        "venue": "binance", "market_type": "swap",
        "symbol": "BTC/USDT:USDT", "timeframe": "1h", "limit": 100,
    })
    assert captured == {"venue": "binance", "market_type": "swap", "symbol": "BTC/USDT:USDT"}
    assert result["market_type"] == "swap"
    assert result["research_ranking"] is None
    assert any("funding" in item.lower() for item in result["limitations"])



def test_market_snapshot_accepts_yahoo_ticker_for_equities(monkeypatch):
    import pandas as pd
    import eti.agents as agents

    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=50, freq="h", tz="UTC"),
        "open": [100.0] * 50, "high": [102.0] * 50,
        "low": [99.0] * 50, "close": [101.0] * 50,
        "volume": [1000.0] * 50,
    })
    names = ("ret_1", "ret_3", "ret_6", "volatility_20", "sma_gap_10",
             "sma_gap_30", "rsi_14", "range_pct", "atr_pct")
    captured = {}
    class FakeYahooProvider:
        def __init__(self, asset_class):
            captured["asset_class"] = asset_class
        def fetch_ohlcv(self, symbol, timeframe, limit):
            captured["symbol"] = symbol
            return frame
    monkeypatch.setattr(agents, "YahooFinanceProvider", FakeYahooProvider)
    monkeypatch.setattr(agents, "build_features",
                        lambda _frame: pd.DataFrame([{name: 0.1 for name in names}]))
    result = agents.market_snapshot({"asset_class": "equity", "symbol": "AAPL"})
    assert captured == {"asset_class": "equity", "symbol": "AAPL"}
    assert result["venue"] == "yahoo"
    assert result["asset_class"] == "equity"
    assert result["research_ranking"] is None
