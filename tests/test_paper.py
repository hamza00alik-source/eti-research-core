import numpy as np
import pandas as pd
import pytest

from eti.features import FEATURE_COLUMNS
from eti.paper import run_paper_tick
from eti.storage import MemoryStore


class _FakeModel:
    def __init__(self):
        self.prediction = 2

    def predict_proba(self, frame):
        if self.prediction == 2:
            return np.array([[0.10, 0.10, 0.80]])
        return np.array([[0.10, 0.80, 0.10]])


def _approved_model(store):
    store.register_model_version(
        "eti-mexc-btc-usdt-1h", "paper-test",
        "artifact://sha256/" + "a" * 64, "a" * 64,
        {"features": FEATURE_COLUMNS}, ["crypto_spot"], "test-dataset",
        {"passed": True}, "calibrated", "trainer",
    )
    store.approve_model_version("eti-mexc-btc-usdt-1h", "paper-test", "reviewer")


def _candles(timestamp):
    return pd.DataFrame({
        "timestamp": pd.date_range(timestamp, periods=80, freq="h", tz="UTC"),
        "open": [100.0] * 80, "high": [102.0] * 80,
        "low": [99.0] * 80, "close": [100.0] * 80,
        "volume": [1000.0] * 80,
    })


def _market_snapshot(frame, bid=99.9, ask=100.1, last=100.0):
    return {
        "schema_version": "eti.live_public_market_snapshot.v1",
        "venue": "mexc", "market_type": "spot", "symbol": "BTC/USDT",
        "quality": {"paper_eligible": True},
        "quote": {
            "bid": bid, "ask": ask, "last": last, "spread_bps": 20.0,
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
        "latest_completed_candle": {
            "open_time": frame.iloc[-1]["timestamp"].isoformat(),
        },
        "snapshot_sha256": "c" * 64,
    }


def test_paper_tick_opens_and_closes_long_position_with_simulated_fills(tmp_path, monkeypatch):
    import eti.paper as paper
    store = MemoryStore(tmp_path / "paper.sqlite3")
    workspace = store.create_workspace("owner", "paper")
    _approved_model(store)
    session = store.create_paper_session(
        "owner", workspace["id"], "eti-mexc-btc-usdt-1h", "paper-test",
        "mexc", "BTC/USDT", "1h", 10000.0,
        {
            "max_risk_per_trade": 0.005, "max_position_fraction": 0.10,
            "max_gross_exposure": 0.25, "max_daily_loss": 0.02,
            "max_drawdown": 0.08, "fee_bps": 6.0, "slippage_bps": 8.0,
            "confidence_threshold": 0.55,
        },
    )
    store.set_paper_session_status("owner", session["id"], "running")
    model = _FakeModel()

    class FakeProvider:
        def __init__(self, venue):
            assert venue == "mexc"
        def fetch_ohlcv(self, symbol, timeframe, limit):
            return _candles("2026-01-01")

    class FakeArtifacts:
        def load_model(self, uri, digest):
            assert digest == "a" * 64
            return model

    monkeypatch.setattr(paper, "CcxtPublicProvider", FakeProvider)
    monkeypatch.setattr(paper, "ModelArtifactStore", FakeArtifacts)
    monkeypatch.setattr(paper, "build_features", lambda frame: pd.DataFrame(
        [[0.0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS, index=[79]
    ))

    opened = run_paper_tick(
        {"owner_id": "owner", "session_id": session["id"]}, store,
        market_snapshot_provider=lambda *args: _market_snapshot(_candles("2026-01-01")),
    )
    assert opened["processed"] is True
    assert opened["live_execution"] is False
    assert opened["position_qty"] > 0
    assert opened["events"][0]["event_type"] == "entry"
    assert opened["events"][0]["price"] == pytest.approx(100.1 * (1 + 8 / 10000))
    assert opened["quote"]["bid"] == 99.9
    assert opened["market_snapshot_sha256"] == "c" * 64
    assert opened["events"][0]["market_snapshot_sha256"] == "c" * 64
    assert opened["events"][0]["quote_timestamp"] == "2026-01-01T00:00:00+00:00"
    persisted = store.list_paper_trades("owner", session["id"])[0]["market_evidence"]
    assert persisted["market_snapshot_sha256"] == "c" * 64
    assert persisted["bid"] == 99.9
    assert persisted["ask"] == 100.1
    assert opened["cash"] < 10000

    # The same completed candle is idempotent and cannot create a duplicate fill.
    duplicate = run_paper_tick(
        {"owner_id": "owner", "session_id": session["id"]}, store,
        market_snapshot_provider=lambda *args: _market_snapshot(_candles("2026-01-01")),
    )
    assert duplicate["processed"] is False
    assert len(store.list_paper_trades("owner", session["id"])) == 1

    model.prediction = 1
    class NextProvider:
        def __init__(self, venue):
            pass
        def fetch_ohlcv(self, symbol, timeframe, limit):
            frame = _candles("2026-01-02")
            return frame
    monkeypatch.setattr(paper, "CcxtPublicProvider", NextProvider)
    monkeypatch.setattr(paper, "build_features", lambda frame: pd.DataFrame(
        [[0.0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS, index=[79]
    ))
    closed = run_paper_tick(
        {"owner_id": "owner", "session_id": session["id"]}, store,
        market_snapshot_provider=lambda *args: _market_snapshot(_candles("2026-01-02")),
    )
    assert closed["processed"] is True
    assert closed["position_qty"] == 0
    assert closed["events"][0]["event_type"] == "exit"
    assert len(store.list_paper_trades("owner", session["id"])) == 2


def test_paper_session_requires_approved_crypto_spot_model(tmp_path):
    store = MemoryStore(tmp_path / "paper-approval.sqlite3")
    workspace = store.create_workspace("owner", "paper")
    store.register_model_version(
        "eti-core", "candidate", "artifact://sha256/" + "b" * 64, "b" * 64,
        {"features": FEATURE_COLUMNS}, ["crypto_spot"], "dataset",
        {"passed": False}, "failed", "trainer",
    )
    import pytest
    with pytest.raises(ValueError, match="approved model"):
        store.create_paper_session(
            "owner", workspace["id"], "eti-core", "candidate",
            "mexc", "BTC/USDT", "1h", 1000.0, {},
        )


def test_paper_scheduler_only_enqueues_running_sessions(tmp_path):
    from datetime import datetime, timezone
    store = MemoryStore(tmp_path / "paper-scheduler.sqlite3")
    workspace = store.create_workspace("owner", "paper")
    _approved_model(store)
    session = store.create_paper_session(
        "owner", workspace["id"], "eti-mexc-btc-usdt-1h", "paper-test",
        "mexc", "BTC/USDT", "1h", 1000.0, {},
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert store.enqueue_due_paper_tasks(start) == 0
    store.set_paper_session_status("owner", session["id"], "running")
    assert store.enqueue_due_paper_tasks(start) == 1
    assert store.enqueue_due_paper_tasks(start) == 0
    tasks = store.list_agent_tasks("owner", workspace["id"], status="queued")
    assert tasks[0]["target_capability"] == "paper.tick"


def test_paper_session_risk_limits_cannot_be_relaxed(tmp_path):
    import pytest
    store = MemoryStore(tmp_path / "paper-risk.sqlite3")
    workspace = store.create_workspace("owner", "paper")
    _approved_model(store)
    with pytest.raises(ValueError, match="max_position_fraction"):
        store.create_paper_session(
            "owner", workspace["id"], "eti-mexc-btc-usdt-1h", "paper-test",
            "mexc", "BTC/USDT", "1h", 1000.0,
            {"max_position_fraction": 0.50},
        )


def test_paper_tick_enforces_stop_loss_and_does_not_reenter_same_candle(tmp_path, monkeypatch):
    import eti.paper as paper
    store = MemoryStore(tmp_path / "paper-stop.sqlite3")
    workspace = store.create_workspace("owner", "paper")
    _approved_model(store)
    session = store.create_paper_session(
        "owner", workspace["id"], "eti-mexc-btc-usdt-1h", "paper-test",
        "mexc", "BTC/USDT", "1h", 10000.0, {},
    )
    store.set_paper_session_status("owner", session["id"], "running")

    class FakeModel:
        def predict_proba(self, frame):
            return np.array([[0.05, 0.05, 0.90]])

    class FakeProvider:
        calls = 0
        def __init__(self, venue):
            pass
        def fetch_ohlcv(self, symbol, timeframe, limit):
            type(self).calls += 1
            price = 100.0 if type(self).calls == 1 else 94.0
            return pd.DataFrame({
                "timestamp": pd.date_range(
                    "2026-02-01" if self.calls == 1 else "2026-02-02",
                    periods=80, freq="h", tz="UTC"
                ),
                "open": [price] * 80, "high": [price * 1.01] * 80,
                "low": [price * 0.99] * 80, "close": [price] * 80,
                "volume": [1000.0] * 80,
            })

    class FakeArtifacts:
        def load_model(self, uri, digest):
            return FakeModel()

    monkeypatch.setattr(paper, "CcxtPublicProvider", FakeProvider)
    monkeypatch.setattr(paper, "ModelArtifactStore", FakeArtifacts)
    monkeypatch.setattr(paper, "build_features", lambda frame: pd.DataFrame(
        [[0.0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS, index=[79]
    ))

    opened = run_paper_tick(
        {"owner_id": "owner", "session_id": session["id"]}, store,
        market_snapshot_provider=lambda *args: _market_snapshot(_candles("2026-02-01")),
    )
    assert opened["position_qty"] > 0
    stopped = run_paper_tick(
        {"owner_id": "owner", "session_id": session["id"]}, store,
        market_snapshot_provider=lambda *args: _market_snapshot(
            _candles("2026-02-02"), bid=94.0, ask=94.1, last=94.0
        ),
    )
    assert stopped["position_qty"] == 0
    assert stopped["events"][0]["event_type"] == "risk_stop"
    assert stopped["events"][0]["risk_decision"]["reason"] == "fixed_five_percent_stop"


def test_paper_feed_health_degrades_after_failures_and_recovers(tmp_path):
    from eti.live_market import MarketDataUnavailable
    store = MemoryStore(tmp_path / "paper-feed-health.sqlite3")
    workspace = store.create_workspace("owner", "paper")
    _approved_model(store)
    session = store.create_paper_session(
        "owner", workspace["id"], "eti-mexc-btc-usdt-1h", "paper-test",
        "mexc", "BTC/USDT", "1h", 1000.0, {},
    )
    first = store.record_paper_feed_health(
        "owner", session["id"], error=MarketDataUnavailable("temporary quote outage")
    )
    assert first["status"] == "degraded"
    assert first["consecutive_failures"] == 1
    assert first["last_error_type"] == "MarketDataUnavailable"

    second = store.record_paper_feed_health(
        "owner", session["id"], error=MarketDataUnavailable("temporary quote outage")
    )
    assert second["status"] == "degraded"
    assert second["consecutive_failures"] == 2

    third = store.record_paper_feed_health(
        "owner", session["id"], error=MarketDataUnavailable("temporary quote outage")
    )
    assert third["status"] == "unavailable"
    assert third["consecutive_failures"] == 3

    recovered = store.record_paper_feed_health(
        "owner", session["id"],
        snapshot={
            "snapshot_sha256": "d" * 64,
            "quote": {"timestamp": "2026-11-20T10:00:00+00:00"},
        },
    )
    assert recovered["status"] == "healthy"
    assert recovered["consecutive_failures"] == 0
    assert recovered["last_error_type"] is None
    assert recovered["last_snapshot_sha256"] == "d" * 64


def test_worker_paper_tick_wrapper_records_retryable_feed_failure(tmp_path, monkeypatch):
    from eti.agents import AgentCapabilityRegistry
    from eti.live_market import MarketDataUnavailable
    import eti.worker as worker_module

    store = MemoryStore(tmp_path / "paper-feed-worker.sqlite3")
    workspace = store.create_workspace("owner", "paper")
    _approved_model(store)
    session = store.create_paper_session(
        "owner", workspace["id"], "eti-mexc-btc-usdt-1h", "paper-test",
        "mexc", "BTC/USDT", "1h", 1000.0, {},
    )
    worker = worker_module.AgentWorker(store, registry=AgentCapabilityRegistry())

    def unavailable(payload, store):
        raise MarketDataUnavailable("temporary quote outage")

    monkeypatch.setattr(worker_module, "run_paper_tick", unavailable)
    with pytest.raises(MarketDataUnavailable):
        worker._run_paper_tick_supervised({
            "owner_id": "owner", "session_id": session["id"],
        })
    health = store.get_paper_feed_health("owner", session["id"])
    assert health["status"] == "degraded"
    assert health["consecutive_failures"] == 1


def test_worker_retries_paper_feed_three_times_then_marks_unavailable(tmp_path, monkeypatch):
    from eti.live_market import MarketDataUnavailable
    import eti.worker as worker_module

    store = MemoryStore(tmp_path / "paper-feed-retry.sqlite3")
    workspace = store.create_workspace("owner", "paper")
    _approved_model(store)
    session = store.create_paper_session(
        "owner", workspace["id"], "eti-mexc-btc-usdt-1h", "paper-test",
        "mexc", "BTC/USDT", "1h", 1000.0, {},
    )
    store.set_paper_session_status("owner", session["id"], "running")
    assert store.enqueue_due_paper_tasks() == 1

    def unavailable(payload, store):
        raise MarketDataUnavailable("temporary quote outage")

    monkeypatch.setattr(worker_module, "run_paper_tick", unavailable)
    worker = worker_module.AgentWorker(store)
    first = worker.run_once()
    assert first["status"] == "queued"
    assert first["attempt_count"] == 1
    second = worker.run_once()
    assert second["status"] == "queued"
    assert second["attempt_count"] == 2
    third = worker.run_once()
    assert third["status"] == "failed"
    assert third["attempt_count"] == 3
    health = store.get_paper_feed_health("owner", session["id"])
    assert health["status"] == "unavailable"
    assert health["consecutive_failures"] == 3
