import pandas as pd
import pytest

from eti.market_data_hub import MarketDataSnapshot
from eti.research_runner import run_approved_multi_market_research


def cost_model():
    return {
        "schema_version": "eti.transaction_cost_model.v1",
        "instrument_identity": "kraken:spot:BTC/USD",
        "quote_currency": "USD", "settlement_currency": "USD", "fee_basis": "taker",
        "fee_bps": 6, "spread_bps": 2, "slippage_bps": 1,
        "market_impact_bps": 1, "funding_bps_per_bar": 0,
        "borrow_bps_per_bar": 0, "contract_multiplier": 1,
        "retrieved_at": "2026-01-01T00:00:00Z",
        "valid_from": "2026-01-01T00:00:00Z",
        "valid_until": "2030-01-01T00:00:00Z",
        "evidence": [{
            "source": "Published venue fee schedule",
            "uri": "https://example.com/fees",
            "license_status": "open",
            "observed_at": "2026-01-01T00:00:00Z",

            "covers": ["fee_bps", "spread_bps", "slippage_bps", "market_impact_bps", "funding_bps_per_bar", "borrow_bps_per_bar", "contract_multiplier", "fee_basis"],
        }],
    }


def approval(**overrides):
    value = {
        "schema_version": "eti.research_approval.v1",
        "approved": True,
        "authorize_research_fits": True,
        "execution_enabled": False,
        "max_model_fit_calls": 32,
        "usd_budget": {
            "estimated_run_cost_usd": 2.0,
            "daily_spend_to_date_usd": 1.0,
            "monthly_spend_to_date_usd": 10.0,
        },
        "evaluation": {
            "horizon": 6, "min_train": 400, "validation_size": 100,
            "test_size": 100, "step_size": 200,
        },
        "instruments": {"kraken:spot:BTC/USD": {
            "fee_bps": 6.0, "slippage_bps": 1.0,
            "spread_bps": 2.0, "market_impact_bps": 1.0,
            "funding_bps_per_bar": 0.0, "borrow_bps_per_bar": 0.0,
            "allow_short": False,
        }},
    }
    value.update(overrides)
    return value


def allocation(eligible=True):
    return {
        "plan_only": True, "training_authorized": False,
        "candidates": [{
            "instrument": {
                "venue": "kraken", "symbol": "BTC/USD",
                "asset_class": "crypto", "market_type": "spot", "contract_multiplier": 1.0, "active": True,
                "supports_short": False, "supports_funding": False,
            },
            "economic_evaluation_eligible": eligible,
            "provenance": {"requested_rows": 1000, "timeframe": "1h"},
            "quality": {"rows": 1000, "ohlcv_validated": True, "timestamp_order": "ascending", "last_candle_age_seconds": 100, "missing_interval_estimate": 0, "max_gap_seconds": 3600},
            "transaction_cost_model_status": "validated" if eligible else "unknown",
            "transaction_cost_model": cost_model() if eligible else None,
            "training_authorized": False,
            "eligibility_reasons": [] if eligible else ["transaction_cost_model_not_validated"],
        }],
    }


def scan():
    return {
        "scan_only": True, "execution_enabled": False,
        "candidates": [{
            "status": "data_ready",
            "provenance": {"requested_rows": 1000, "timeframe": "1h"},
            "quality": {"rows": 1000, "ohlcv_validated": True, "timestamp_order": "ascending", "last_candle_age_seconds": 100, "missing_interval_estimate": 0, "max_gap_seconds": 3600},
            "instrument": {"venue": "kraken", "symbol": "BTC/USD", "market_type": "spot", "contract_multiplier": 1.0, "active": True, "supports_short": False, "supports_funding": False, "asset_class": "crypto"},
        }],
        "config": {
            "timeframe": "1h", "requested_rows": 1000,
            "minimum_valid_rows": 662,
        },
    }


class FakeHub:
    def __init__(self, *, variable_volume=False):
        self.variable_volume = variable_volume

    def fetch(self, instrument, timeframe, limit, min_rows):
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=1000, freq="h", tz="UTC"),
            "open": [100.0 + i * 0.01 for i in range(1000)],
            "high": [101.0 + i * 0.01 for i in range(1000)],
            "low": [99.0 + i * 0.01 for i in range(1000)],
            "close": [100.5 + i * 0.01 for i in range(1000)],
            # The volume z-score is a required feature; constant volume makes
            # its rolling standard deviation zero and drops every row.
            "volume": [10.0 + (i % 13) for i in range(1000)] if self.variable_volume else [10.0] * 1000,
        })
        return MarketDataSnapshot(
            instrument, timeframe, frame,
            {"venue": instrument.venue, "symbol": instrument.symbol},
            {"rows": len(frame), "ohlcv_validated": True},
        )


def test_runner_requires_explicit_approval_and_keeps_execution_disabled():
    unapproved = approval(approved=False)
    with pytest.raises(ValueError, match="must both be explicit"):
        run_approved_multi_market_research(scan(), allocation(), unapproved, hub=FakeHub())


def test_runner_rejects_candidates_without_validated_economic_eligibility():
    with pytest.raises(ValueError, match="not eligible"):
        run_approved_multi_market_research(
            scan(), allocation(eligible=False), approval(), hub=FakeHub()
        )


@pytest.mark.parametrize("budget", [
    {"estimated_run_cost_usd": 11, "daily_spend_to_date_usd": 0, "monthly_spend_to_date_usd": 0},
    {"estimated_run_cost_usd": 2, "daily_spend_to_date_usd": 9, "monthly_spend_to_date_usd": 0},
    {"estimated_run_cost_usd": 2, "daily_spend_to_date_usd": 0, "monthly_spend_to_date_usd": 99},
])
def test_runner_fails_closed_on_usd_ceilings(budget):
    with pytest.raises(ValueError, match="ceiling exceeded"):
        run_approved_multi_market_research(
            scan(), allocation(), approval(usd_budget=budget), hub=FakeHub()
        )


def test_runner_preflights_global_fit_cap_before_any_fit(monkeypatch):
    import eti.research_runner as runner

    monkeypatch.setattr(runner, "estimate_nested_model_fits_from_rows", lambda *a, **k: 40)
    def should_not_fit(*args, **kwargs):
        raise AssertionError("model fitting must not start when global cap is exceeded")
    monkeypatch.setattr(runner, "run_nested_experiment", should_not_fit)

    with pytest.raises(ValueError, match="stopped before any model fitting"):
        run_approved_multi_market_research(
            scan(), allocation(), approval(max_model_fit_calls=32), hub=FakeHub()
        )


def test_runner_rejects_execution_enabled_approval():
    with pytest.raises(ValueError, match="execution_enabled=false"):
        run_approved_multi_market_research(
            scan(), allocation(), approval(execution_enabled=True), hub=FakeHub()
        )


def test_runner_rejects_zero_contract_multiplier_instead_of_defaulting():
    from eti.research_runner import _instrument
    row = allocation()["candidates"][0]
    row["instrument"]["contract_multiplier"] = 0
    with pytest.raises(ValueError, match="contract_multiplier"):
        _instrument(row)


def test_runner_rejects_inactive_instrument():
    inactive_scan = scan()
    inactive_allocation = allocation()
    inactive_scan["candidates"][0]["instrument"]["active"] = False
    inactive_allocation["candidates"][0]["instrument"]["active"] = False
    with pytest.raises(ValueError, match="inactive"):
        run_approved_multi_market_research(
            inactive_scan, inactive_allocation, approval(), hub=FakeHub()
        )


def test_runner_rejects_modified_scan_quality_in_allocation():
    altered = allocation()
    altered["candidates"][0]["quality"]["last_candle_age_seconds"] = 0
    with pytest.raises(ValueError, match="provenance/quality differs"):
        run_approved_multi_market_research(scan(), altered, approval(), hub=FakeHub())


def test_signed_funding_number_is_accepted_without_relaxing_other_costs():
    from eti.research_runner import _number
    assert _number("funding", -3.25, allow_negative=True) == -3.25
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        _number("fee", -3.25)


def test_derivative_economic_evaluation_is_blocked_before_market_data_fetch():
    candidate_scan = scan()
    candidate_allocation = allocation()
    for report in (candidate_scan, candidate_allocation):
        instrument = report["candidates"][0]["instrument"]
        instrument["market_type"] = "swap"
        instrument["supports_funding"] = True
    candidate_allocation["candidates"][0]["transaction_cost_model"]["instrument_identity"] = "kraken:swap:BTC/USD"
    approved = approval()
    approved["instruments"]["kraken:swap:BTC/USD"] = approved["instruments"].pop("kraken:spot:BTC/USD")
    class NoFetchHub:
        def fetch(self, *args, **kwargs):
            raise AssertionError("derivative data must not be fetched for unsupported economics")

    with pytest.raises(ValueError, match="derivative economic evaluation is blocked"):
        run_approved_multi_market_research(
            candidate_scan, candidate_allocation, approved, hub=NoFetchHub()
        )



def test_funding_event_alignment_preflight_rejects_off_bar_settlement():
    from eti.research_runner import _validate_funding_event_alignment

    bars = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")
    with pytest.raises(ValueError, match="does not align.*before model fitting"):
        _validate_funding_event_alignment(
            [{"timestamp": "2026-01-01T02:30:00Z", "rate_bps": 2.0}],
            pd.Series(bars), "kraken:swap:BTC/USD",
        )


def test_funding_event_alignment_preflight_accepts_exact_utc_bar():
    from eti.research_runner import _validate_funding_event_alignment

    bars = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")
    _validate_funding_event_alignment(
        [{"timestamp": "2026-01-01T02:00:00+00:00", "rate_bps": 2.0}],
        pd.Series(bars), "kraken:swap:BTC/USD",
    )



def test_spot_short_economic_evaluation_is_blocked_until_borrow_costs_supported():
    approved = approval()
    approved["instruments"]["kraken:spot:BTC/USD"]["allow_short"] = True
    candidate_scan = scan()
    candidate_allocation = allocation()
    candidate_scan["candidates"][0]["instrument"]["supports_short"] = True
    candidate_allocation["candidates"][0]["instrument"]["supports_short"] = True
    with pytest.raises(ValueError, match="short research is blocked"):
        run_approved_multi_market_research(
            candidate_scan, candidate_allocation, approved, hub=FakeHub()
        )


def test_runner_records_every_candidate_in_the_trial_ledger(monkeypatch):
    import eti.research_runner as runner
    from types import SimpleNamespace

    monkeypatch.setattr(runner, "estimate_nested_model_fits_from_rows", lambda *args, **kwargs: 4)
    def fake_folds(n_rows, **kwargs):
        return [SimpleNamespace(
            train_end=min(4, n_rows),
            test_start=min(8, n_rows),
            test_end=min(12, n_rows),
        )]
    monkeypatch.setattr("eti.evaluation.walk_forward_folds", fake_folds)

    class FakeLedger:
        def __init__(self):
            self.registered, self.started, self.completed, self.failed = [], [], [], []
        def register_research_trial(self, *args):
            self.registered.append(args)
        def start_research_trial(self, *args):
            self.started.append(args)
        def complete_research_trial(self, *args):
            self.completed.append(args)
        def fail_research_trial(self, *args):
            self.failed.append(args)

    timestamps = pd.date_range("2026-01-01", periods=16, freq="h", tz="UTC")
    def fake_nested(*args, **kwargs):
        return {"candidate_oos_returns": {
            name: {
                "timestamps": [value.isoformat() for value in timestamps],
                "net_return_differentials": [0.001 + i * 0.00001 for i in range(16)],
                "benchmark_id": "no_trade", "ledger_compatible": True,
            } for name in runner.NESTED_CANDIDATE_NAMES
        }}
    monkeypatch.setattr(runner, "run_nested_experiment", fake_nested)
    ledger = FakeLedger()
    result = run_approved_multi_market_research(
        scan(), allocation(), approval(evaluation={
            "horizon": 6, "min_train": 400, "validation_size": 100,
            "test_size": 100, "step_size": 100,
        }), hub=FakeHub(variable_volume=True),
        trial_ledger=ledger, owner_id="owner-1", workspace_id="workspace-1",
    )
    market = result["markets"]["kraken:spot:BTC/USD"]
    assert market["status"] == "evaluated"
    assert len(ledger.registered) == len(ledger.started) == len(ledger.completed) == 2
    assert ledger.failed == []
    assert len({record[3] for record in ledger.registered}) == 2
    assert all(record[9] == "no_trade:kraken:spot:BTC/USD" for record in ledger.registered)
    manifests = [record[10] for record in ledger.registered]
    assert all(item["schema_version"] == "eti.research_source_evidence.v1" for item in manifests)
    assert all(item["capture_method"] == "MarketDataHub.fetch" for item in manifests)
    assert all(item["training_data_sha256"] == record[6] for item, record in zip(manifests, ledger.registered))
    assert market["report"]["trial_ledger"]["source_evidence_sha256"]
    assert market["report"]["trial_ledger"]["completed_candidate_count"] == 2


def test_runner_marks_registered_trials_failed_when_evaluation_fails(monkeypatch):
    import eti.research_runner as runner

    class FakeLedger:
        def __init__(self):
            self.started, self.failed = [], []
        def register_research_trial(self, *args):
            pass
        def start_research_trial(self, *args):
            self.started.append(args)
        def complete_research_trial(self, *args):
            raise AssertionError("failed evaluation must not complete a trial")
        def fail_research_trial(self, *args):
            self.failed.append(args)

    monkeypatch.setattr(runner, "estimate_nested_model_fits_from_rows", lambda *args, **kwargs: 4)
    from types import SimpleNamespace
    def fake_folds(n_rows, **kwargs):
        return [SimpleNamespace(
            train_end=min(4, n_rows),
            test_start=min(8, n_rows),
            test_end=min(12, n_rows),
        )]
    monkeypatch.setattr("eti.evaluation.walk_forward_folds", fake_folds)
    monkeypatch.setattr(
        runner, "run_nested_experiment",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic training failure")),
    )
    ledger = FakeLedger()
    result = run_approved_multi_market_research(
        scan(), allocation(), approval(evaluation={
            "horizon": 6, "min_train": 400, "validation_size": 100,
            "test_size": 100, "step_size": 100,
        }), hub=FakeHub(),
        trial_ledger=ledger, owner_id="owner-1", workspace_id="workspace-1",
    )
    assert result["markets"]["kraken:spot:BTC/USD"]["status"] == "evaluation_failed"
    assert len(ledger.started) == len(ledger.failed) == 2


def test_runner_persists_candidate_trials_in_real_sqlite_ledger(tmp_path, monkeypatch):
    import eti.research_runner as runner
    from types import SimpleNamespace
    from eti.storage import MemoryStore

    monkeypatch.setattr(runner, "estimate_nested_model_fits_from_rows", lambda *args, **kwargs: 4)
    monkeypatch.setattr(
        "eti.evaluation.walk_forward_folds",
        lambda n_rows, **kwargs: [SimpleNamespace(
            train_end=min(4, n_rows),
            test_start=min(8, n_rows),
            test_end=min(12, n_rows),
        )],
    )

    timestamps = pd.date_range("2026-01-01", periods=16, freq="h", tz="UTC")
    monkeypatch.setattr(
        runner, "run_nested_experiment",
        lambda *args, **kwargs: {"candidate_oos_returns": {
            name: {
                "timestamps": [value.isoformat() for value in timestamps],
                "net_return_differentials": [0.001 + i * 0.00001 for i in range(16)],
                "benchmark_id": "no_trade",
                "ledger_compatible": True,
            } for name in runner.NESTED_CANDIDATE_NAMES
        }},
    )

    database = tmp_path / "runner-ledger.sqlite3"
    store = MemoryStore(database)
    workspace = store.create_workspace("owner-1", "runner integration")
    result = run_approved_multi_market_research(
        scan(), allocation(), approval(evaluation={
            "horizon": 6, "min_train": 400, "validation_size": 100,
            "test_size": 100, "step_size": 100,
        }), hub=FakeHub(variable_volume=True),
        trial_ledger=store, owner_id="owner-1", workspace_id=workspace["id"],
    )

    market = result["markets"]["kraken:spot:BTC/USD"]
    assert market["status"] == "evaluated"
    ledger_report = market["report"]["trial_ledger"]
    assert ledger_report["registered_candidate_count"] == 2
    assert ledger_report["completed_candidate_count"] == 2

    # Reopen the database to prove this is durable storage, not an in-memory fake.
    reopened = MemoryStore(database)
    matrix = reopened.get_research_trial_matrix(
        "owner-1", workspace["id"], ledger_report["family_id"]
    )
    assert matrix["registered_trial_count"] == 2
    assert matrix["attempted_trial_count"] == 2
    assert matrix["completed_trial_count"] == 2
    assert matrix["failed_trial_count"] == 0
    assert matrix["trial_ids"] == sorted(ledger_report["trial_ids"].values())
    assert len(matrix["observation_times"]) == 16
    assert len(matrix["net_return_differentials"]) == 16
    assert all(len(row) == 2 for row in matrix["net_return_differentials"])

    # Continue from the durable SQLite matrix into the same statistical
    # assessment used by the authenticated diagnostics API.
    from eti.statistical_validation import assess_research_trial_family
    diagnostics = assess_research_trial_family(
        matrix,
        selected_trial_id=matrix["trial_ids"][0],
        bootstrap_repetitions=200,
        block_length=2,
        pbo_splits=8,
        seed=19,
    )
    assert diagnostics["attempted_trial_count"] == 2
    assert diagnostics["family_pbo"]["pbo"] == 0.5
    assert diagnostics["white_reality_check"]["bootstrap_repetitions"] == 200
    assert diagnostics["hansen_spa"]["bootstrap_repetitions"] == 200
    assert result["execution_enabled"] is False


def test_runner_exposes_registered_but_unstarted_trial_without_counting_it_as_attempted(monkeypatch):
    import eti.research_runner as runner
    from types import SimpleNamespace

    class StartFailureLedger:
        def __init__(self):
            self.registered = []
            self.failed = []
        def register_research_trial(self, *args):
            self.registered.append(args)
        def start_research_trial(self, *args):
            raise RuntimeError("synthetic ledger start failure")
        def fail_research_trial(self, *args):
            self.failed.append(args)

    monkeypatch.setattr(runner, "estimate_nested_model_fits_from_rows", lambda *args, **kwargs: 4)
    monkeypatch.setattr(
        "eti.evaluation.walk_forward_folds",
        lambda n_rows, **kwargs: [SimpleNamespace(
            train_end=min(4, n_rows),
            test_start=min(8, n_rows),
            test_end=min(12, n_rows),
        )],
    )
    ledger = StartFailureLedger()
    result = run_approved_multi_market_research(
        scan(), allocation(), approval(evaluation={
            "horizon": 6, "min_train": 400, "validation_size": 100,
            "test_size": 100, "step_size": 100,
        }), hub=FakeHub(variable_volume=True),
        trial_ledger=ledger, owner_id="owner-1", workspace_id="workspace-1",
    )

    market = result["markets"]["kraken:spot:BTC/USD"]
    assert market["status"] == "evaluation_failed"
    lifecycle = market["trial_ledger"]
    assert len(ledger.registered) == 1
    assert lifecycle["registered_candidate_count"] == 1
    assert lifecycle["started_candidate_count"] == 0
    assert lifecycle["registered_trial_ids"]
    assert lifecycle["started_trial_ids"] == {}
    assert ledger.failed == []  # Registered-only trials cannot be failed as running attempts.


def test_runner_real_model_pipeline_persists_oos_trials_to_sqlite(tmp_path):
    import numpy as np
    from eti.storage import MemoryStore

    rng = np.random.default_rng(20261119)
    count = 1000
    returns = rng.normal(0.0001, 0.012, count)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=count, freq="h", tz="UTC"),
        "open": open_,
        "high": np.maximum(open_, close) * 1.002,
        "low": np.minimum(open_, close) * 0.998,
        "close": close,
        "volume": rng.uniform(1.0, 100.0, count),
    })

    class SyntheticHub:
        def fetch(self, instrument, timeframe, limit, min_rows):
            return MarketDataSnapshot(
                instrument, timeframe, frame.copy(),
                {"venue": instrument.venue, "symbol": instrument.symbol, "source": "seeded_synthetic_test"},
                {"rows": len(frame), "ohlcv_validated": True},
            )

    database = tmp_path / "real-runner-pipeline.sqlite3"
    store = MemoryStore(database)
    workspace = store.create_workspace("owner-1", "real pipeline")
    approved = approval(evaluation={
        "horizon": 6, "min_train": 400, "validation_size": 100,
        "test_size": 80, "step_size": 80,
    })
    result = run_approved_multi_market_research(
        scan(), allocation(), approved, hub=SyntheticHub(),
        trial_ledger=store, owner_id="owner-1", workspace_id=workspace["id"],
    )

    market = result["markets"]["kraken:spot:BTC/USD"]
    assert market["status"] == "evaluated", market
    assert market["report"]["candidate_oos_returns"]
    assert market["report"]["trial_ledger"]["completed_candidate_count"] == 2

    reopened = MemoryStore(database)
    family_id = market["report"]["trial_ledger"]["family_id"]
    family = reopened.get_research_trial_matrix("owner-1", workspace["id"], family_id)
    assert family["attempted_trial_count"] == 2
    assert family["completed_trial_count"] == 2
    assert len(family["observation_times"]) >= 8
    assert len(family["observation_times"]) % 8 == 0
    assert all(len(row) == 2 for row in family["net_return_differentials"])
    assert result["execution_enabled"] is False



def test_trial_ledger_rejects_gaps_between_outer_test_windows(monkeypatch):
    import eti.research_runner as runner

    monkeypatch.setattr(
        runner, "estimate_nested_model_fits_from_rows",
        lambda *args, **kwargs: 4,
    )

    class Ledger:
        def register_research_trial(self, *args):
            raise AssertionError("ledger registration must not begin for gapped folds")

    approved = approval(evaluation={
        "horizon": 6, "min_train": 400, "validation_size": 100,
        "test_size": 80, "step_size": 100,
    })
    with pytest.raises(ValueError, match="step_size must equal test_size"):
        run_approved_multi_market_research(
            scan(), allocation(), approved,
            hub=FakeHub(variable_volume=True),
            trial_ledger=Ledger(), owner_id="owner-1", workspace_id="workspace-1",
        )


def test_market_data_snapshot_fingerprint_is_stable_and_content_sensitive():
    import pandas as pd
    from eti.market_data_hub import _ohlcv_snapshot_sha256

    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
        "open": [100.0, 101.0, 102.0],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
        "close": [100.5, 101.5, 102.5],
        "volume": [10.0, 11.0, 12.0],
    })
    first = _ohlcv_snapshot_sha256(frame)
    assert first == _ohlcv_snapshot_sha256(frame.copy())
    changed = frame.copy()
    changed.loc[1, "close"] += 0.000001
    assert first != _ohlcv_snapshot_sha256(changed)
