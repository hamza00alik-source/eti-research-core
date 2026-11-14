import math
import pytest

from eti.solana_evaluation import evaluate_solana_research_folds


def _row(outcome, buys, sells, stamp):
    return {
        "mint": "mint-" + stamp,
        "pool_address": "pool-" + stamp,
        "baseline_at": stamp,
        "features": {"txns_h1_buys": buys, "txns_h1_sells": sells},
        "target_sampled_price_return_pct": outcome,
    }


def test_benchmarks_are_fixed_non_fitting_and_validation_only():
    folds = [{
        "fold": 1,
        "train": [{"target_sampled_price_return_pct": 999}],
        "validation": [
            _row(10.0, 8, 2, "2026-01-01T00:00:00Z"),
            _row(-5.0, 2, 8, "2026-01-02T00:00:00Z"),
            _row(4.0, 0, 0, "2026-01-03T00:00:00Z"),
        ],
    }]
    report = evaluate_solana_research_folds(folds)
    assert report["status"] == "BENCHMARK_DIAGNOSTICS_ONLY"
    assert report["model_fitting_performed"] is False
    assert report["training_authorized"] is False
    assert report["aggregate"]["always_long"]["trade_count"] == 3
    assert report["aggregate"]["always_long"]["mean_return_pct"] == pytest.approx(3.0)
    assert report["aggregate"]["fixed_buy_pressure"]["trade_count"] == 1
    assert report["aggregate"]["fixed_buy_pressure"]["mean_return_pct"] == pytest.approx(10.0)
    assert report["aggregate"]["no_trade"]["trade_count"] == 0
    assert "999" not in str(report)


def test_benchmark_excludes_malformed_validation_rows_and_rejects_bad_threshold():
    report = evaluate_solana_research_folds([{
        "validation": [
            _row(2.0, 3, 1, "2026-01-01T00:00:00Z"),
            _row(float("nan"), 3, 1, "2026-01-02T00:00:00Z"),
            _row(-100.1, 3, 1, "2026-01-03T00:00:00Z"),
            {"features": {}, "target_sampled_price_return_pct": 1.0},
        ],
    }])
    assert report["excluded_validation_rows"] == 3
    assert report["aggregate"]["always_long"]["trade_count"] == 1
    with pytest.raises(ValueError):
        evaluate_solana_research_folds([], buy_pressure_threshold=0.5)


def test_walk_forward_runner_joins_only_validation_labels_by_id():
    from eti.solana_evaluation import evaluate_prepared_solana_walk_forward

    exported = []
    prepared = []
    for index in range(8):
        stamp = f"2026-01-{1 + index * 2:02d}T00:00:00Z"
        label_id = f"label-{index}"
        exported.append({
            "id": label_id,
            "mint": f"mint-{index}",
            "pool_address": f"pool-{index}",
            "horizon_seconds": 3600,
            "label": {
                "baseline_at": stamp,
                "target_at": f"2026-01-{1 + index * 2:02d}T01:00:00Z",
            },
        })
        prepared.append({
            "label_id": label_id,
            "mint": f"mint-{index}",
            "pool_address": f"pool-{index}",
            "baseline_at": stamp,
            "horizon_seconds": 3600,
            "features": {"txns_h1_buys": 8, "txns_h1_sells": 2},
            "target_sampled_price_return_pct": 3.0,
        })

    # A second overlapping horizon for the same mint must be visible as dependence.
    exported.append({
        "id": "label-6-longer",
        "mint": "mint-6",
        "pool_address": "pool-6",
        "horizon_seconds": 7200,
        "label": {
            "baseline_at": "2026-01-13T00:00:00Z",
            "target_at": "2026-01-13T02:00:00Z",
        },
    })
    prepared.append({
        "label_id": "label-6-longer",
        "mint": "mint-6",
        "pool_address": "pool-6",
        "baseline_at": "2026-01-13T00:00:00Z",
        "horizon_seconds": 7200,
        "features": {"txns_h1_buys": 8, "txns_h1_sells": 2},
        "target_sampled_price_return_pct": 4.0,
    })

    report = evaluate_prepared_solana_walk_forward(
        exported,
        {"status": "PREPARATION_ONLY", "training_authorized": False, "rows": prepared},
        n_splits=2,
        min_train_groups=3,
        embargo_seconds=3600,
    )
    assert report["status"] == "BENCHMARK_DIAGNOSTICS_ONLY"
    assert report["training_authorized"] is False
    assert report["walk_forward"]["training_rows_used_for_metrics"] is False
    assert "3600" in report["by_horizon_seconds"]
    assert report["by_horizon_seconds"]["3600"]["aggregate"]["always_long"]["trade_count"] > 0
    assert report["validation_dependence"]["repeated_mint_label_count"] >= 1
    assert report["validation_dependence"]["overlapping_interval_pair_count"] >= 1
    assert "cross_mint_overlapping_interval_pair_count" in report["validation_dependence"]
    assert "distinct_mints_repeated_across_validation_folds" in report["validation_dependence"]
    assert "7200" in report["by_horizon_seconds"]
    assert len(report["walk_forward"]["fold_metadata"]) == 2
    assert all(
        item["prepared_validation_count"] + item["unprepared_validation_count"]
        == item["validation_label_count"]
        for item in report["walk_forward"]["fold_metadata"]
    )
    assert report["aggregate"]["always_long"]["trade_count"] == sum(
        item["prepared_validation_count"] for item in report["walk_forward"]["fold_metadata"]
    )


def test_walk_forward_runner_rejects_duplicate_prepared_ids_and_non_preparation_input():
    from eti.solana_evaluation import evaluate_prepared_solana_walk_forward

    duplicate = {"label_id": "x", "features": {}, "target_sampled_price_return_pct": 1}
    with pytest.raises(ValueError):
        evaluate_prepared_solana_walk_forward(
            [], {"status": "PREPARATION_ONLY", "training_authorized": False, "rows": [duplicate, duplicate]}
        )
    with pytest.raises(ValueError):
        evaluate_prepared_solana_walk_forward(
            [], {"status": "OTHER", "training_authorized": True, "rows": []}
        )


def test_walk_forward_cost_evidence_is_applied_only_to_validation_rows():
    from eti.solana_evaluation import evaluate_prepared_solana_walk_forward

    exported, prepared, evidence_by_id = [], [], {}
    for index in range(8):
        day = 1 + index * 2
        stamp = f"2026-01-{day:02d}T00:00:00Z"
        label_id = f"cost-label-{index}"
        exported.append({
            "id": label_id,
            "mint": f"cost-mint-{index}",
            "pool_address": f"cost-pool-{index}",
            "horizon_seconds": 3600,
            "label": {
                "baseline_at": stamp,
                "target_at": f"2026-01-{day:02d}T01:00:00Z",
            },
        })
        prepared.append({
            "label_id": label_id,
            "mint": f"cost-mint-{index}",
            "pool_address": f"cost-pool-{index}",
            "baseline_at": stamp,
            "horizon_seconds": 3600,
            "features": {"txns_h1_buys": 8, "txns_h1_sells": 2},
            "target_sampled_price_return_pct": 5.0,
        })
        evidence_by_id[label_id] = {
            "label_id": label_id,
            "mint": f"cost-mint-{index}",
            "pool_address": f"cost-pool-{index}",
            "cost_components": {
                name: {
                    "cost_bps": 10,
                    "unit": "bps_of_entry_notional",
                    "measurement": "measured",
                    "source_id": f"{label_id}-{name}",
                    "observed_at": "2026-11-14T12:00:00Z",
                }
                for name in (
                    "spread", "pool_price_impact", "protocol_fee",
                    "priority_fee", "latency_slippage",
                )
            },
            "exit_evidence": {
                "status": "exit_observed",
                "source_id": f"{label_id}-exit",
                "transaction_id": f"{label_id}-tx",
                "observed_at": "2026-11-14T12:01:00Z",
            },
        }

    report = evaluate_prepared_solana_walk_forward(
        exported,
        {"status": "PREPARATION_ONLY", "training_authorized": False, "rows": prepared},
        n_splits=2,
        min_train_groups=3,
        embargo_seconds=3600,
        execution_cost_evidence_by_label_id=evidence_by_id,
    )
    costs = report["execution_cost_evidence"]
    assert costs["status"] == "COST_EVIDENCE_DIAGNOSTICS_ONLY"
    assert costs["validation_rows_checked"] > 0
    assert costs["complete_evidence_count"] == costs["validation_rows_checked"]
    assert costs["incomplete_evidence_count"] == 0
    assert costs["cost_adjusted_sampled_mark_scenario_mean_pct"] == pytest.approx(4.5)
    assert costs["net_return_pct"] is None
    assert costs["realized_pnl_available"] is False
    assert all(row["label_id"] in {item["id"] for item in exported} for row in costs["rows"])


def test_walk_forward_reports_missing_cost_evidence_as_incomplete_not_zero():
    from eti.solana_evaluation import evaluate_prepared_solana_walk_forward

    exported, prepared = [], []
    for index in range(8):
        day = 1 + index * 2
        stamp = f"2026-01-{day:02d}T00:00:00Z"
        label_id = f"missing-cost-label-{index}"
        exported.append({
            "id": label_id,
            "mint": f"missing-cost-mint-{index}",
            "pool_address": f"missing-cost-pool-{index}",
            "horizon_seconds": 3600,
            "label": {"baseline_at": stamp, "target_at": f"2026-01-{day:02d}T01:00:00Z"},
        })
        prepared.append({
            "label_id": label_id,
            "mint": f"missing-cost-mint-{index}",
            "pool_address": f"missing-cost-pool-{index}",
            "baseline_at": stamp,
            "horizon_seconds": 3600,
            "features": {"txns_h1_buys": 8, "txns_h1_sells": 2},
            "target_sampled_price_return_pct": 5.0,
        })

    report = evaluate_prepared_solana_walk_forward(
        exported,
        {"status": "PREPARATION_ONLY", "training_authorized": False, "rows": prepared},
        n_splits=2,
        min_train_groups=3,
        embargo_seconds=3600,
        execution_cost_evidence_by_label_id={},
    )
    costs = report["execution_cost_evidence"]
    assert costs["validation_rows_checked"] > 0
    assert costs["complete_evidence_count"] == 0
    assert costs["incomplete_evidence_count"] == costs["validation_rows_checked"]
    assert costs["cost_adjusted_sampled_mark_scenario_mean_pct"] is None
    assert costs["net_return_pct"] is None


def test_walk_forward_cost_evidence_rejects_cross_label_identity_reuse():
    from eti.solana_evaluation import evaluate_prepared_solana_walk_forward

    exported, prepared, evidence_by_id = [], [], {}
    for index in range(8):
        day = 1 + index * 2
        stamp = f"2026-01-{day:02d}T00:00:00Z"
        label_id = f"identity-label-{index}"
        exported.append({
            "id": label_id, "mint": f"identity-mint-{index}",
            "pool_address": f"identity-pool-{index}", "horizon_seconds": 3600,
            "label": {"baseline_at": stamp, "target_at": f"2026-01-{day:02d}T01:00:00Z"},
        })
        prepared.append({
            "label_id": label_id, "mint": f"identity-mint-{index}",
            "pool_address": f"identity-pool-{index}", "horizon_seconds": 3600,
            "features": {"txns_h1_buys": 8, "txns_h1_sells": 2},
            "target_sampled_price_return_pct": 5.0,
        })
        evidence_by_id[label_id] = {
            "label_id": label_id, "mint": "some-other-mint",
            "pool_address": f"identity-pool-{index}",
            "cost_components": {}, "exit_evidence": {},
        }

    report = evaluate_prepared_solana_walk_forward(
        exported,
        {"status": "PREPARATION_ONLY", "training_authorized": False, "rows": prepared},
        n_splits=2, min_train_groups=3, embargo_seconds=3600,
        execution_cost_evidence_by_label_id=evidence_by_id,
    )
    costs = report["execution_cost_evidence"]
    assert costs["complete_evidence_count"] == 0
    assert costs["incomplete_evidence_count"] == costs["validation_rows_checked"]
    assert all("evidence_identity_mismatch" in row["invalid_components"] for row in costs["rows"])


def test_walk_forward_rejects_prepared_row_mint_or_pool_identity_mismatch():
    from eti.solana_evaluation import evaluate_prepared_solana_walk_forward

    exported, prepared = [], []
    for index in range(8):
        day = 1 + index * 2
        stamp = f"2026-01-{day:02d}T00:00:00Z"
        label_id = f"prepared-identity-label-{index}"
        exported.append({
            "id": label_id, "mint": f"prepared-mint-{index}",
            "pool_address": f"prepared-pool-{index}", "horizon_seconds": 3600,
            "label": {"baseline_at": stamp, "target_at": f"2026-01-{day:02d}T01:00:00Z"},
        })
        prepared.append({
            "label_id": label_id, "mint": "wrong-mint",
            "pool_address": f"prepared-pool-{index}", "horizon_seconds": 3600,
            "features": {"txns_h1_buys": 8, "txns_h1_sells": 2},
            "target_sampled_price_return_pct": 5.0,
        })
    with pytest.raises(ValueError, match="identity"):
        evaluate_prepared_solana_walk_forward(
            exported,
            {"status": "PREPARATION_ONLY", "training_authorized": False, "rows": prepared},
            n_splits=2, min_train_groups=3, embargo_seconds=3600,
        )


def test_walk_forward_transaction_receipts_are_integrity_checked_and_never_verified_swaps():
    import hashlib
    import json
    from eti.solana_evaluation import evaluate_prepared_solana_walk_forward

    exported, prepared, attestations = [], [], {}
    for index in range(8):
        day = 1 + index * 2
        stamp = f"2026-01-{day:02d}T00:00:00Z"
        label_id = f"receipt-label-{index}"
        mint = f"receipt-mint-{index}"
        pool = f"receipt-pool-{index}"
        exported.append({
            "id": label_id, "mint": mint, "pool_address": pool, "horizon_seconds": 3600,
            "label": {"baseline_at": stamp, "target_at": f"2026-01-{day:02d}T01:00:00Z"},
        })
        prepared.append({
            "label_id": label_id, "mint": mint, "pool_address": pool,
            "baseline_at": stamp, "horizon_seconds": 3600,
            "features": {"txns_h1_buys": 8, "txns_h1_sells": 2},
            "target_sampled_price_return_pct": 2.0,
        })
        attestation = {
            "status": "FINALIZED_MINT_BALANCE_DEBIT_OBSERVED",
            "verified": False, "transaction_receipt_verified": True,
            "signature": f"signature-{index}", "mint": mint,
            "expected_pool_address": pool, "wallet_address": f"wallet-{index}",
        }
        encoded = json.dumps(attestation, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        attestations[label_id] = [{
            "id": f"attestation-{index}", "label_id": label_id, "signature": f"signature-{index}",
            "mint": mint, "pool_address": pool, "wallet_address": f"wallet-{index}",
            "attestation": attestation,
            "attestation_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "integrity_valid": True,
        }]

    report = evaluate_prepared_solana_walk_forward(
        exported,
        {"status": "PREPARATION_ONLY", "training_authorized": False, "rows": prepared},
        n_splits=2, min_train_groups=3, embargo_seconds=3600,
        transaction_attestations_by_label_id=attestations,
    )
    receipt = report["transaction_receipt_attestations"]
    assert receipt["status"] == "RECEIPT_OBSERVATION_DIAGNOSTICS_ONLY"
    assert receipt["validation_rows_checked"] > 0
    assert receipt["attestation_record_count"] == receipt["validation_rows_checked"]
    assert receipt["integrity_valid_record_count"] == receipt["finalized_receipt_observation_count"]
    assert receipt["verified_swap_count"] == 0
    assert receipt["net_return_pct"] is None
    assert receipt["realized_pnl_available"] is False


def test_pooled_compounding_and_dependence_are_explicitly_non_portfolio_diagnostics():
    from eti.solana_evaluation import evaluate_prepared_solana_walk_forward

    exported, prepared = [], []
    for index in range(8):
        day = 1 + index * 2
        stamp = f"2026-01-{day:02d}T00:00:00Z"
        label_id = f"interpretation-label-{index}"
        exported.append({
            "id": label_id,
            "mint": f"interpretation-mint-{index}",
            "pool_address": f"interpretation-pool-{index}",
            "horizon_seconds": 3600,
            "label": {"baseline_at": stamp, "target_at": f"2026-01-{day:02d}T01:00:00Z"},
        })
        prepared.append({
            "label_id": label_id,
            "mint": f"interpretation-mint-{index}",
            "pool_address": f"interpretation-pool-{index}",
            "baseline_at": stamp,
            "horizon_seconds": 3600,
            "features": {"txns_h1_buys": 8, "txns_h1_sells": 2},
            "target_sampled_price_return_pct": 2.0,
        })

    report = evaluate_prepared_solana_walk_forward(
        exported,
        {"status": "PREPARATION_ONLY", "training_authorized": False, "rows": prepared},
        n_splits=2, min_train_groups=3, embargo_seconds=3600,
    )
    summary = report["aggregate"]["always_long"]
    assert summary["compounding_interpretation"] == "POOLED_SAMPLED_OBSERVATION_DIAGNOSTIC_NOT_PORTFOLIO_PNL"
    assert "not a portfolio equity curve" in report["aggregate_metric_interpretation"]
    dependence = report["validation_dependence"]
    assert dependence["fold_validation_label_count"] >= dependence["prepared_validation_interval_count"]
    assert dependence["prepared_validation_interval_count"] == dependence["validation_label_count"]
    assert dependence["unprepared_validation_label_count"] == 0
    assert dependence["dependence_scope"] == "PREPARED_VALIDATION_ROWS_WITH_VALID_INTERVALS_ONLY"
    assert report["training_authorized"] is False


def test_mint_cluster_bootstrap_withholds_small_cluster_counts_and_is_reproducible():
    from eti.solana_evaluation import evaluate_solana_research_folds

    small = evaluate_solana_research_folds([{
        "validation": [
            _row(1.0, 8, 2, f"2026-01-{index + 1:02d}T00:00:00Z")
            for index in range(5)
        ]
    }])
    assert small["cluster_uncertainty"]["always_long"]["status"] == "INSUFFICIENT_MINT_CLUSTERS"
    assert small["cluster_uncertainty"]["always_long"]["mean_return_pct_interval"] is None

    rows = []
    for index in range(25):
        rows.append({
            "mint": f"cluster-mint-{index}",
            "pool_address": f"cluster-pool-{index}",
            "baseline_at": f"2026-02-{index + 1:02d}T00:00:00Z",
            "features": {"txns_h1_buys": 8, "txns_h1_sells": 2},
            "target_sampled_price_return_pct": float(index - 10),
        })
    report_a = evaluate_solana_research_folds([{"validation": rows}])
    report_b = evaluate_solana_research_folds([{"validation": rows}])
    interval = report_a["cluster_uncertainty"]["always_long"]
    assert interval["status"] == "EXPLORATORY_INTERVAL_AVAILABLE"
    assert interval["cluster_count"] == 25
    assert interval["mean_return_pct_interval"] == report_b["cluster_uncertainty"]["always_long"]["mean_return_pct_interval"]
    low, high = interval["mean_return_pct_interval"]
    assert low <= report_a["aggregate"]["always_long"]["mean_return_pct"] <= high
    assert report_a["training_authorized"] is False


def test_missing_mint_rows_are_not_treated_as_independent_bootstrap_clusters():
    from eti.solana_evaluation import evaluate_solana_research_folds

    rows = [
        {
            "pool_address": f"pool-{index}",
            "baseline_at": f"2026-03-{index + 1:02d}T00:00:00Z",
            "features": {"txns_h1_buys": 8, "txns_h1_sells": 2},
            "target_sampled_price_return_pct": 1.0,
        }
        for index in range(25)
    ]
    report = evaluate_solana_research_folds([{"validation": rows}])
    interval = report["cluster_uncertainty"]["always_long"]
    assert interval["cluster_count"] == 1
    assert interval["status"] == "INSUFFICIENT_MINT_CLUSTERS"
    assert interval["mean_return_pct_interval"] is None



def test_regime_diagnostics_require_predeclared_thresholds_and_label_only_snapshot_proxies():
    from eti.solana_evaluation import evaluate_solana_regime_diagnostics

    rows = [{
        "features": {
            "txns_h1_buys": 8, "txns_h1_sells": 2,
            "volume_h24_usd": 5000, "liquidity_usd": 1000,
        },
        "target_sampled_price_return_pct": 4.0,
    }, {
        "features": {
            "txns_h1_buys": 1, "txns_h1_sells": 9,
            "volume_h24_usd": 100, "liquidity_usd": 1000,
        },
        "target_sampled_price_return_pct": -3.0,
    }]
    unconfigured = evaluate_solana_regime_diagnostics(rows)
    assert unconfigured["status"] == "NOT_CONFIGURED"
    assert unconfigured["classified_row_count"] == 0

    report = evaluate_solana_regime_diagnostics(rows, {
        "buy_pressure_high": 0.65,
        "buy_pressure_low": 0.35,
        "turnover_liquidity_ratio_high": 2.0,
        "turnover_liquidity_ratio_low": 0.5,
    })
    assert report["status"] == "BASELINE_PROXY_REGIME_DIAGNOSTICS_ONLY"
    assert report["classified_row_count"] == 2
    assert "BUY_DOMINANT__HIGH_TURNOVER_PROXY" in report["regimes"]
    assert "SELL_DOMINANT__LOW_TURNOVER_PROXY" in report["regimes"]
    assert "not trend" in report["interpretation"]


def test_regime_diagnostics_reject_invalid_or_unapproved_threshold_shapes():
    from eti.solana_evaluation import evaluate_solana_regime_diagnostics

    with pytest.raises(ValueError):
        evaluate_solana_regime_diagnostics([], {"buy_pressure_high": 0.6})
    with pytest.raises(ValueError):
        evaluate_solana_regime_diagnostics([], {
            "buy_pressure_high": 0.3,
            "buy_pressure_low": 0.4,
            "turnover_liquidity_ratio_high": 2.0,
            "turnover_liquidity_ratio_low": 0.5,
        })



def test_time_series_regimes_use_only_prior_same_mint_pool_prices():
    from datetime import datetime, timedelta, timezone
    from eti.solana_evaluation import evaluate_solana_time_series_regimes

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = []
    for index in range(12):
        history.append({
            "mint": "mint-a",
            "pool_address": "pool-a",
            "baseline_at": (start + timedelta(hours=index)).isoformat(),
            "features": {"price_usd": 100.0 * (1.01 ** index)},
        })
    validation = [{
        "mint": "mint-a",
        "pool_address": "pool-a",
        "baseline_at": (start + timedelta(hours=11)).isoformat(),
        "target_sampled_price_return_pct": 2.5,
    }]
    config = {
        "lookback_observations": 10,
        "minimum_observations": 10,
        "trend_abs_return_pct": 3.0,
        "volatility_high_pct": 2.0,
        "volatility_low_pct": 0.1,
        "max_gap_seconds": 7200,
        "minimum_history_span_seconds": 3600,
        "minimum_rows_per_regime_horizon": 5,
        "buy_pressure_threshold": 0.55,
    }
    report = evaluate_solana_time_series_regimes(history, validation, config)
    assert report["status"] == "TIME_SERIES_PROXY_REGIME_DIAGNOSTICS_ONLY"
    assert report["classified_row_count"] == 1
    assert len(report["classified_observations"]) == 1
    assert report["regime_transitions"]["training_authorized"] is False
    assert "UPWARD_TREND_PROXY__LOW_TIME_NORMALIZED_VOLATILITY_PROXY" in report["regimes"]

    # A future price observation and another pool must not affect this label.
    polluted_history = history + [{
        "mint": "mint-a",
        "pool_address": "pool-a",
        "baseline_at": (start + timedelta(hours=99)).isoformat(),
        "features": {"price_usd": 1_000_000.0},
    }, {
        "mint": "mint-a",
        "pool_address": "other-pool",
        "baseline_at": (start + timedelta(hours=12)).isoformat(),
        "features": {"price_usd": 1_000_000.0},
    }]
    future_safe = evaluate_solana_time_series_regimes(polluted_history, validation, config)
    assert future_safe["regimes"] == report["regimes"]


def test_time_series_regimes_withhold_sparse_and_conflicting_history():
    from datetime import datetime, timedelta, timezone
    from eti.solana_evaluation import evaluate_solana_time_series_regimes

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    sparse = [{
        "mint": "mint-a", "pool_address": "pool-a",
        "baseline_at": (start + timedelta(hours=i)).isoformat(),
        "features": {"price_usd": 100.0 + i},
    } for i in range(4)]
    validation = [{
        "mint": "mint-a", "pool_address": "pool-a",
        "baseline_at": (start + timedelta(hours=3)).isoformat(),
        "target_sampled_price_return_pct": 1.0,
    }]
    config = {
        "lookback_observations": 10, "minimum_observations": 5,
        "trend_abs_return_pct": 2.0, "volatility_high_pct": 2.0,
        "volatility_low_pct": 0.2,
        "max_gap_seconds": 7200,
        "minimum_history_span_seconds": 3600,
        "minimum_rows_per_regime_horizon": 5,
        "buy_pressure_threshold": 0.55,
    }
    report = evaluate_solana_time_series_regimes(sparse, validation, config)
    assert report["classified_row_count"] == 0
    assert report["unclassified_reasons"]["insufficient_prior_baseline_observations"] == 1

    conflicting = sparse + [{
        **sparse[-1], "features": {"price_usd": 999.0},
    }]
    report = evaluate_solana_time_series_regimes(conflicting, validation, config)
    assert report["classified_row_count"] == 0
    assert report["unclassified_reasons"]["conflicting_prices_at_same_timestamp"] == 1


def test_time_series_regime_config_is_strict_and_training_stays_locked():
    from eti.solana_evaluation import evaluate_solana_time_series_regimes

    with pytest.raises(ValueError):
        evaluate_solana_time_series_regimes([], [], {"lookback_observations": 5})
    with pytest.raises(ValueError):
        evaluate_solana_time_series_regimes([], [], {
            "lookback_observations": 3, "minimum_observations": 5,
            "trend_abs_return_pct": 1.0, "volatility_high_pct": 2.0,
            "volatility_low_pct": 0.5,
        })


def test_time_series_regimes_normalize_volatility_and_separate_horizons():
    from datetime import datetime, timedelta, timezone
    from eti.solana_evaluation import evaluate_solana_time_series_regimes

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # Equal 1% log-return increments over unequal intervals should be
    # normalized by elapsed time before volatility is calculated.
    history = []
    intervals = [3600, 7200, 3600, 7200, 3600]
    stamp = start
    price = 100.0
    for index in range(6):
        history.append({
            "mint": "mint-a", "pool_address": "pool-a",
            "baseline_at": stamp.isoformat(), "features": {"price_usd": price},
        })
        if index < len(intervals):
            stamp += timedelta(seconds=intervals[index])
            price *= math.exp(0.01 * math.sqrt(intervals[index] / 3600.0))
    validation = [
        {"mint": "mint-a", "pool_address": "pool-a",
         "baseline_at": stamp.isoformat(), "horizon_seconds": horizon,
         "features": {"txns_h1_buys": 6, "txns_h1_sells": 4},
         "target_sampled_price_return_pct": float(horizon) / 1000}
        for horizon in (60, 300)
    ]
    config = {
        "lookback_observations": 6, "minimum_observations": 6,
        "trend_abs_return_pct": 1.0, "volatility_high_pct": 1.0,
        "volatility_low_pct": 0.001, "max_gap_seconds": 7200,
        "minimum_history_span_seconds": 3600,
        "minimum_rows_per_regime_horizon": 5,
        "buy_pressure_threshold": 0.55,
    }
    report = evaluate_solana_time_series_regimes(history, validation, config)
    assert report["classified_row_count"] == 2
    assert set(report["regimes_by_horizon_seconds"]) == {"60", "300"}
    stratum = next(iter(report["regimes_by_horizon_seconds"]["60"].values()))
    assert stratum["status"] == "INSUFFICIENT_REGIME_HORIZON_ROWS"
    assert stratum["summary"] is None
    assert stratum["mint_cluster_bootstrap"]["status"] == "INSUFFICIENT_MINT_CLUSTERS"
    benchmark = stratum["fixed_benchmark_comparison"]
    assert benchmark["model_fitting_performed"] is False
    assert benchmark["buy_pressure_rule"]["fit_on_validation_data"] is False
    assert benchmark["aggregate"]["always_long"]["trade_count"] == 1
    assert benchmark["aggregate"]["fixed_buy_pressure"]["trade_count"] == 1
    assert report["sampling_quality"]["mean_max_gap_seconds"] == 7200
    assert report["method"]["volatility_unit"] == "percent_per_sqrt_hour"


def test_time_series_regimes_reject_excessive_sampling_gaps():
    from datetime import datetime, timedelta, timezone
    from eti.solana_evaluation import evaluate_solana_time_series_regimes

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = [{
        "mint": "mint-a", "pool_address": "pool-a",
        "baseline_at": (start + timedelta(hours=i * 5)).isoformat(),
        "features": {"price_usd": 100.0 + i},
    } for i in range(5)]
    validation = [{
        "mint": "mint-a", "pool_address": "pool-a",
        "baseline_at": (start + timedelta(hours=20)).isoformat(),
        "target_sampled_price_return_pct": 1.0,
    }]
    config = {
        "lookback_observations": 5, "minimum_observations": 5,
        "trend_abs_return_pct": 1.0, "volatility_high_pct": 2.0,
        "volatility_low_pct": 0.2, "max_gap_seconds": 3600,
        "minimum_history_span_seconds": 3600,
        "minimum_rows_per_regime_horizon": 5,
        "buy_pressure_threshold": 0.55,
    }
    report = evaluate_solana_time_series_regimes(history, validation, config)
    assert report["classified_row_count"] == 0
    assert report["unclassified_reasons"]["sampling_gap_exceeds_configured_maximum"] == 1


def test_time_series_regime_horizon_summary_requires_predeclared_row_count():
    from datetime import datetime, timedelta, timezone
    from eti.solana_evaluation import evaluate_solana_time_series_regimes

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = []
    for index in range(3):
        history.append({
            "mint": "mint-a", "pool_address": "pool-a",
            "baseline_at": (start + timedelta(hours=index)).isoformat(),
            "features": {"price_usd": 100.0 + index},
        })
    validation = [{
        "mint": "mint-a", "pool_address": "pool-a",
        "baseline_at": (start + timedelta(hours=3)).isoformat(),
        "horizon_seconds": 60, "target_sampled_price_return_pct": float(i),
        "features": {"txns_h1_buys": 8, "txns_h1_sells": 2},
    } for i in range(5)]
    config = {
        "lookback_observations": 3, "minimum_observations": 3,
        "trend_abs_return_pct": 1.0, "volatility_high_pct": 2.0,
        "volatility_low_pct": 0.2, "max_gap_seconds": 3600,
        "minimum_history_span_seconds": 3600,
        "minimum_rows_per_regime_horizon": 5,
        "buy_pressure_threshold": 0.55,
    }
    report = evaluate_solana_time_series_regimes(history, validation, config)
    stratum = report["regimes_by_horizon_seconds"]["60"]["UPWARD_TREND_PROXY__LOW_TIME_NORMALIZED_VOLATILITY_PROXY"]
    assert stratum["row_count"] == 5
    assert stratum["status"] == "DESCRIPTIVE_SAMPLE_THRESHOLD_MET"
    assert stratum["summary"]["trade_count"] == 5
    assert stratum["distinct_mint_count"] == 1
    assert stratum["mint_cluster_bootstrap"]["status"] == "INSUFFICIENT_MINT_CLUSTERS"


def test_fold_regime_stability_rejects_mint_concentration_and_reports_sign_consistency():
    from eti.solana_evaluation import evaluate_solana_fold_regime_stability

    thresholds = {
        "buy_pressure_high": 0.7,
        "buy_pressure_low": 0.3,
        "turnover_liquidity_ratio_high": 5.0,
        "turnover_liquidity_ratio_low": 1.0,
    }

    def row(mint, outcome):
        return {
            "mint": mint,
            "target_sampled_price_return_pct": outcome,
            "features": {
                "txns_h1_buys": 8, "txns_h1_sells": 2,
                "volume_h24_usd": 100, "liquidity_usd": 10,
            },
        }

    report = evaluate_solana_fold_regime_stability(
        [
            {"fold": 1, "validation": [
                row("mint-a", 2.0), row("mint-a", 1.0),
                row("mint-b", 3.0), row("mint-b", 2.0),
                row("mint-c", 1.0),
            ]},
            {"fold": 2, "validation": [
                row("mint-z", -2.0), row("mint-z", -1.0),
                row("mint-z", -3.0), row("mint-z", -2.0),
                row("mint-z", -1.0),
            ]},
        ],
        thresholds,
    )
    regime = "BUY_DOMINANT__HIGH_TURNOVER_PROXY"
    first = report["folds"][0]["regimes"][regime]
    second = report["folds"][1]["regimes"][regime]
    assert first["stability_eligible"] is True
    assert first["distinct_mint_count"] == 3
    assert first["largest_mint_row_share"] == 0.4
    assert second["stability_eligible"] is False
    assert second["distinct_mint_count"] == 1
    stability = report["regimes"][regime]["always_long"]
    assert stability["eligible_fold_count"] == 1
    assert stability["positive_mean_fold_count"] == 1
    assert stability["negative_mean_fold_count"] == 0
    assert stability["mean_return_sign_consistency"] == 1.0
    assert report["training_authorized"] is False


def test_fold_regime_stability_is_not_configured_without_predeclared_thresholds():
    from eti.solana_evaluation import evaluate_solana_fold_regime_stability

    report = evaluate_solana_fold_regime_stability(
        [{"fold": 1, "validation": []}], None
    )
    assert report["status"] == "NOT_CONFIGURED"
    assert report["regimes"] == {}


def test_fold_time_series_regime_stability_uses_point_in_time_history_and_mint_gates():
    from datetime import datetime, timedelta, timezone
    from eti.solana_evaluation import evaluate_solana_fold_time_series_regime_stability

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = []
    for mint in ("mint-a", "mint-b", "mint-c", "mint-z"):
        for index, price in enumerate((100.0, 101.0, 102.0)):
            history.append({
                "mint": mint,
                "pool_address": "pool-" + mint,
                "baseline_at": (start + timedelta(hours=index)).isoformat(),
                "features": {"price_usd": price},
            })

    def validation(mint, outcome):
        return {
            "mint": mint,
            "pool_address": "pool-" + mint,
            "baseline_at": (start + timedelta(hours=2)).isoformat(),
            "horizon_seconds": 60,
            "target_sampled_price_return_pct": outcome,
            "features": {"txns_h1_buys": 8, "txns_h1_sells": 2},
        }

    config = {
        "lookback_observations": 3,
        "minimum_observations": 3,
        "trend_abs_return_pct": 1.0,
        "volatility_high_pct": 2.0,
        "volatility_low_pct": 0.1,
        "max_gap_seconds": 3600,
        "minimum_history_span_seconds": 3600,
        "minimum_rows_per_regime_horizon": 5,
        "buy_pressure_threshold": 0.55,
    }
    folds = [
        {"fold": 1, "validation": [
            validation("mint-a", 2.0), validation("mint-a", 1.0),
            validation("mint-b", 3.0), validation("mint-b", 2.0),
            validation("mint-c", 1.0), validation("mint-c", 2.0),
        ]},
        {"fold": 2, "validation": [
            validation("mint-z", -2.0) for _ in range(5)
        ]},
    ]
    report = evaluate_solana_fold_time_series_regime_stability(
        history, folds, config
    )
    regime = "UPWARD_TREND_PROXY__LOW_TIME_NORMALIZED_VOLATILITY_PROXY"
    first = report["folds"][0]["regimes_by_horizon_seconds"]["60"][regime]
    second = report["folds"][1]["regimes_by_horizon_seconds"]["60"][regime]
    assert first["stability_eligible"] is True
    assert first["row_count"] == 6
    assert first["distinct_mint_count"] == 3
    assert second["stability_eligible"] is False
    assert second["distinct_mint_count"] == 1
    stability = report["regimes_by_horizon_seconds"]["60"][regime]["always_long"]
    assert stability["eligible_fold_count"] == 1
    assert stability["positive_mean_fold_count"] == 1
    assert stability["mean_return_sign_consistency"] == 1.0
    assert report["future_history_excluded_per_validation_row"] is True
    assert report["training_authorized"] is False


def test_fold_time_series_regime_stability_requires_explicit_config():
    from eti.solana_evaluation import evaluate_solana_fold_time_series_regime_stability

    report = evaluate_solana_fold_time_series_regime_stability(
        [], [{"fold": 1, "validation": []}], None
    )
    assert report["status"] == "NOT_CONFIGURED"
    assert report["regimes_by_horizon_seconds"] == {}



def test_regime_transitions_are_mint_pool_horizon_local_and_fail_closed():
    from eti.solana_evaluation import evaluate_solana_regime_transitions

    observations = []
    for mint in ("mint-a", "mint-b", "mint-c"):
        observations.extend([
            {"mint": mint, "pool_address": "pool-" + mint, "horizon_seconds": "60",
             "baseline_at": "2026-01-01T00:00:00+00:00", "regime": "UP"},
            {"mint": mint, "pool_address": "pool-" + mint, "horizon_seconds": "60",
             "baseline_at": "2026-01-01T00:01:00+00:00", "regime": "DOWN"},
            {"mint": mint, "pool_address": "pool-" + mint, "horizon_seconds": "60",
             "baseline_at": "2026-01-01T00:02:00+00:00", "regime": "UP"},
        ])
    # Same timestamp with contradictory assignments is excluded, not ordered.
    observations.extend([
        {"mint": "mint-conflict", "pool_address": "pool-conflict", "horizon_seconds": "60",
         "baseline_at": "2026-01-01T00:00:00+00:00", "regime": "UP"},
        {"mint": "mint-conflict", "pool_address": "pool-conflict", "horizon_seconds": "60",
         "baseline_at": "2026-01-01T00:00:00+00:00", "regime": "DOWN"},
        {"mint": "mint-conflict", "pool_address": "pool-conflict", "horizon_seconds": "60",
         "baseline_at": "2026-01-01T00:01:00+00:00", "regime": "UP"},
    ])
    report = evaluate_solana_regime_transitions(
        observations, minimum_transitions=3, minimum_mints=3
    )
    assert report["total_transition_count"] == 6
    assert report["distinct_mint_count_with_transitions"] == 3
    assert report["conflicting_timestamp_count"] == 1
    assert report["training_authorized"] is False
    up_down = report["transitions_by_horizon_seconds"]["60"]["UP -> DOWN"]
    assert up_down["transition_count"] == 3
    assert up_down["distinct_mint_count"] == 3
    assert up_down["status"] == "DESCRIPTIVE_TRANSITION_THRESHOLD_MET"

    with pytest.raises(ValueError):
        evaluate_solana_regime_transitions([], minimum_transitions=0)



def test_regime_sampling_sensitivity_matches_only_identical_point_in_time_rows():
    from eti.solana_evaluation import evaluate_solana_regime_sampling_sensitivity

    row = {"mint": "mint-a", "pool_address": "pool-a",
           "horizon_seconds": "60", "baseline_at": "2026-01-01T00:00:00+00:00",
           "regime": "UP"}
    changed = dict(row, regime="DOWN")
    other = dict(row, mint="mint-b")
    report = evaluate_solana_regime_sampling_sensitivity({
        "tight": {"classified_observations": [row, other]},
        "wide": {"classified_observations": [changed]},
    })
    comparison = report["pairwise_comparisons"]["tight vs wide"]
    assert comparison["matched_observation_count"] == 1
    assert comparison["matching_regime_count"] == 0
    assert comparison["different_regime_count"] == 1
    assert comparison["left_only_observation_count"] == 1
    assert comparison["right_only_observation_count"] == 0
    assert report["all_configuration_unanimous_regime_count"] == 0
    assert report["training_authorized"] is False

    with pytest.raises(ValueError):
        evaluate_solana_regime_sampling_sensitivity({"only": {"classified_observations": []}})



def test_regime_persistence_censors_gaps_conflicts_and_sample_end():
    from eti.solana_evaluation import evaluate_solana_regime_persistence

    observations = []
    for mint in ("mint-a", "mint-b", "mint-c"):
        observations.extend([
            {"mint": mint, "pool_address": "pool-" + mint, "horizon_seconds": "60",
             "baseline_at": "2026-01-01T00:00:00+00:00", "regime": "UP"},
            {"mint": mint, "pool_address": "pool-" + mint, "horizon_seconds": "60",
             "baseline_at": "2026-01-01T00:00:30+00:00", "regime": "UP"},
            {"mint": mint, "pool_address": "pool-" + mint, "horizon_seconds": "60",
             "baseline_at": "2026-01-01T00:01:00+00:00", "regime": "DOWN"},
            {"mint": mint, "pool_address": "pool-" + mint, "horizon_seconds": "60",
             "baseline_at": "2026-01-01T00:05:00+00:00", "regime": "DOWN"},
        ])
    observations.extend([
        {"mint": "mint-gap", "pool_address": "pool-gap", "horizon_seconds": "60",
         "baseline_at": "2026-01-01T00:00:00+00:00", "regime": "UP"},
        {"mint": "mint-gap", "pool_address": "pool-gap", "horizon_seconds": "60",
         "baseline_at": "2026-01-01T00:10:00+00:00", "regime": "UP"},
    ])
    report = evaluate_solana_regime_persistence(
        observations, maximum_gap_seconds=60, minimum_episodes=3, minimum_mints=3
    )
    up = report["persistence_by_horizon_seconds"]["60"]["UP"]
    down = report["persistence_by_horizon_seconds"]["60"]["DOWN"]
    assert up["episode_count"] == 6
    assert up["distinct_mint_count"] == 4
    assert up["median_observed_span_seconds"] == 30
    assert up["right_censored_episode_count"] == 4
    assert down["episode_count"] == 6
    assert down["median_observed_span_seconds"] == 0
    assert report["training_authorized"] is False

    with pytest.raises(ValueError):
        evaluate_solana_regime_persistence([], maximum_gap_seconds=0)
