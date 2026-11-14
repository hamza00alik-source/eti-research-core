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
