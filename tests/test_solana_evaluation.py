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
