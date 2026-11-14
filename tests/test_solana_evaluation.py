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
            {"features": {}, "target_sampled_price_return_pct": 1.0},
        ],
    }])
    assert report["excluded_validation_rows"] == 2
    assert report["aggregate"]["always_long"]["trade_count"] == 1
    with pytest.raises(ValueError):
        evaluate_solana_research_folds([], buy_pressure_threshold=0.5)
