import math

import pytest

from eti.portfolio_quality import build_portfolio_quality_report


def market_result(offset=0):
    folds = []
    for fold_id in range(2):
        timestamps = [
            f"2025-01-01T0{hour + offset}:00:00+00:00"
            for hour in range(1, 4)
        ]
        folds.append({
            "fold": fold_id,
            "timestamps": timestamps,
            "trading": {
                "model": {
                    "bar_returns": [0.10, -0.10, 0.05],
                    "positions": [1.0, 1.0, 0.0],
                }
            },
        })
    return {"status": "evaluated", "report": {
        "timeframe": "1h",
        "trading_performance": {"folds": folds},
    }}


def source():
    return {
        "schema_version": "eti.multi_market_research_result.v1",
        "training_performed": True,
        "execution_enabled": False,
        "markets": {
            "coinbase:spot:BTC/USD": market_result(),
            "kraken:spot:ETH/USD": market_result(),
        },
    }


def test_builds_synchronized_weighted_portfolio_with_cash_and_risk_evidence():
    report = build_portfolio_quality_report(
        source(),
        {"coinbase:spot:BTC/USD": 0.4, "kraken:spot:ETH/USD": 0.4},
        minimum_folds=2,
    )
    assert report["schema_version"] == "eti.portfolio_quality_report.v1"
    assert report["cash_weight_fraction"] == pytest.approx(0.2)
    assert report["fold_count"] == 2
    assert report["evidence_status"] == "human_review_required"
    assert report["folds"][0]["bar_count"] == 3
    assert report["folds"][0]["total_return"] == pytest.approx(
        (1 + 0.08) * (1 - 0.08) * (1 + 0.04) - 1
    )
    assert report["folds"][0]["correlation_matrix"]["coinbase:spot:BTC/USD"]["kraken:spot:ETH/USD"] == pytest.approx(1.0)
    assert report["execution_authority"] is False
    assert report["model_promotion_authority"] is False


def test_rejects_weights_above_one_and_unknown_instruments():
    with pytest.raises(ValueError, match="cannot exceed 1.0"):
        build_portfolio_quality_report(source(), {
            "coinbase:spot:BTC/USD": 0.7, "kraken:spot:ETH/USD": 0.4
        })
    with pytest.raises(ValueError, match="unknown instrument"):
        build_portfolio_quality_report(source(), {
            "coinbase:spot:BTC/USD": 0.5, "unknown:spot:X": 0.5
        })


def test_rejects_missing_synchronized_return_streams():
    broken = source()
    del broken["markets"]["coinbase:spot:BTC/USD"]["report"]["trading_performance"]["folds"][0]["timestamps"]
    with pytest.raises(ValueError, match="lacks timestamped"):
        build_portfolio_quality_report(
            broken, {"coinbase:spot:BTC/USD": 0.5, "kraken:spot:ETH/USD": 0.5}
        )


def test_requires_completed_training_and_disabled_execution():
    broken = source()
    broken["execution_enabled"] = True
    with pytest.raises(ValueError, match="disabled execution"):
        build_portfolio_quality_report(
            broken, {"coinbase:spot:BTC/USD": 0.5, "kraken:spot:ETH/USD": 0.5}
        )


def test_rejects_invalid_stream_values():
    broken = source()
    broken["markets"]["coinbase:spot:BTC/USD"]["report"]["trading_performance"]["folds"][0]["trading"]["model"]["bar_returns"][0] = math.nan
    with pytest.raises(ValueError, match="finite number"):
        build_portfolio_quality_report(
            broken, {"coinbase:spot:BTC/USD": 0.5, "kraken:spot:ETH/USD": 0.5}
        )


def test_insufficient_fold_count_is_not_promoted_to_approval():
    report = build_portfolio_quality_report(
        source(),
        {"coinbase:spot:BTC/USD": 0.5, "kraken:spot:ETH/USD": 0.5},
        minimum_folds=3,
    )
    assert report["evidence_status"] == "insufficient_synchronized_folds"
    assert report["execution_authority"] is False


def test_rejects_inconsistent_outer_test_fold_identities():
    broken = source()
    broken["markets"]["kraken:spot:ETH/USD"]["report"]["trading_performance"]["folds"].pop()
    with pytest.raises(ValueError, match="identical outer-test fold identifiers"):
        build_portfolio_quality_report(
            broken, {"coinbase:spot:BTC/USD": 0.5, "kraken:spot:ETH/USD": 0.5}
        )


def test_held_position_is_carried_across_another_market_timestamp():
    broken = source()
    first = broken["markets"]["coinbase:spot:BTC/USD"]["report"]["trading_performance"]["folds"][0]
    first["timestamps"] = [first["timestamps"][0], first["timestamps"][2]]
    first["trading"]["model"]["bar_returns"] = [0.10, 0.05]
    first["trading"]["model"]["positions"] = [1.0, 0.0]
    report = build_portfolio_quality_report(
        broken, {"coinbase:spot:BTC/USD": 0.5, "kraken:spot:ETH/USD": 0.5}
    )
    assert report["folds"][0]["bar_count"] == 3
    assert report["folds"][0]["mean_gross_exposure_fraction"] == pytest.approx(2 / 3)
