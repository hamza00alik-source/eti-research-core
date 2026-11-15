import pytest

from eti.solana_provider_health import evaluate_solana_provider_health


def test_provider_health_uses_latest_explicit_observation_only():
    report = evaluate_solana_provider_health([
        {"provider": "rpc", "observed_at": "2026-01-01T00:00:00Z",
         "status": "error", "error_code": "timeout"},
        {"provider": "rpc", "observed_at": "2026-01-01T00:04:00Z",
         "status": "success", "latency_ms": 120},
        {"provider": "dexscreener", "observed_at": "2026-01-01T00:04:00Z",
         "status": "success", "latency_ms": 4200},
    ], now="2026-01-01T00:04:30Z")
    providers = {row["provider"]: row for row in report["providers"]}
    assert providers["rpc"]["status"] == "HEALTHY"
    assert providers["rpc"]["supplied_observation_count"] == 2
    assert providers["rpc"]["supplied_error_count"] == 1
    assert providers["dexscreener"]["status"] == "DEGRADED"
    assert report["uptime_estimated"] is False
    assert report["trading_authorized"] is False


def test_provider_health_marks_missing_stale_and_future_evidence_unknown():
    report = evaluate_solana_provider_health([
        {"provider": "stale", "observed_at": "2026-01-01T00:00:00Z",
         "status": "success", "latency_ms": 10},
        {"provider": "future", "observed_at": "2026-01-01T00:06:00Z",
         "status": "success", "latency_ms": 10},
        {"provider": "no-latency", "observed_at": "2026-01-01T00:04:00Z",
         "status": "success"},
    ], now="2026-01-01T00:05:00Z", stale_after_seconds=60)
    statuses = {row["provider"]: row["status"] for row in report["providers"]}
    assert statuses == {"future": "UNKNOWN", "no-latency": "UNKNOWN", "stale": "UNKNOWN"}
    assert report["missing_observation_means_failure"] is False


def test_provider_health_rejects_naive_clock_and_invalid_thresholds():
    with pytest.raises(ValueError):
        evaluate_solana_provider_health([], now="2026-01-01T00:00:00")
    with pytest.raises(ValueError):
        evaluate_solana_provider_health([], now="2026-01-01T00:00:00Z", stale_after_seconds=0)
