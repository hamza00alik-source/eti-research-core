import pytest

from eti.colony_intelligence import build_colony_intelligence


def task(capability="market.snapshot", status="completed", attempt_count=1,
         created_at="2026-11-15T00:00:00Z", updated_at="2026-11-15T00:00:05Z"):
    return {
        "target_capability": capability, "status": status,
        "attempt_count": attempt_count, "created_at": created_at,
        "updated_at": updated_at,
    }


def test_colony_intelligence_reports_capability_reliability_with_confidence():
    records = [task(status="completed") for _ in range(18)]
    records += [task(status="failed", attempt_count=2) for _ in range(2)]
    report = build_colony_intelligence(records, min_capability_sample=20)
    metric = report["capabilities"]["market.snapshot"]
    assert metric["terminal_tasks"] == 20
    assert metric["success_rate"] == 0.9
    assert metric["retry_count"] == 2
    assert metric["success_rate_wilson_95"][0] < 0.9
    assert metric["success_rate_wilson_95"][1] > 0.9
    assert metric["routing_reliability_eligible"] is True
    assert report["execution_authority"] is False


def test_small_samples_are_not_eligible_for_reliability_ranking():
    report = build_colony_intelligence([task()] * 3, min_capability_sample=20)
    metric = report["capabilities"]["market.snapshot"]
    assert metric["evidence_status"] == "insufficient_sample"
    assert metric["routing_reliability_eligible"] is False


def test_pending_and_cancelled_tasks_are_not_counted_as_success_or_failure():
    report = build_colony_intelligence([
        task(status="queued"), task(status="running"), task(status="cancelled")
    ])
    metric = report["capabilities"]["market.snapshot"]
    assert metric["terminal_tasks"] == 0
    assert metric["success_rate"] is None
    assert report["status_counts"]["queued"] == 1
    assert report["status_counts"]["running"] == 1
    assert report["status_counts"]["cancelled"] == 1


def test_end_to_end_latency_is_not_reported_for_invalid_or_naive_timestamps():
    report = build_colony_intelligence([
        task(created_at="2026-11-15T00:00:00", updated_at="2026-11-15T00:00:05Z"),
        task(created_at="2026-11-15T00:00:05Z", updated_at="2026-11-15T00:00:01Z"),
    ])
    metric = report["capabilities"]["market.snapshot"]
    assert metric["latency_observations"] == 0
    assert metric["median_end_to_end_seconds"] is None


def test_history_truncation_and_input_validation():
    records = [task() for _ in range(5)]
    report = build_colony_intelligence(records, history_limit=5)
    assert report["history_may_be_truncated"] is True
    with pytest.raises(ValueError):
        build_colony_intelligence(records, history_limit=501)
    with pytest.raises(ValueError):
        build_colony_intelligence(records, min_capability_sample=0)
