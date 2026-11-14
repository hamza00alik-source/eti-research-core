from eti.agents import _training_data_quality_review


def sample_quality(status="fresh", gap_count=0, coverage=1.0):
    return {
        "status": status,
        "gap_count": gap_count,
        "calendar_coverage_ratio": coverage,
    }


def test_crypto_freshness_is_a_required_training_gate():
    review = _training_data_quality_review(sample_quality("stale", 2, 0.8), "crypto")

    assert review["freshness_gate_applied"] is True
    assert review["freshness_gate_passed"] is False
    assert review["calendar_gaps_blocking"] is False


def test_session_based_market_staleness_is_informational():
    review = _training_data_quality_review(sample_quality("stale", 12, 0.4), "equity")

    assert review["freshness_gate_applied"] is False
    assert review["freshness_gate_passed"] is None
    assert review["calendar_gaps_blocking"] is False
    assert "weekends and holidays" in review["review_note"]


def test_fresh_crypto_data_passes_freshness_gate():
    review = _training_data_quality_review(sample_quality("fresh", 0, 1.0), "crypto")

    assert review["freshness_gate_passed"] is True
