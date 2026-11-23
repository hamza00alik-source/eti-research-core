import pytest

from eti.queen_decision import queen_trade_review


def valid_evidence():
    return {
        "market_evidence": {"status": "valid", "candle_age_seconds": 60, "spread_bps": 8},
        "model_evidence": {
            "status": "validated", "probability_up": 0.72,
            "oos_net_edge_bps": 28, "oos_observations": 240,
            "expected_edge_bps": 45,
        },
        "strategy_evidence": {
            "status": "validated", "oos_net_return_bps": 180,
            "oos_trades": 75, "oos_max_drawdown": 0.06,
        },
        "cost_evidence": {"status": "validated", "round_trip_cost_bps": 12},
        "risk_evidence": {"approved": True, "hard_blocks": []},
        "qwen_advisory": {"status": "completed", "recommendation": "watch"},
    }


def test_complete_independent_evidence_can_only_become_paper_candidate():
    result = queen_trade_review(valid_evidence())
    assert result["disposition"] == "paper_candidate"
    assert result["blockers"] == []
    assert result["execution_authority"] is False
    assert result["live_orders_allowed"] is False
    assert result["model_promotion_authority"] is False
    assert result["requires_separate_paper_engine"] is True


@pytest.mark.parametrize(
    ("section", "field", "value", "blocker"),
    [
        ("market_evidence", "status", "unknown", "market_data_not_valid"),
        ("market_evidence", "candle_age_seconds", 9000, "market_data_stale_or_unknown"),
        ("market_evidence", "spread_bps", None, "spread_unknown_or_excessive"),
        ("model_evidence", "probability_up", 0.51, "model_confidence_below_threshold"),
        ("model_evidence", "oos_net_edge_bps", -2, "model_has_no_positive_oos_net_edge"),
        ("model_evidence", "oos_observations", 20, "model_oos_sample_too_small"),
        ("strategy_evidence", "oos_net_return_bps", 0, "strategy_has_no_positive_oos_net_return"),
        ("strategy_evidence", "oos_trades", 5, "strategy_oos_trade_sample_too_small"),
        ("strategy_evidence", "oos_max_drawdown", 0.3, "strategy_drawdown_unknown_or_excessive"),
        ("cost_evidence", "status", "estimated", "transaction_costs_not_validated"),
        ("cost_evidence", "round_trip_cost_bps", 40, "expected_edge_does_not_clear_cost_buffer"),
        ("risk_evidence", "approved", False, "deterministic_risk_gate_not_approved"),
    ],
)
def test_any_failed_gate_holds_candidate(section, field, value, blocker):
    payload = valid_evidence()
    payload[section][field] = value
    result = queen_trade_review(payload)
    assert result["disposition"] == "hold"
    assert blocker in result["blockers"]
    assert result["execution_authority"] is False


def test_qwen_can_warn_or_veto_but_never_approve():
    payload = valid_evidence()
    payload["qwen_advisory"] = {"status": "completed", "recommendation": "avoid"}
    result = queen_trade_review(payload)
    assert result["disposition"] == "hold"
    assert "qwen_advisory_caution" in result["blockers"]
    payload["qwen_advisory"] = {"status": "completed", "recommendation": "buy"}
    result = queen_trade_review(payload)
    assert result["disposition"] == "paper_candidate"
    assert "qwen_advisory_schema_unrecognized" in result["warnings"]
    assert result["qwen_can_approve"] is False


def test_qwen_unavailable_does_not_substitute_for_missing_hard_evidence():
    payload = valid_evidence()
    payload["risk_evidence"]["approved"] = False
    payload["qwen_advisory"] = {"status": "provider_unavailable"}
    result = queen_trade_review(payload)
    assert result["disposition"] == "hold"
    assert "deterministic_risk_gate_not_approved" in result["blockers"]
    assert "qwen_advisory_unavailable" in result["warnings"]


def test_malformed_payload_fails_closed():
    result = queen_trade_review({"market_evidence": None})
    assert result["disposition"] == "hold"
    assert result["execution_authority"] is False
