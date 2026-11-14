import pytest

from eti.solana_costs import REQUIRED_COST_COMPONENTS, evaluate_execution_cost_evidence


def _evidence():
    components = {
        name: {
            "cost_bps": value,
            "unit": "bps_of_entry_notional",
            "measurement": "measured",
            "source_id": f"source-{name}",
            "observed_at": "2026-11-14T12:00:00Z",
        }
        for name, value in zip(REQUIRED_COST_COMPONENTS, (10, 20, 5, 3, 7))
    }
    return {
        "cost_components": components,
        "exit_evidence": {
            "status": "exit_observed",
            "source_id": "chain-observation-1",
            "transaction_id": "signature-1",
            "observed_at": "2026-11-14T12:01:00Z",
        },
    }


def test_missing_costs_never_default_to_zero_or_report_net_return():
    report = evaluate_execution_cost_evidence(5.0, {})
    assert report["status"] == "COST_EVIDENCE_INCOMPLETE"
    assert set(report["missing_components"]) == set(REQUIRED_COST_COMPONENTS) | {"successful_exit_evidence"}
    assert report["measured_round_trip_cost_bps"] is None
    assert report["cost_adjusted_sampled_mark_scenario_return_pct"] is None
    assert report["net_return_pct"] is None
    assert report["realized_pnl_available"] is False


def test_complete_measured_evidence_only_produces_labeled_sampled_mark_scenario():
    report = evaluate_execution_cost_evidence(5.0, _evidence())
    assert report["status"] == "COST_EVIDENCE_COMPLETE"
    assert report["measured_round_trip_cost_bps"] == pytest.approx(45.0)
    assert report["cost_adjusted_sampled_mark_scenario_return_pct"] == pytest.approx(4.55)
    assert report["net_return_pct"] is None
    assert report["realized_pnl_available"] is False


@pytest.mark.parametrize("bad_value", [-1, float("nan"), float("inf"), True, "2"])
def test_invalid_cost_values_are_rejected(bad_value):
    evidence = _evidence()
    evidence["cost_components"]["spread"]["cost_bps"] = bad_value
    report = evaluate_execution_cost_evidence(5.0, evidence)
    assert report["status"] == "COST_EVIDENCE_INCOMPLETE"
    assert "spread" in report["invalid_components"]
    assert report["net_return_pct"] is None


@pytest.mark.parametrize("field,value", [
    ("source_id", ""),
    ("observed_at", "not-a-timestamp"),
    ("observed_at", "2026-11-14T12:00:00"),
    ("unit", "percent"),
    ("measurement", "estimate"),
])
def test_unsourced_undated_estimated_or_wrong_unit_costs_are_rejected(field, value):
    evidence = _evidence()
    evidence["cost_components"]["spread"][field] = value
    report = evaluate_execution_cost_evidence(5.0, evidence)
    assert "spread" in report["invalid_components"]
    assert report["net_return_pct"] is None


def test_failed_or_missing_exit_evidence_blocks_cost_adjusted_scenario():
    evidence = _evidence()
    evidence["exit_evidence"]["status"] = "exit_failed"
    report = evaluate_execution_cost_evidence(5.0, evidence)
    assert report["status"] == "COST_EVIDENCE_INCOMPLETE"
    assert "exit_evidence_failed" in report["invalid_components"]
    assert report["cost_adjusted_sampled_mark_scenario_return_pct"] is None


def test_invalid_gross_return_is_rejected():
    report = evaluate_execution_cost_evidence(-100.0, _evidence())
    assert "gross_sampled_return_pct" in report["invalid_components"]
    assert report["net_return_pct"] is None
