import json
import sys

from eti import research_cli


def test_scan_command_writes_scan_only_artifact(tmp_path, monkeypatch):
    output = tmp_path / "nested" / "scan.json"
    monkeypatch.setattr(research_cli, "scan_market_universe", lambda **kwargs: {
        "scan_only": True, "training_performed": False,
        "execution_enabled": False, "summary": {"candidate_count": 0},
        "candidates": [],
    })
    monkeypatch.setattr(sys, "argv", [
        "eti.research_cli", "scan", "--output", str(output),
    ])
    research_cli.main()
    result = json.loads(output.read_text())
    assert result["scan_only"] is True
    assert result["execution_enabled"] is False


def test_allocate_command_requires_explicit_cost_inputs_and_writes_plan(tmp_path, monkeypatch):
    scan_path = tmp_path / "scan.json"
    costs_path = tmp_path / "costs.json"
    output = tmp_path / "allocation.json"
    scan_path.write_text(json.dumps({
        "scan_only": True, "execution_enabled": False,
        "candidates": [],
    }))
    costs_path.write_text("{}")
    monkeypatch.setattr(sys, "argv", [
        "eti.research_cli", "allocate", str(scan_path),
        "--cost-models", str(costs_path),
        "--estimated-run-cost-usd", "1",
        "--daily-spend-to-date-usd", "0",
        "--monthly-spend-to-date-usd", "0",
        "--output", str(output),
    ])
    research_cli.main()
    result = json.loads(output.read_text())
    assert result["plan_only"] is True
    assert result["training_authorized"] is False
    assert result["execution_enabled"] is False
