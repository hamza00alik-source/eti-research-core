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



def test_calendar_config_registers_explicit_venue_scope(tmp_path):
    from eti.market_data_hub import MarketDataHub

    path = tmp_path / "calendars.json"
    path.write_text(json.dumps({
        "calendars": [{
            "venue": "yahoo", "asset_class": "equity", "market_type": "spot",
            "calendar_id": "TEST-NYSE", "timezone_name": "America/New_York",
            "sessions": {str(day): ["09:30", "16:00"] for day in range(5)},
            "holidays": ["2026-01-01"],
            "early_closes": {"2026-11-27": "13:00"},
        }]
    }))
    hub = MarketDataHub()
    research_cli._register_calendar_config(hub, path)
    assert ("yahoo", "equity", "spot") in hub._session_calendars
    calendar = hub._session_calendars[("yahoo", "equity", "spot")]
    assert calendar.calendar_id == "TEST-NYSE"
    assert len(calendar.holidays) == 1
