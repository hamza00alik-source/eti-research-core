import pytest

from eti.intelligence import (
    DEPARTMENTS,
    build_intelligence_manifest,
    department_for_capability,
    route_department_task,
    select_specialist_models,
)


def model(*, department="prediction", lifecycle="approved", classes=None,
          market_types=None, timeframes=None, version="1.0.0"):
    schema = {"department": department}
    if market_types is not None:
        schema["market_types"] = market_types
    if timeframes is not None:
        schema["timeframes"] = timeframes
    return {
        "model_id": "specialist",
        "version": version,
        "lifecycle": lifecycle,
        "input_schema": schema,
        "supported_classes": classes or ["crypto_spot"],
        "evaluation_report": {"passed": True},
    }


def test_architecture_has_distinct_specialist_departments():
    assert set(DEPARTMENTS) == {
        "market_data", "market_intelligence", "prediction", "strategy",
        "risk", "macro", "execution", "learning",
    }


@pytest.mark.parametrize(
    ("capability", "department"),
    [
        ("market.coverage", "market_data"),
        ("market.discover", "market_data"),
        ("market.snapshot", "market_intelligence"),
        ("derivatives.context", "market_intelligence"),
        ("model.train_research", "prediction"),
        ("strategy.walk_forward", "strategy"),
        ("paper.tick", "strategy"),
        ("live.tick", "execution"),
    ],
)
def test_capability_routes_to_explicit_department(capability, department):
    assert department_for_capability(capability) == department
    result = route_department_task(capability, {"symbol": "BTC/USDT"})
    assert result["department"] == department
    assert result["new_permissions_granted"] is False
    assert result["model_selected"] is False
    assert result["execution_enabled"] is False


def test_unknown_capability_does_not_get_guessed_department():
    with pytest.raises(LookupError):
        department_for_capability("model.generated.arbitrary")


def test_specialist_selection_requires_approved_explicitly_assigned_model():
    versions = [
        model(),
        model(lifecycle="candidate", version="1.0.1"),
        model(department="risk", version="1.0.2"),
        {
            "model_id": "legacy-btc-model", "version": "0.1.0",
            "lifecycle": "approved", "supported_classes": ["crypto_spot"],
            "input_schema": {"features": ["rsi"]},
        },
    ]
    selected = select_specialist_models(versions, "prediction", asset_class="crypto", market_type="spot")
    assert [(item["model_id"], item["version"]) for item in selected] == [("specialist", "1.0.0")]


def test_specialist_selection_respects_market_type_and_timeframe():
    candidate = model(market_types=["spot"], timeframes=["1h"])
    assert select_specialist_models(
        [candidate], "prediction", asset_class="crypto", market_type="swap", timeframe="1h"
    ) == []
    assert select_specialist_models(
        [candidate], "prediction", asset_class="crypto", market_type="spot", timeframe="4h"
    ) == []
    assert len(select_specialist_models(
        [candidate], "prediction", asset_class="crypto", market_type="spot", timeframe="1h"
    )) == 1


def test_conflicting_department_metadata_fails_closed():
    item = model()
    item["evaluation_report"]["department"] = "risk"
    with pytest.raises(ValueError, match="conflicting"):
        select_specialist_models([item], "prediction")


def test_manifest_reports_unconfigured_departments_and_legacy_models():
    manifest = build_intelligence_manifest(
        [model(), model(department="strategy", lifecycle="candidate"),
         {"model_id": "legacy", "version": "0.1", "lifecycle": "approved"}],
        capabilities=["market.snapshot", "model.train_research"],
    )
    assert manifest["schema"] == "eti.shared_intelligence_architecture.v1"
    assert manifest["departments"]["prediction"]["status"] == "ready"
    assert manifest["departments"]["strategy"]["status"] == "candidate_review"
    assert manifest["departments"]["macro"]["status"] == "not_configured"
    assert manifest["unassigned_legacy_model_versions"] == 1
    assert manifest["model_promotion_authority"] is False
    assert manifest["budget_authority"] is False
    assert manifest["execution_authority"] is False


def test_manifest_never_calls_task_success_trading_quality():
    manifest = build_intelligence_manifest([])
    assert any("not proof of profitability" in item for item in manifest["limitations"])


def test_specialist_selection_is_role_scoped_and_manifest_reports_role_readiness():
    direction = model()
    direction["input_schema"]["model_role"] = "direction_classifier"
    volatility = model(version="1.0.1")
    volatility["input_schema"]["model_role"] = "volatility_forecast"
    selected = select_specialist_models(
        [direction, volatility], "prediction", asset_class="crypto",
        market_type="spot", model_role="direction_classifier"
    )
    assert [item["version"] for item in selected] == ["1.0.0"]
    manifest = build_intelligence_manifest([direction, volatility])
    roles = manifest["departments"]["prediction"]["role_readiness"]
    assert roles["direction_classifier"]["status"] == "ready"
    assert roles["volatility_forecast"]["status"] == "ready"
    assert roles["return_distribution"]["status"] == "not_configured"


def test_role_selection_rejects_role_from_another_department():
    with pytest.raises(ValueError, match="model_role"):
        select_specialist_models([], "prediction", model_role="slippage_estimator")


def test_model_with_invalid_declared_role_fails_closed():
    candidate = model()
    candidate["input_schema"]["model_role"] = "macro_regime"
    with pytest.raises(ValueError, match="model_role"):
        select_specialist_models([candidate], "prediction")


def test_department_specialist_capabilities_are_explicitly_routed():
    expected = {
        "market.regime": "market_intelligence",
        "strategy.select": "strategy",
        "risk.assess": "risk",
        "macro.as_of": "macro",
        "execution.cost_estimate": "execution",
        "learning.economic_assess": "learning",
    }
    for capability, department in expected.items():
        assert department_for_capability(capability) == department
        routed = route_department_task(capability, {})
        assert routed["department"] == department
        assert routed["execution_enabled"] is False
