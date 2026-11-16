import pytest

from eti.promotion import assess_challenger_promotion


def digest(ch):
    return ch * 64


def report(model, returns, *, evaluation=None, cost=None, ece=0.03, drawdown=0.05):
    return {
        "model_sha256": digest(model),
        "training_data_sha256": digest("a" if model == "b" else "c"),
        "evaluation_data_sha256": digest(evaluation or "d"),
        "cost_model_sha256": digest(cost or "e"),
        "cost_adjusted": True,
        "evaluation_locked": True,
        "folds": [{
            "test_start": f"2026-01-{i*2+1:02d}T00:00:00Z",
            "test_end": f"2026-01-{i*2+2:02d}T00:00:00Z",
            "test_bars": 200,
            "net_return_bps": value,
            "max_drawdown": drawdown,
            "out_of_sample_ece": ece,
        } for i, value in enumerate(returns)],
    }


def assess(champ, chall, **kwargs):
    return assess_challenger_promotion(
        champ, chall, comparison_family_size=1, **kwargs
    )


def test_promotion_gate_requires_five_nonoverlapping_paired_folds():
    result = assess(
        report("a", [1, 2, 1, 2]),
        report("b", [3, 4, 3, 4]),
    )
    assert result["promotion_eligible"] is False
    assert result["status"] == "insufficient_independent_evidence"


def test_promotion_gate_requires_same_evaluation_and_cost_hashes():
    with pytest.raises(ValueError, match="must share evaluation_data_sha256"):
        assess(
            report("a", [1] * 5),
            report("b", [2] * 5, evaluation="f"),
        )


def test_promotion_gate_rejects_overlapping_test_windows():
    champ = report("a", [1] * 5)
    chall = report("b", [2] * 5)
    champ["folds"][1]["test_start"] = champ["folds"][0]["test_start"]
    chall["folds"][1]["test_start"] = chall["folds"][0]["test_start"]
    with pytest.raises(ValueError, match="test folds overlap"):
        assess(champ, chall)


def test_all_positive_folds_can_only_be_eligible_for_human_review():
    result = assess(
        report("a", [1] * 5),
        report("b", [4] * 5),
    )
    assert result["promotion_eligible"] is True
    assert result["status"] == "eligible_for_human_review"
    assert result["automatic_promotion"] is False
    assert result["requires_human_review"] is True


def test_bonferroni_penalty_blocks_small_comparison_family_significance():
    result = assess_challenger_promotion(
        report("a", [1] * 5), report("b", [4] * 5),
        comparison_family_size=3,
    )
    assert result["sign_test_p_value"] == pytest.approx(1 / 32)
    assert result["bonferroni_adjusted_p_value"] == pytest.approx(3 / 32)
    assert result["promotion_eligible"] is False


def test_drawdown_and_oos_calibration_gates_apply():
    drawdown = assess(
        report("a", [1] * 5), report("b", [4] * 5, drawdown=0.20)
    )
    calibration = assess(
        report("a", [1] * 5), report("b", [4] * 5, ece=0.12)
    )
    assert drawdown["promotion_eligible"] is False
    assert calibration["promotion_eligible"] is False


def test_promotion_gate_requires_cost_adjusted_locked_evidence():
    champ = report("a", [1] * 5)
    chall = report("b", [4] * 5)
    chall["cost_adjusted"] = False
    with pytest.raises(ValueError, match="cost-adjusted"):
        assess(champ, chall)



def test_promotion_gate_rejects_material_calibration_or_drawdown_regression():
    champion = report("a", [1] * 5, ece=0.03, drawdown=0.05)
    challenger_ece = report("b", [4] * 5, ece=0.07, drawdown=0.05)
    challenger_drawdown = report("b", [4] * 5, ece=0.03, drawdown=0.08)
    assert assess(champion, challenger_ece)["promotion_eligible"] is False
    assert assess(champion, challenger_drawdown)["promotion_eligible"] is False


def test_sign_test_excludes_tied_fold_differences():
    result = assess(
        report("a", [1, 1, 1, 1, 1]),
        report("b", [2, 1, 2, 1, 2]),
    )
    assert result["non_tied_fold_count"] == 3
    assert result["fold_count"] == 5
    assert result["sign_test_p_value"] == pytest.approx(0.125)
    assert result["promotion_eligible"] is False


def test_promotion_gate_rejects_single_fold_drawdown_or_ece_breach():
    champion = report("a", [1] * 5, ece=0.03, drawdown=0.05)
    challenger = report("b", [4] * 5, ece=0.03, drawdown=0.05)
    challenger["folds"][0]["max_drawdown"] = 0.20
    assert assess(champion, challenger)["promotion_eligible"] is False
