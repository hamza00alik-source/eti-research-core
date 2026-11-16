import unittest

from eti.economic_disposition import assess_economic_viability


def report():
    return {
        "training_performed": True,
        "execution_enabled": False,
        "fold_count": 3,
        "folds": [{}, {}, {}],
        "trading_performance": {"summary": {
            "mean_net_return": 0.02,
            "folds_beating_no_trade": 3,
        }},
        "cost_sensitivity": {"scenarios": [
            {"cost_multiplier": 1.0, "fold_count": 3,
             "mean_net_return": 0.02, "positive_return_folds": 3},
            {"cost_multiplier": 2.0, "fold_count": 3,
             "mean_net_return": 0.01, "positive_return_folds": 3}
        ]},
    }


class EconomicDispositionTests(unittest.TestCase):
    def test_profitable_cost_robust_result_still_requires_human_review(self):
        result = assess_economic_viability(report())
        self.assertEqual(result["status"], "human_review_required")
        self.assertFalse(result["live_execution_eligible"])

    def test_negative_base_return_is_no_go(self):
        item = report()
        item["trading_performance"]["summary"]["mean_net_return"] = -0.01
        result = assess_economic_viability(item)
        self.assertEqual(result["status"], "no_go")
        self.assertIn("mean net return is not positive at base costs", result["reasons"])

    def test_negative_two_x_cost_result_is_no_go(self):
        item = report()
        item["cost_sensitivity"]["scenarios"][1]["mean_net_return"] = -0.02
        result = assess_economic_viability(item)
        self.assertEqual(result["status"], "no_go")

    def test_not_all_folds_beat_no_trade_is_no_go(self):
        item = report()
        item["trading_performance"]["summary"]["folds_beating_no_trade"] = 2
        result = assess_economic_viability(item)
        self.assertEqual(result["status"], "no_go")

    def test_missing_evidence_fails_closed(self):
        result = assess_economic_viability({"training_performed": True})
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertFalse(result["live_execution_eligible"])

    def test_execution_enabled_is_insufficient_evidence(self):
        item = report()
        item["execution_enabled"] = True
        result = assess_economic_viability(item)
        self.assertEqual(result["status"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
