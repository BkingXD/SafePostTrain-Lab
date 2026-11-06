import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labs"))
from core_math import dpo_loss, group_advantages, ppo_surrogate, sft_loss
from toy_training import train
from data_audit import SAMPLES, audit_and_split
from reward_model import fit


class TeachingMathTests(unittest.TestCase):
    def test_worked_examples(self):
        self.assertAlmostEqual(sft_loss([.8, .6, .5, .25], [0, 0, 1, 1]), 1.0397207708399179)
        self.assertAlmostEqual(dpo_loss(-2, -4, -3, -4, .1), .6443966600735709)
        self.assertAlmostEqual(ppo_surrogate(.5, -1), -.8)
        self.assertEqual(group_advantages([1, 1, 1]), [0, 0, 0])

    def test_gradient_direction_and_reproducible_updates(self):
        for method in ["sft", "dpo", "ppo", "grpo"]:
            a, b = train(method, 80, 7), train(method, 80, 7)
            self.assertEqual(a, b)
            self.assertTrue(.7 < a["final_probability"] < 1, method)
            self.assertTrue(all(math.isfinite(x["p_action_1"]) for x in a["history"]))

    def test_empty_supervision_rejected(self):
        with self.assertRaises(ValueError):
            sft_loss([.5], [0])

    def test_group_split_and_exact_duplicate_detection(self):
        result = audit_and_split(SAMPLES)
        self.assertEqual(result["duplicates"], ["b2"])
        groups = [{r["group"] for r in rows} for rows in result["splits"].values()]
        for i, group in enumerate(groups):
            for other in groups[i+1:]:
                self.assertFalse(group & other)

    def test_reward_prefers_correctness_over_length(self):
        result = fit()
        self.assertGreater(result["short_correct_vs_long_incorrect_probability"], .9)
        self.assertEqual(result, fit())


if __name__ == "__main__":
    unittest.main()
