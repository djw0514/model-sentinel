import json
import math
import unittest
from unittest.mock import patch

from model_sentinel.sequential import SequentialConfig, SequentialTest, _binomial_tail, fixed_sample_verdict


class SequentialTests(unittest.TestCase):
    def test_twelve_clean_samples_stop_as_stable(self):
        test = SequentialTest()
        decisions = [test.add(False) for _ in range(12)]
        self.assertEqual(decisions[-1], "stable")
        self.assertEqual(test.samples, 12)

    def test_two_early_regressions_stop_as_regressed(self):
        test = SequentialTest()
        outcomes = [True, True, False, False, False, False, False, False]
        for outcome in outcomes:
            decision = test.add(outcome)
        self.assertEqual(decision, "regressed")

    def test_boundary_case_is_inconclusive_at_cap(self):
        config = SequentialConfig(max_samples=32)
        test = SequentialTest(config)
        for index in range(32):
            test.add(index in {0, 10})
        self.assertEqual(test.decision(), "inconclusive")

    def test_invalid_hypotheses_are_rejected(self):
        with self.assertRaises(ValueError):
            SequentialConfig(p0=0.2, p1=0.1)

    def test_terminal_decision_cannot_be_reversed(self):
        test = SequentialTest()
        for _ in range(12):
            test.add(False)
        with self.assertRaises(ValueError):
            test.add(True)
        self.assertEqual(test.samples, 12)
        self.assertEqual(test.decision(), "stable")

    def test_quality_gate_defers_stable_until_a_later_valid_observation(self):
        test = SequentialTest()
        for _ in range(12):
            self.assertEqual(test.add(False, allow_stable=False), "running")
        self.assertEqual(test.statistical_decision(), "stable")
        self.assertEqual(test.snapshot()["decision"], "running")
        self.assertEqual(test.add(False, allow_stable=True), "stable")
        with self.assertRaises(ValueError):
            test.add(True, allow_stable=False)
        self.assertEqual(test.samples, 13)

    def test_failures_after_deferred_stability_change_the_evidence(self):
        test = SequentialTest()
        for _ in range(12):
            test.add(False, allow_stable=False)
        self.assertEqual(test.add(True), "running")
        self.assertEqual(test.statistical_decision(), "running")
        self.assertEqual(test.add(True), "running")
        self.assertEqual(test.add(True), "regressed")
        self.assertEqual(test.regressions, 3)

    def test_quality_gate_never_suppresses_regression_or_reopens_it(self):
        test = SequentialTest()
        for outcome in (True, True, False, False, False, False, False, False):
            test.add(outcome, allow_stable=False)
        self.assertEqual(test.decision(), "regressed")
        with self.assertRaises(ValueError):
            test.add(False, allow_stable=True)

    def test_quality_gate_respects_sample_cap_and_terminal_lock(self):
        test = SequentialTest(SequentialConfig(max_samples=12))
        for _ in range(12):
            decision = test.add(False, allow_stable=False)
        self.assertEqual(decision, "inconclusive")
        self.assertEqual(test.statistical_decision(), "stable")
        with self.assertRaises(ValueError):
            test.add(False, allow_stable=True)
        self.assertEqual(test.decision(), "inconclusive")

    def test_quality_gate_requires_a_boolean_without_consuming_samples(self):
        test = SequentialTest()
        with self.assertRaises(ValueError):
            test.add(False, allow_stable=1)
        self.assertEqual(test.samples, 0)

    def test_invalid_limits_and_error_targets_are_rejected(self):
        for kwargs in ({"alpha": 0.8, "beta": 0.8}, {"p0": float("nan")},
                       {"beta": float("inf")}, {"min_samples": True},
                       {"max_samples": 10.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SequentialConfig(**kwargs)

    def test_fixed_comparison_uses_all_samples(self):
        result = fixed_sample_verdict(32, 20, SequentialConfig(max_samples=8))
        self.assertEqual(result["decision"], "regressed")
        self.assertEqual(result["samples"], 32)

    def test_fixed_clean_sample_and_exact_tail(self):
        result = fixed_sample_verdict(12, 0)
        self.assertEqual(result["decision"], "stable")
        self.assertAlmostEqual(result["stable_p_value"], 0.8 ** 12)
        self.assertAlmostEqual(result["regression_p_value"], 1.0)

    def test_binomial_endpoints_are_exact_and_do_not_expand_large_support(self):
        with patch("model_sentinel.sequential.math.lgamma", side_effect=AssertionError("unexpected distribution expansion")):
            clean = fixed_sample_verdict(1_000_000, 0)
            failed = fixed_sample_verdict(1_000_000, 1_000_000)
            self.assertEqual(clean["regression_p_value"], 1.0)
            self.assertEqual(clean["decision"], "stable")
            self.assertEqual(failed["stable_p_value"], 1.0)
            self.assertEqual(failed["decision"], "regressed")
            self.assertEqual(_binomial_tail(12, 0.2, 5, 4), 0.0)
            self.assertAlmostEqual(_binomial_tail(12, 0.2, 0, 0), 0.8 ** 12)
            self.assertAlmostEqual(_binomial_tail(12, 0.2, 12, 12), 0.2 ** 12)
            self.assertEqual(_binomial_tail(0, 0.2, 0, 0), 1.0)

    def test_fixed_small_sample_and_indifference_region(self):
        self.assertEqual(fixed_sample_verdict(0, 0)["decision"], "inconclusive")
        self.assertEqual(fixed_sample_verdict(4, 4)["decision"], "inconclusive")
        self.assertEqual(fixed_sample_verdict(1000, 100)["decision"], "inconclusive")

    def test_fixed_count_validation(self):
        for counts in ((-1, 0), (4, 5), (True, 0), (4, 1.0)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                fixed_sample_verdict(*counts)

    def test_subnormal_probabilities_do_not_overflow_or_produce_nan(self):
        config = SequentialConfig(p0=5e-324, alpha=5e-324, beta=5e-324)
        test = SequentialTest(config)
        for value in (True, False, False, False, False, False, False, False):
            test.add(value)
        self.assertTrue(math.isfinite(test.llr))
        json.dumps(test.snapshot(), allow_nan=False)
        json.dumps(fixed_sample_verdict(1, 0, config), allow_nan=False)

    def test_extreme_integer_parameter_is_a_validation_error(self):
        with self.assertRaises(ValueError):
            SequentialConfig(alpha=10 ** 400)


if __name__ == "__main__":
    unittest.main()
