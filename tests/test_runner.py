import copy
import json
import threading
import time
import unittest
from dataclasses import replace

from model_sentinel.api import APIError, APIResponse
from model_sentinel.pack import load_pack
from model_sentinel.runner import adaptive_check, compare_runs, run_full, validate_baseline
from model_sentinel.sequential import SequentialConfig


def correct_answer(case):
    grader = case["grader"]
    if grader["type"] == "exact":
        return str(grader["answers"][0])
    return json.dumps(grader["value"], ensure_ascii=False)


class FakeClient:
    model = "fake-model"

    def __init__(self, pack, overrides=None):
        self.answers = {case["prompt"]: correct_answer(case) for case in pack["cases"]}
        self.overrides = {key: list(value) for key, value in (overrides or {}).items()}
        self.calls = 0
        self.lock = threading.Lock()

    def complete(self, prompt, max_tokens):
        with self.lock:
            self.calls += 1
            queued = self.overrides.get(prompt, [])
            output = queued.pop(0) if queued else self.answers[prompt]
        if isinstance(output, Exception):
            raise output
        if isinstance(output, APIResponse):
            return output
        return APIResponse(
            text=output,
            usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            latency_ms=1,
        )


class RunnerTests(unittest.TestCase):
    def test_programmatic_invalid_pack_is_rejected_before_requests(self):
        pack = load_pack()
        client = FakeClient(pack)
        pack["cases"][0]["grader"] = {"type": "json"}
        with self.assertRaises(ValueError):
            run_full(pack, client)
        self.assertEqual(client.calls, 0)

    @classmethod
    def setUpClass(cls):
        cls.pack = load_pack()
        cls.baseline = run_full(cls.pack, FakeClient(cls.pack))

    def test_clean_stratified_check_is_reproducible_and_covers_categories(self):
        client = FakeClient(self.pack)
        report = adaptive_check(self.pack, self.baseline, client, sample_seed=42)
        self.assertEqual(report["verdict"]["decision"], "stable")
        self.assertEqual(report["verdict"]["samples"], 12)
        self.assertEqual(client.calls, 12)
        self.assertEqual(report["usage"]["total_tokens"], 132)
        self.assertTrue(report["coverage"]["category_coverage_met"])
        again = adaptive_check(self.pack, self.baseline, FakeClient(self.pack), sample_seed=42)
        self.assertEqual(report["sampling"], again["sampling"])
        self.assertEqual(report["parameters"]["max_samples"], 32)

    def test_repeated_baseline_keeps_only_consistent_passes(self):
        first = self.pack["cases"][0]
        client = FakeClient(self.pack, {first["prompt"]: [correct_answer(first), "wrong", correct_answer(first)]})
        report = run_full(self.pack, client, limit=8, repeats=3)
        self.assertFalse(report["results"][0]["passed"])
        self.assertAlmostEqual(report["results"][0]["pass_rate"], 2 / 3)
        self.assertEqual(report["summary"]["repeats"], 3)
        self.assertEqual(client.calls, 24)
        validate_baseline(self.pack, report)

    def test_mixed_success_and_error_is_not_a_capability_failure(self):
        first = self.pack["cases"][0]
        report = run_full(self.pack, FakeClient(self.pack, {
            first["prompt"]: [correct_answer(first), APIError("offline"), correct_answer(first)]
        }), repeats=3)
        self.assertIsNone(report["results"][0]["passed"])
        self.assertEqual(report["summary"]["failed"], 0)
        self.assertEqual(report["summary"]["errors"], 1)
        self.assertEqual(report["summary"]["errored_probes"], 1)
        validate_baseline(self.pack, report)

    def test_confirmed_early_failures_trigger_regression(self):
        overrides = {case["prompt"]: ["wrong", "still wrong"] for case in self.pack["cases"][:2]}
        client = FakeClient(self.pack, overrides)
        report = adaptive_check(self.pack, self.baseline, client, sampling="ordered")
        self.assertEqual(report["verdict"]["decision"], "regressed")
        self.assertEqual(report["verdict"]["samples"], 8)
        self.assertEqual(report["verdict"]["confirmed_regressions"], 2)
        self.assertEqual(client.calls, 10)

    def test_online_and_offline_agree_on_transient_failures(self):
        overrides = {case["prompt"]: ["wrong"] for case in self.pack["cases"][:2]}
        online = adaptive_check(self.pack, self.baseline, FakeClient(self.pack, overrides), sampling="ordered")
        candidate = run_full(self.pack, FakeClient(self.pack, overrides), repeats=2)
        offline = compare_runs(self.pack, self.baseline, candidate)
        for report in (online, offline):
            self.assertEqual(report["verdict"]["decision"], "stable")
            self.assertEqual(report["verdict"]["confirmed_regressions"], 0)
            self.assertEqual(report["flaky_probes"], 2)

    def test_full_comparison_includes_failures_after_clean_prefix(self):
        overrides = {case["prompt"]: ["wrong", "wrong"] for case in self.pack["cases"][12:]}
        candidate = run_full(self.pack, FakeClient(self.pack, overrides), repeats=2)
        report = compare_runs(self.pack, self.baseline, candidate, config=SequentialConfig(max_samples=12))
        self.assertEqual(report["verdict"]["decision"], "regressed")
        self.assertEqual(report["verdict"]["samples"], 32)
        self.assertEqual(report["summary"]["regressions"], 20)
        replay = compare_runs(self.pack, self.baseline, candidate, replay=True)
        self.assertEqual(replay["capability_verdict"]["decision"], "stable")
        self.assertEqual(replay["full_verdict"]["decision"], "regressed")
        self.assertEqual(replay["summary"]["regressions"], 20)
        self.assertEqual(replay["replay"]["ignored_observations"], 20)

    def test_missing_confirmation_is_unresolved_and_blocks_stable(self):
        overrides = {case["prompt"]: ["wrong"] for case in self.pack["cases"][:20]}
        candidate = run_full(self.pack, FakeClient(self.pack, overrides))
        report = compare_runs(self.pack, self.baseline, candidate)
        self.assertEqual(report["capability_verdict"]["decision"], "stable")
        self.assertEqual(report["verdict"]["decision"], "inconclusive")
        self.assertEqual(report["coverage"]["unresolved"], 20)
        self.assertEqual(report["results"][0]["reason"], "missing_confirmation")
        unconfirmed = compare_runs(self.pack, self.baseline, candidate, confirm_failures=False)
        self.assertEqual(unconfirmed["verdict"]["decision"], "regressed")

    def test_missing_candidate_probes_block_stable(self):
        candidate = run_full(self.pack, FakeClient(self.pack), limit=12)
        report = compare_runs(self.pack, self.baseline, candidate)
        self.assertEqual(report["capability_verdict"]["decision"], "stable")
        self.assertEqual(report["verdict"]["decision"], "inconclusive")
        self.assertEqual(report["coverage"]["missing"], 20)
        self.assertAlmostEqual(report["coverage"]["valid_rate"], 12 / 32)

    def test_api_errors_are_reported_and_gate_an_otherwise_clean_check(self):
        overrides = {case["prompt"]: [APIError("offline")] for case in self.pack["cases"][:20]}
        report = adaptive_check(self.pack, self.baseline, FakeClient(self.pack, overrides), sampling="ordered")
        self.assertEqual(report["capability_verdict"]["decision"], "stable")
        self.assertEqual(report["verdict"]["decision"], "inconclusive")
        self.assertEqual(report["summary"]["errors"], 20)
        self.assertEqual(report["availability"]["status"], "degraded")
        self.assertEqual(report["coverage"]["counted"], 12)

    def test_unreachable_category_coverage_returns_inconclusive(self):
        overrides = {case["prompt"]: ["wrong"] for case in self.pack["cases"] if case["category"] == "structured"}
        baseline = run_full(self.pack, FakeClient(self.pack, overrides))
        client = FakeClient(self.pack)
        report = adaptive_check(self.pack, baseline, client, sample_seed=2)
        self.assertEqual(client.calls, 0)
        self.assertIn("insufficient_healthy_baseline_category_coverage", report["preflight_issues"])
        self.assertEqual(report["verdict"]["decision"], "inconclusive")
        self.assertFalse(report["coverage"]["category_coverage_met"])

    def test_category_quota_delays_early_stop(self):
        report = adaptive_check(self.pack, self.baseline, FakeClient(self.pack), sample_seed=3, min_category_samples=3)
        self.assertEqual(report["verdict"]["decision"], "stable")
        self.assertEqual(report["verdict"]["samples"], 15)
        self.assertEqual(report["parameters"]["effective_min_samples"], 15)
        self.assertTrue(report["coverage"]["category_coverage_met"])

    def test_impossible_category_budget_fails_before_requests(self):
        client = FakeClient(self.pack)
        with self.assertRaisesRegex(ValueError, "max_samples"):
            adaptive_check(self.pack, self.baseline, client, min_category_samples=3, config=SequentialConfig(max_samples=12))
        self.assertEqual(client.calls, 0)

    def test_compare_preserves_cost_and_errors_on_ineligible_probes(self):
        first = self.pack["cases"][0]
        baseline = run_full(self.pack, FakeClient(self.pack, {first["prompt"]: ["wrong"]}))
        candidate = run_full(self.pack, FakeClient(self.pack, {first["prompt"]: [APIError("offline")]}))
        report = compare_runs(self.pack, baseline, candidate)
        self.assertEqual(report["summary"]["errors"], 1)
        self.assertEqual(report["availability"]["errors"], 1)
        self.assertEqual(report["comparison_summary"]["errors"], 0)
        self.assertEqual(report["usage"], candidate["usage"])
        self.assertEqual(report["coverage"]["eligible"], 31)

    def test_no_healthy_baseline_probes_makes_no_requests(self):
        overrides = {case["prompt"]: ["wrong"] for case in self.pack["cases"]}
        baseline = run_full(self.pack, FakeClient(self.pack, overrides))
        client = FakeClient(self.pack)
        report = adaptive_check(self.pack, baseline, client)
        self.assertEqual(client.calls, 0)
        self.assertEqual(report["verdict"]["decision"], "inconclusive")

    def test_token_budget_mismatch_rejected_before_call_and_override_recorded(self):
        client = FakeClient(self.pack)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            adaptive_check(self.pack, self.baseline, client, max_tokens=256)
        self.assertEqual(client.calls, 0)
        report = adaptive_check(self.pack, self.baseline, client, max_tokens=256, allow_config_mismatch=True, sample_seed=1)
        self.assertIn("max_tokens_by_case", report["config_mismatches"])
        self.assertEqual(report["configuration"]["max_tokens_by_case"]["arith-01"], 256)

    def test_model_mismatch_requires_explicit_override(self):
        client = FakeClient(self.pack)
        client.model = "candidate-model"
        candidate = run_full(self.pack, client)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            compare_runs(self.pack, self.baseline, candidate)
        report = compare_runs(self.pack, self.baseline, candidate, allow_config_mismatch=True)
        self.assertEqual(report["config_mismatches"]["model"]["current"], "candidate-model")

    def test_smoke_and_legacy_reports_are_not_accepted_as_baselines(self):
        smoke = run_full(self.pack, FakeClient(self.pack), mode="smoke", limit=12)
        with self.assertRaisesRegex(ValueError, "mode"):
            validate_baseline(self.pack, smoke)
        legacy = copy.deepcopy(self.baseline)
        legacy["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "schema_version"):
            validate_baseline(self.pack, legacy)

    def test_fatal_error_stops_full_run_and_preserves_partial_attempts(self):
        case = self.pack["cases"][0]
        client = FakeClient(self.pack, {case["prompt"]: [correct_answer(case), APIError("denied", fatal=True)]})
        report = run_full(self.pack, client, repeats=3)
        self.assertEqual(client.calls, 2)
        self.assertTrue(report["aborted"])
        self.assertEqual(len(report["results"]), 1)
        self.assertIsNone(report["results"][0]["passed"])
        self.assertEqual(report["summary"]["errors"], 1)
        validate_baseline(self.pack, report)

    def test_fatal_error_stops_adaptive_check(self):
        case = self.pack["cases"][0]
        client = FakeClient(self.pack, {case["prompt"]: [APIError("budget exhausted", kind="budget_exhausted", fatal=True)]})
        report = adaptive_check(self.pack, self.baseline, client, sampling="ordered")
        self.assertEqual(client.calls, 1)
        self.assertTrue(report["aborted"])
        self.assertEqual(report["verdict"]["decision"], "inconclusive")

    def test_fatal_parallel_error_stops_submitting_new_cases(self):
        class FatalClient(FakeClient):
            def complete(self, prompt, max_tokens):
                if prompt != self.pack_first:
                    time.sleep(0.02)
                return super().complete(prompt, max_tokens)

        case = self.pack["cases"][0]
        client = FatalClient(self.pack, {case["prompt"]: [APIError("denied", fatal=True)]})
        client.pack_first = case["prompt"]
        report = run_full(self.pack, client, workers=3)
        self.assertTrue(report["aborted"])
        self.assertLessEqual(client.calls, 3)
        self.assertLessEqual(len(report["results"]), 3)
        validate_baseline(self.pack, report)

    def test_truncation_refusal_and_missing_finish_reason_are_errors(self):
        case = self.pack["cases"][0]
        good = APIResponse(text=correct_answer(case), usage={"completion_tokens": 2}, latency_ms=1)
        for response, status in (
            (replace(good, finish_reason="length"), "truncated"),
            (replace(good, refusal="no"), "refused"),
            (replace(good, finish_reason=None), "protocol_error"),
        ):
            with self.subTest(status=status):
                report = run_full(self.pack, FakeClient(self.pack, {case["prompt"]: [response]}), limit=1)
                self.assertIsNone(report["results"][0]["passed"])
                self.assertEqual(report["results"][0]["attempts"][0]["status"], status)
                self.assertEqual(report["usage"]["completion_tokens"], 2)
                validate_baseline(self.pack, report)

    def test_usage_preserves_nested_details(self):
        case = self.pack["cases"][0]
        response = APIResponse(text=correct_answer(case), usage={"completion_tokens": 4, "completion_tokens_details": {"reasoning_tokens": 3}}, latency_ms=1)
        report = run_full(self.pack, FakeClient(self.pack, {case["prompt"]: [response, response]}), limit=1, repeats=2)
        self.assertEqual(report["usage"]["completion_tokens_details"]["reasoning_tokens"], 6)
        validate_baseline(self.pack, report)

    def test_workers_are_bounded_and_results_keep_pack_order(self):
        class ConcurrentClient(FakeClient):
            def __init__(self, pack):
                super().__init__(pack)
                self.active = 0
                self.peak = 0

            def complete(self, prompt, max_tokens):
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                time.sleep(0.005)
                try:
                    return super().complete(prompt, max_tokens)
                finally:
                    with self.lock:
                        self.active -= 1

        client = ConcurrentClient(self.pack)
        report = run_full(self.pack, client, workers=3, repeats=2)
        self.assertGreater(client.peak, 1)
        self.assertLessEqual(client.peak, 3)
        self.assertEqual(client.calls, 64)
        self.assertEqual([item["id"] for item in report["results"]], [case["id"] for case in self.pack["cases"]])
        validate_baseline(self.pack, report)

    def test_invalid_run_options_fail_before_requests(self):
        for options in ({"workers": 0}, {"workers": True}, {"workers": 33}, {"repeats": 0}, {"limit": 0}, {"max_tokens": 0}):
            client = FakeClient(self.pack)
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_full(self.pack, client, **options)
            self.assertEqual(client.calls, 0)


if __name__ == "__main__":
    unittest.main()
