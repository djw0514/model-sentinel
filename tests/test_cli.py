import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model_sentinel.api import APIError, APIResponse
from model_sentinel.cli import _client, build_parser, main
from model_sentinel.pack import load_pack
from model_sentinel.runner import run_full, save_report


class LocalClient:
    model = "test-model"

    def __init__(self, pack, *, wrong_after=None, error=False):
        self.calls = 0
        self.error = error
        self.answers = {}
        for index, case in enumerate(pack["cases"]):
            grader = case["grader"]
            answer = (grader["answers"][0] if grader["type"] == "exact"
                      else json.dumps(grader["value"]))
            self.answers[case["prompt"]] = "wrong" if wrong_after is not None and index >= wrong_after else answer

    def complete(self, prompt, max_tokens):
        self.calls += 1
        if self.error:
            raise APIError("offline failure", fatal=True)
        return APIResponse(self.answers[prompt], {"total_tokens": 10}, 1)


class CLITests(unittest.TestCase):
    def setUp(self):
        self.pack = load_pack()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.baseline_path = Path(self.temp.name) / "baseline.json"
        self.output_path = Path(self.temp.name) / "output.json"
        save_report(self.baseline_path, run_full(self.pack, LocalClient(self.pack)))

    def invoke(self, args, client=None):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if client is None:
                return main(args)
            with patch("model_sentinel.cli._client", return_value=client):
                return main(args)

    def check_args(self, mode="standard"):
        return ["check", "--mode", mode, "--baseline", str(self.baseline_path),
                "--output", str(self.output_path)]

    def test_standard_mode_saves_reproducible_verdict(self):
        code = self.invoke(self.check_args() + ["--sample-seed", "42"], LocalClient(self.pack))
        self.assertEqual(code, 0)
        report = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(report["verdict"]["decision"], "stable")
        self.assertEqual(report["sampling"]["seed"], 42)
        self.assertEqual(report["schema_version"], 2)

    def test_quick_mode_has_smoke_verdict_and_no_baseline(self):
        code = self.invoke(["check", "--mode", "quick", "--output", str(self.output_path)],
                           LocalClient(self.pack))
        self.assertEqual(code, 0)
        report = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(report["mode"], "smoke")
        self.assertEqual(report["smoke_verdict"], "smoke_pass")
        self.assertNotIn("verdict", report)

    def test_deep_mode_detects_late_failures_and_keeps_raw_results(self):
        client = LocalClient(self.pack, wrong_after=12)
        self.assertEqual(self.invoke(self.check_args("deep"), client), 2)
        report = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(report["verdict"]["decision"], "regressed")
        self.assertEqual(report["verdict"]["samples"], 32)
        self.assertEqual(len(report["candidate_run"]["results"]), 32)
        self.assertEqual(client.calls, 64)

    def test_api_failure_saves_report_and_returns_execution_error(self):
        code = self.invoke(self.check_args(), LocalClient(self.pack, error=True))
        self.assertEqual(code, 1)
        report = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertTrue(report["aborted"])
        self.assertEqual(report["verdict"]["decision"], "inconclusive")

    def test_deep_rejects_unusable_baseline_before_requests(self):
        save_report(self.baseline_path, run_full(self.pack, LocalClient(self.pack, wrong_after=0)))
        client = LocalClient(self.pack)
        self.assertEqual(self.invoke(self.check_args("deep"), client), 1)
        self.assertEqual(client.calls, 0)

    def test_missing_baseline_and_invalid_arguments_do_not_send_requests(self):
        client = LocalClient(self.pack)
        cases = [
            ["check", "--output", str(self.output_path)],
            self.check_args() + ["--max-error-rate", "nan"],
            self.check_args() + ["--workers", "2"],
            self.check_args("deep") + ["--repeats", "1"],
            ["check", "--not-a-real-option"],
        ]
        for args in cases:
            with self.subTest(args=args):
                self.assertEqual(self.invoke(args, client), 1)
        self.assertEqual(client.calls, 0)

    def test_invalid_request_parameters_fail_before_network(self):
        for option, value in (("--timeout", "-1"), ("--retries", "-1"),
                              ("--temperature", "nan"), ("--max-tokens", "0"),
                              ("--deadline", "inf"), ("--max-requests", "0")):
            with self.subTest(option=option), patch("model_sentinel.api.ChatCompletionsClient.complete") as call:
                code = self.invoke(["run", "--model", "test-model", "--output", str(self.output_path), option, value])
                self.assertEqual(code, 1)
                call.assert_not_called()

    def test_explicit_api_key_variable_takes_precedence(self):
        parser = build_parser()
        args = parser.parse_args(["run", "--model", "test-model", "--output", "unused.json",
                                  "--api-key-env", "EXPLICIT_KEY"])
        with patch.dict(os.environ, {"MODEL_SENTINEL_API_KEY": "default-test-key",
                                     "EXPLICIT_KEY": "explicit-test-key"}, clear=True):
            self.assertEqual(_client(args, parser).api_key, "explicit-test-key")
        with patch.dict(os.environ, {"MODEL_SENTINEL_API_KEY": "default-test-key"}, clear=True):
            with self.assertRaisesRegex(ValueError, "empty or missing"):
                _client(args, parser)

    def test_malformed_pack_has_readable_error_exit(self):
        path = Path(self.temp.name) / "bad.json"
        path.write_text("[]", encoding="utf-8")
        self.assertEqual(self.invoke(["validate-pack", str(path)]), 1)

    def test_calibrate_uses_saved_labeled_runs_without_model(self):
        manifest = Path(self.temp.name) / "labels.json"
        manifest.write_text(json.dumps({"cases": [{"baseline": "baseline.json",
            "candidate": "baseline.json", "expected": "stable"}]}), encoding="utf-8")
        with patch("model_sentinel.cli._client") as client:
            code = self.invoke(["calibrate", str(manifest), "--output", str(self.output_path)])
        client.assert_not_called()
        self.assertEqual(code, 0)
        report = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["false_alarm_rate"], 0)
        self.assertIsNone(report["miss_rate"])


if __name__ == "__main__":
    unittest.main()
