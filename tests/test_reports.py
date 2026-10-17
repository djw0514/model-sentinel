import copy
import tempfile
import unittest
from pathlib import Path

from model_sentinel.api import ChatCompletionsClient
from model_sentinel.pack import load_pack
from model_sentinel.reports import fingerprint, observation, usage_total, validate_run_report
from model_sentinel.runner import load_report, run_full, save_report
from test_runner import FakeClient


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = load_pack()
        cls.report = run_full(cls.pack, FakeClient(cls.pack))

    def test_result_and_attempt_shapes_are_validated(self):
        changes = [
            (lambda r: r.update(results={})),
            (lambda r: r["results"].append(copy.deepcopy(r["results"][0]))),
            (lambda r: r["results"][0].update(id="unknown")),
            (lambda r: r["results"][0].update(category="different")),
            (lambda r: r["results"][0].update(passed=1)),
            (lambda r: r["results"][0].update(passed=False)),
            (lambda r: r["results"][0].update(attempts=[])),
            (lambda r: r["results"][0]["attempts"][0].update(passed=1)),
            (lambda r: r["results"][0]["attempts"][0].update(passed=None)),
            (lambda r: r["results"][0]["attempts"][0].update(error="inconsistent")),
            (lambda r: r["results"][0]["attempts"][0].update(output="wrong")),
            (lambda r: r["results"][0]["attempts"][0].update(finish_reason="length")),
        ]
        for index, change in enumerate(changes):
            broken = copy.deepcopy(self.report)
            change(broken)
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_run_report(self.pack, broken)

    def test_metadata_summary_and_coverage_are_checked(self):
        changes = [
            lambda r: r.update(schema_version=True),
            lambda r: r.update(mode="codex_cli_batch_smoke"),
            lambda r: r["pack"].update(digest="different"),
            lambda r: r.update(configuration_fingerprint="bad"),
            lambda r: r["configuration"].update(grader_version="1"),
            lambda r: r["summary"].update(errors=20),
            lambda r: r["summary"].update(repeats=True),
            lambda r: r["coverage"].update(selected_ids=["arith-01", "arith-01"]),
            lambda r: r["coverage"].update(observed=0),
            lambda r: r["coverage"].update(complete=False),
            lambda r: r.update(aborted=True),
            lambda r: r["usage"].update(total_tokens=-1),
        ]
        for index, change in enumerate(changes):
            broken = copy.deepcopy(self.report)
            change(broken)
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_run_report(self.pack, broken)

    def test_chat_configuration_requires_all_settings_even_with_valid_fingerprint(self):
        client = ChatCompletionsClient(base_url="https://example.test/v1", api_key=None, model="fake-model")
        report = copy.deepcopy(self.report)
        report["configuration"].update(client.configuration())
        report["configuration_fingerprint"] = fingerprint(report["configuration"])
        validate_run_report(self.pack, report)
        for key in ("endpoint", "system_prompt_digest", "temperature", "seed", "token_param", "reasoning_effort"):
            broken = copy.deepcopy(report)
            del broken["configuration"][key]
            broken["configuration_fingerprint"] = fingerprint(broken["configuration"])
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "settings"):
                validate_run_report(self.pack, broken)

    def test_malformed_usage_fails_cleanly_and_large_integers_are_supported(self):
        for value in (True, -1, float("nan"), float("inf"), "10"):
            broken = copy.deepcopy(self.report)
            broken["results"][0]["attempts"][0]["usage"]["prompt_tokens"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_run_report(self.pack, broken)
        large = 10 ** 400
        report = copy.deepcopy(self.report)
        report["results"][0]["attempts"][0]["usage"]["prompt_tokens"] = large
        report["usage"] = usage_total(report["results"])
        validate_run_report(self.pack, report)
        self.assertEqual(report["usage"]["prompt_tokens"], large + 310)

    def test_configuration_rejects_wrong_types_without_runtime_exceptions(self):
        client = ChatCompletionsClient(base_url="https://example.test/v1", api_key=None, model="fake-model")
        for key, value in (("token_param", []), ("endpoint", []), ("seed", True), ("temperature", 10 ** 400), ("reasoning_effort", {})):
            report = copy.deepcopy(self.report)
            report["configuration"].update(client.configuration())
            report["configuration"][key] = value
            report["configuration_fingerprint"] = fingerprint(report["configuration"])
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_run_report(self.pack, report)

    def test_load_rejects_duplicate_keys_nonfinite_numbers_and_nonobject_root(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            for content in ('{"schema_version": 2, "schema_version": 1}', '{"usage": NaN}', '[]'):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_report(path)

    def test_save_roundtrip_and_invalid_output_preserves_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            save_report(path, self.report)
            self.assertEqual(load_report(path), self.report)
            with self.assertRaises(ValueError):
                save_report(path, {"invalid": float("nan")})
            self.assertEqual(load_report(path), self.report)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_observation_uses_the_same_first_two_attempts_for_both_modes(self):
        cases = [
            ([True, False], True, False, False),
            ([False, True], True, False, True),
            ([False, False], True, True, False),
            ([False], False, None, None),
            ([None, True], False, None, None),
            ([False, None], False, None, None),
            ([], False, None, None),
        ]
        for values, counted, regression, flaky in cases:
            result = observation([{"passed": value} for value in values], True)
            with self.subTest(values=values):
                self.assertEqual(result["counted"], counted)
                self.assertEqual(result.get("regressed"), regression)
                self.assertEqual(result.get("flaky"), flaky)


if __name__ == "__main__":
    unittest.main()
