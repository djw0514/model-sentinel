import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model_sentinel.api import APIResponse
from model_sentinel.calibration import calibrate_manifest
from model_sentinel.pack import DEFAULT_PACK, load_pack
from model_sentinel.runner import compare_runs, load_report, run_full, save_report
from model_sentinel.sequential import SequentialConfig


class CorrectClient:
    model = "calibration-test"

    def __init__(self, pack):
        self.answers = {}
        for case in pack["cases"]:
            grader = case["grader"]
            self.answers[case["prompt"]] = grader["answers"][0] if grader["type"] == "exact" else json.dumps(grader["value"])

    def complete(self, prompt, max_tokens):
        return APIResponse(text=self.answers[prompt], usage={}, latency_ms=1)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = self.root / "manifest.json"
        for name in ("baseline.json", "candidate.json"):
            (self.root / name).write_text("{}", encoding="utf-8")

    def write(self, cases, **extra):
        self.manifest.write_text(json.dumps({"cases": cases, **extra}), encoding="utf-8")

    def case(self, expected="stable", **extra):
        return {"baseline": "baseline.json", "candidate": "candidate.json", "expected": expected, **extra}

    def test_rates_include_inconclusive_cases_in_each_class_denominator(self):
        self.write([self.case(label) for label in ("stable", "stable", "stable", "regressed", "regressed", "regressed")])
        decisions = ["stable", "regressed", "inconclusive", "stable", "regressed", "inconclusive"]
        with patch("model_sentinel.calibration.compare_runs", side_effect=[{"verdict": {"decision": decision, "samples": 12}} for decision in decisions]):
            report = calibrate_manifest(self.manifest)
        self.assertEqual(report["total"], 6)
        self.assertEqual(report["pack_path"], str(DEFAULT_PACK.resolve()))
        self.assertAlmostEqual(report["false_alarm_rate"], 1 / 3)
        self.assertAlmostEqual(report["miss_rate"], 1 / 3)
        self.assertAlmostEqual(report["inconclusive_rate"], 1 / 3)
        self.assertEqual(report["confusion"]["stable"], {"stable": 1, "regressed": 1, "inconclusive": 1})
        self.assertEqual([item["matched"] for item in report["results"]], [True, False, False, False, True, False])

    def test_missing_label_has_null_rate_and_parameters_are_forwarded(self):
        pack = load_pack()
        (self.root / "custom-pack.json").write_text(json.dumps(pack), encoding="utf-8")
        self.write([self.case(name="healthy")], pack="custom-pack.json")
        config = SequentialConfig(p1=0.3)
        with patch("model_sentinel.calibration.compare_runs", return_value={"verdict": {"decision": "stable", "samples": 12}}) as compare:
            report = calibrate_manifest(self.manifest, config=config, confirm_failures=False, replay=True, allow_config_mismatch=True, max_error_rate=0.1)
        self.assertIsNone(report["miss_rate"])
        self.assertEqual(report["false_alarm_rate"], 0)
        self.assertEqual(report["results"][0]["baseline"], str((self.root / "baseline.json").resolve()))
        self.assertEqual(compare.call_args.kwargs, {"config": config, "confirm_failures": False, "replay": True, "allow_config_mismatch": True, "max_error_rate": 0.1})
        self.assertEqual(report["parameters"]["p1"], 0.3)
        self.assertEqual(report["pack_path"], str((self.root / "custom-pack.json").resolve()))

    def test_manifest_shape_labels_duplicate_names_and_fields_are_rejected(self):
        malformed = [
            ([], "manifest must"), ({"cases": []}, r"manifest\.cases"),
            ({"cases": [None]}, r"manifest\.cases\[0\]"),
            ({"cases": [self.case(expected=[])]}, r"manifest\.cases\[0\]\.expected"),
            ({"cases": [self.case(name="a"), self.case(name="a")]}, r"manifest\.cases\[1\]\.name"),
            ({"cases": [self.case(typo=True)]}, r"manifest\.cases\[0\]\.typo"),
        ]
        for value, message in malformed:
            with self.subTest(value=value):
                self.manifest.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    calibrate_manifest(self.manifest)
        self.manifest.write_text('{"cases":[],"cases":[]}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, r"manifest\.cases.*duplicate"):
            calibrate_manifest(self.manifest)

    def test_url_and_network_paths_are_rejected_before_loading_reports(self):
        for value in ("https://example.invalid/data.json", "file:///tmp/data.json", "\\\\server\\share\\data.json", "//server/share/data.json"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "manifest.*local file"):
                    calibrate_manifest(value)
                self.write([self.case(candidate=value)])
                with patch("model_sentinel.calibration.load_report") as load:
                    with self.assertRaisesRegex(ValueError, r"manifest\.cases\[0\]\.candidate"):
                        calibrate_manifest(self.manifest)
                    load.assert_not_called()
        self.write([self.case()], pack="https://example.invalid/pack.json")
        with self.assertRaisesRegex(ValueError, r"manifest\.pack"):
            calibrate_manifest(self.manifest)

    def test_real_reports_are_validated_and_partial_data_remains_inconclusive(self):
        pack = load_pack()
        baseline = run_full(pack, CorrectClient(pack))
        candidate = run_full(pack, CorrectClient(pack), limit=1)
        save_report(self.root / "baseline.json", baseline)
        save_report(self.root / "candidate.json", candidate)
        self.write([self.case()])
        report = calibrate_manifest(self.manifest)
        self.assertEqual(report["results"][0]["decision"], "inconclusive")
        self.assertEqual(report["inconclusive_rate"], 1)
        self.assertEqual(report["false_alarm_rate"], 0)
        candidate["schema_version"] = 999
        save_report(self.root / "candidate.json", candidate)
        with self.assertRaisesRegex(ValueError, r"manifest\.cases\[0\].*schema_version"):
            calibrate_manifest(self.manifest)

    def test_invalid_options_and_missing_reports_have_actionable_paths(self):
        for options, field in (({"max_error_rate": float("nan")}, "max_error_rate"), ({"max_error_rate": 10**1000}, "max_error_rate"), ({"confirm_failures": 1}, "confirm_failures"), ({"config": {}}, "config")):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, f"parameters.{field}"):
                    calibrate_manifest(self.manifest, **options)
        self.write([self.case(candidate="missing.json")])
        with self.assertRaisesRegex(ValueError, r"manifest\.cases\[0\]\.candidate"):
            calibrate_manifest(self.manifest)

    def test_shared_reports_and_path_aliases_are_read_once_per_call(self):
        baseline = (self.root / "baseline.json").resolve()
        candidate = (self.root / "candidate.json").resolve()
        self.write([
            self.case(name="relative"),
            self.case(name="absolute", baseline=str(baseline), candidate=str(candidate)),
            self.case(name="parent", baseline="child/../baseline.json", candidate="./candidate.json"),
        ])
        with patch("model_sentinel.calibration.load_report", wraps=load_report) as load, patch(
            "model_sentinel.calibration.compare_runs",
            return_value={"verdict": {"decision": "stable", "samples": 12}},
        ) as compare:
            report = calibrate_manifest(self.manifest)
        self.assertEqual(Counter(call.args[0] for call in load.call_args_list), {baseline: 1, candidate: 1})
        self.assertEqual(compare.call_count, 3)
        self.assertEqual([item["name"] for item in report["results"]], ["relative", "absolute", "parent"])
        self.assertEqual(report["confusion"]["stable"]["stable"], 3)

    def test_report_cache_does_not_hide_changes_between_calibrations(self):
        candidate = self.root / "candidate.json"
        self.write([self.case(), self.case()])

        def compare(pack, baseline, current, **kwargs):
            return {"verdict": {"decision": current["decision"], "samples": 12}}

        with patch("model_sentinel.calibration.load_report", wraps=load_report) as load, patch(
            "model_sentinel.calibration.compare_runs", side_effect=compare,
        ):
            candidate.write_text('{"decision":"stable"}', encoding="utf-8")
            first = calibrate_manifest(self.manifest)
            candidate.write_text('{"decision":"regressed"}', encoding="utf-8")
            second = calibrate_manifest(self.manifest)
        self.assertEqual(load.call_count, 4)
        self.assertEqual(first["confusion"]["stable"]["stable"], 2)
        self.assertEqual(second["confusion"]["stable"]["regressed"], 2)

    def test_actual_comparisons_leave_cached_reports_unchanged_including_replay(self):
        baseline = self.root / "baseline.json"
        pack = load_pack()
        original = run_full(pack, CorrectClient(pack), repeats=3)
        save_report(baseline, original)
        self.write([self.case(candidate="./baseline.json"), self.case(candidate=str(baseline))])

        for replay in (False, True):
            with self.subTest(replay=replay):
                observed = []

                def compare(pack, old, current, **kwargs):
                    before = deepcopy(old)
                    self.assertIs(old, current)
                    result = compare_runs(pack, old, current, **kwargs)
                    self.assertEqual(old, before)
                    self.assertEqual(current, before)
                    observed.append(old)
                    return result

                with patch("model_sentinel.calibration.load_report", wraps=load_report) as load, patch(
                    "model_sentinel.calibration.compare_runs", side_effect=compare,
                ):
                    report = calibrate_manifest(self.manifest, replay=replay)
                self.assertEqual(load.call_count, 1)
                self.assertEqual(len(observed), 2)
                self.assertIs(observed[0], observed[1])
                self.assertEqual(observed[0], original)
                self.assertEqual(report["confusion"]["stable"]["stable"], 2)

    def test_cached_reports_preserve_later_case_load_and_comparison_error_paths(self):
        self.write([self.case(), self.case(candidate="missing.json")])
        stable = {"verdict": {"decision": "stable", "samples": 12}}
        with patch("model_sentinel.calibration.compare_runs", return_value=stable):
            with self.assertRaisesRegex(ValueError, r"manifest\.cases\[1\]\.candidate"):
                calibrate_manifest(self.manifest)
        self.write([self.case(), self.case()])
        with patch("model_sentinel.calibration.compare_runs", side_effect=[stable, ValueError("invalid report")]):
            with self.assertRaisesRegex(ValueError, r"manifest\.cases\[1\]: invalid report"):
                calibrate_manifest(self.manifest)


if __name__ == "__main__":
    unittest.main()
