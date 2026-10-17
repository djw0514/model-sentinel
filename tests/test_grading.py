import unittest

from model_sentinel.grading import GRADER_VERSION, grade, normalize_text, strict_json_loads


class GradingTests(unittest.TestCase):
    def test_exact_normalizes_case_width_and_whitespace(self):
        self.assertEqual(normalize_text("  Ｂ,Ｃ\n"), "b,c")
        passed, normalized = grade("  NIA\n", {"type": "exact", "answers": ["Nia"]})
        self.assertTrue(passed)
        self.assertEqual(normalized, "nia")

    def test_exact_rejects_explanation(self):
        passed, _ = grade("The answer is 3", {"type": "exact", "answers": ["3"]})
        self.assertFalse(passed)

    def test_json_ignores_key_order_and_spacing(self):
        passed, normalized = grade(
            '{ "y": "blue", "x": 7 }',
            {"type": "json", "value": {"x": 7, "y": "blue"}},
        )
        self.assertTrue(passed)
        self.assertEqual(normalized, '{"x":7,"y":"blue"}')

    def test_invalid_json_fails(self):
        passed, _ = grade("x=7", {"type": "json", "value": {"x": 7}})
        self.assertFalse(passed)

    def test_json_boolean_and_number_are_distinct_at_every_depth(self):
        cases = [
            ("0", False),
            ("true", 1),
            ('{"id":"09","active":0}', {"id": "09", "active": False}),
            ('{"b":[2,true],"a":1}', {"b": [2, 1], "a": True}),
            ('{"a":[{"value":false}]}', {"a": [{"value": 0}]}),
        ]
        for output, expected in cases:
            with self.subTest(output=output):
                self.assertFalse(grade(output, {"type": "json", "value": expected})[0])

    def test_json_numbers_share_one_category_but_strings_do_not(self):
        self.assertTrue(grade('{"x":[1.0,2]}', {"type": "json", "value": {"x": [1, 2.0]}})[0])
        self.assertFalse(grade('"1"', {"type": "json", "value": 1})[0])

    def test_json_arrays_and_objects_require_exact_structure(self):
        for output in ('{"a":[2,1]}', '{"a":[1,2,3]}', '{"a":[1,2],"extra":0}', '{}'):
            with self.subTest(output=output):
                self.assertFalse(grade(output, {"type": "json", "value": {"a": [1, 2]}})[0])

    def test_json_rejects_nonfinite_numbers_and_duplicate_fields(self):
        for output in ('{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e400}', '{"x":0,"x":1}'):
            with self.subTest(output=output):
                self.assertFalse(grade(output, {"type": "json", "value": {"x": 1}})[0])
        with self.assertRaisesRegex(ValueError, r"output\.nested\[0\]\.x.*duplicate"):
            strict_json_loads('{"nested":[{"x":0,"x":1}]}', path="output")

    def test_json_requires_value_and_supports_explicit_null(self):
        with self.assertRaisesRegex(ValueError, r"grader\.value"):
            grade("null", {"type": "json"})
        self.assertTrue(grade("null", {"type": "json", "value": None})[0])
        self.assertFalse(grade("false", {"type": "json", "value": None})[0])

    def test_exact_can_preserve_case(self):
        grader = {"type": "exact", "answers": ["Nia"], "case_sensitive": True}
        self.assertFalse(grade("NIA", grader)[0])
        self.assertTrue(grade("  Nia\n", grader)[0])

    def test_exact_can_preserve_all_whitespace(self):
        grader = {"type": "exact", "answers": ["a  b\n"], "normalize_whitespace": False}
        self.assertTrue(grade("a  b\n", grader)[0])
        for output in ("a b\n", "a  b", " a  b\n"):
            with self.subTest(output=output):
                self.assertFalse(grade(output, grader)[0])

    def test_exact_can_preserve_unicode_width(self):
        grader = {"type": "exact", "answers": ["ABC"], "normalize_unicode": False}
        self.assertFalse(grade("ＡＢＣ", grader)[0])
        self.assertTrue(grade("abc", grader)[0])

    def test_exact_options_can_make_comparison_literal(self):
        grader = {
            "type": "exact", "answers": ["Ａ b\n"], "case_sensitive": True,
            "normalize_whitespace": False, "normalize_unicode": False,
        }
        self.assertEqual(grade("Ａ b\n", grader), (True, "Ａ b\n"))
        self.assertFalse(grade("A b\n", grader)[0])

    def test_invalid_grader_declarations_raise_path_errors(self):
        cases = [
            ({"type": []}, r"grader\.type"),
            ({"type": "exact", "answers": [1]}, r"grader\.answers\[0\]"),
            ({"type": "exact", "answers": [" "]}, r"grader\.answers\[0\]"),
            ({"type": "exact", "answers": ["a"], "case_sensitive": 1}, r"grader\.case_sensitive"),
            ({"type": "json", "value": {"x": float("inf")}}, r"grader\.value\.x"),
        ]
        for grader, path in cases:
            with self.subTest(grader=grader):
                with self.assertRaisesRegex(ValueError, path):
                    grade("a", grader)

    def test_grader_version_marks_changed_json_semantics(self):
        self.assertEqual(GRADER_VERSION, "2")


if __name__ == "__main__":
    unittest.main()
