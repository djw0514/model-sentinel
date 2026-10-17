import copy
import json
from pathlib import Path
import tempfile
import unittest

from model_sentinel.pack import load_pack, pack_digest, validate_pack


class PackTests(unittest.TestCase):
    def test_bundled_pack_is_valid_and_has_32_unique_cases(self):
        pack = load_pack()
        self.assertEqual(len(pack["cases"]), 32)
        self.assertEqual(len({case["id"] for case in pack["cases"]}), 32)
        self.assertEqual(len(pack_digest(pack)), 64)

    def test_duplicate_id_is_rejected(self):
        pack = load_pack()
        broken = copy.deepcopy(pack)
        broken["cases"][1]["id"] = broken["cases"][0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_pack(broken)

    def test_root_and_schema_version_have_strict_types(self):
        for root in (None, [], "pack", 1):
            with self.subTest(root=root):
                with self.assertRaisesRegex(ValueError, "pack must be an object"):
                    validate_pack(root)
        for version in (True, 1.0, "1", 2, None):
            with self.subTest(version=version):
                pack = load_pack()
                pack["schema_version"] = version
                with self.assertRaisesRegex(ValueError, r"pack\.schema_version"):
                    validate_pack(pack)

    def test_case_fields_reject_blank_strings_and_bad_types(self):
        for key in ("id", "category", "prompt"):
            for value in (None, "", " \n", 123):
                with self.subTest(key=key, value=value):
                    pack = load_pack()
                    pack["cases"][0][key] = value
                    with self.assertRaisesRegex(ValueError, rf"pack\.cases\[0\]\.{key}"):
                        validate_pack(pack)

    def test_metadata_and_case_container_are_validated(self):
        cases = [("name", " "), ("version", 1), ("description", []), ("cases", []), ("cases", [None])]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                pack = load_pack()
                pack[field] = value
                with self.assertRaisesRegex(ValueError, rf"pack\.{field}"):
                    validate_pack(pack)

    def test_json_grader_requires_value_but_null_is_allowed(self):
        pack = load_pack()
        pack["cases"][0]["grader"] = {"type": "json"}
        with self.assertRaisesRegex(ValueError, r"pack\.cases\[0\]\.grader\.value"):
            validate_pack(pack)
        pack["cases"][0]["grader"]["value"] = None
        validate_pack(pack)

    def test_grader_type_and_answers_are_validated(self):
        graders = [
            None, {"type": []}, {"type": "unknown"},
            {"type": "exact", "answers": []}, {"type": "exact", "answers": "yes"},
            {"type": "exact", "answers": [1]}, {"type": "exact", "answers": [None]},
            {"type": "exact", "answers": [" "]},
        ]
        for grader in graders:
            with self.subTest(grader=grader):
                pack = load_pack()
                pack["cases"][0]["grader"] = grader
                with self.assertRaisesRegex(ValueError, r"pack\.cases\[0\]\.grader"):
                    validate_pack(pack)

    def test_exact_option_values_must_be_booleans(self):
        for option in ("case_sensitive", "normalize_whitespace", "normalize_unicode"):
            for value in (True, False, "false", 0, None):
                with self.subTest(option=option, value=value):
                    pack = load_pack()
                    pack["cases"][0]["grader"][option] = value
                    if isinstance(value, bool):
                        validate_pack(pack)
                    else:
                        with self.assertRaisesRegex(ValueError, option):
                            validate_pack(pack)

    def test_token_cap_is_an_integer_excluding_boolean(self):
        for value in (True, False, 0, -1, 4097, 1.0, "8", None):
            with self.subTest(value=value):
                pack = load_pack()
                pack["cases"][0]["max_tokens"] = value
                with self.assertRaisesRegex(ValueError, r"pack\.cases\[0\]\.max_tokens"):
                    validate_pack(pack)
        pack = load_pack()
        del pack["cases"][0]["max_tokens"]
        validate_pack(pack)

    def test_nonfinite_and_non_json_grader_values_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), (1, 2), {1: "one"}):
            with self.subTest(value=value):
                pack = load_pack()
                pack["cases"][0]["grader"] = {"type": "json", "value": {"nested": [value]}}
                with self.assertRaisesRegex(ValueError, r"pack\.cases\[0\]\.grader\.value\.nested\[0\]"):
                    validate_pack(pack)

    def test_unknown_fields_are_rejected_including_option_typos(self):
        for location in ("pack", "case", "grader"):
            with self.subTest(location=location):
                pack = load_pack()
                target = {"pack": pack, "case": pack["cases"][0], "grader": pack["cases"][0]["grader"]}[location]
                target["typo"] = True
                with self.assertRaisesRegex(ValueError, "typo"):
                    validate_pack(pack)

    def test_load_rejects_duplicate_fields_and_nonfinite_json_with_paths(self):
        pack_text = json.dumps(load_pack())
        malformed = [
            (pack_text.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1', 1), r"pack\.schema_version.*duplicate"),
            (pack_text.replace('"type": "exact"', '"type": "exact", "type": "exact"', 1), r"pack\.cases\[0\]\.grader\.type.*duplicate"),
            (pack_text.replace('"max_tokens": 8', '"max_tokens": NaN', 1), r"pack\.cases\[0\]\.max_tokens"),
            (pack_text.replace('"max_tokens": 8', '"max_tokens": 1e400', 1), r"pack\.cases\[0\]\.max_tokens"),
            ('{"cases":', "pack: invalid JSON"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pack.json"
            for text, message in malformed:
                with self.subTest(message=message):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        load_pack(path)

    def test_digest_is_independent_of_object_key_order(self):
        pack = load_pack()
        reordered = dict(reversed(list(pack.items())))
        self.assertEqual(pack_digest(pack), pack_digest(reordered))


if __name__ == "__main__":
    unittest.main()
