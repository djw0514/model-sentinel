"""Deterministic graders that do not consume model tokens."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any


GRADER_VERSION = "2"
EXACT_OPTIONS = {
    "case_sensitive": False,
    "normalize_whitespace": True,
    "normalize_unicode": True,
}


class _ObjectPairs(list):
    """Preserve duplicate JSON fields until their full path is known."""


def _field_path(path: str, key: str) -> str:
    return f"{path}.{key}" if key.isidentifier() else f"{path}[{json.dumps(key)}]"


def validate_json_value(value: Any, path: str = "$") -> None:
    """Reject non-JSON Python values, cycles, and non-finite numbers."""
    ancestors: set[int] = set()

    def visit(item: Any, location: str) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError(f"{location} must be a finite JSON number")
            return
        if type(item) not in (list, dict):
            raise ValueError(f"{location} must be a JSON value")
        identity = id(item)
        if identity in ancestors:
            raise ValueError(f"{location} must not contain a circular reference")
        ancestors.add(identity)
        try:
            if isinstance(item, list):
                for index, child in enumerate(item):
                    visit(child, f"{location}[{index}]")
            else:
                for key, child in item.items():
                    if not isinstance(key, str):
                        raise ValueError(f"{location} object keys must be strings")
                    visit(child, _field_path(location, key))
        finally:
            ancestors.remove(identity)

    try:
        visit(value, path)
    except RecursionError as exc:
        raise ValueError(f"{path} exceeds the supported JSON nesting depth") from exc


def strict_json_loads(text: str, *, path: str = "$") -> Any:
    """Read JSON without silently discarding duplicate object fields."""
    def convert(value: Any, location: str) -> Any:
        if isinstance(value, _ObjectPairs):
            result: dict[str, Any] = {}
            for key, child in value:
                child_path = _field_path(location, key)
                if key in result:
                    raise ValueError(f"{child_path} is a duplicate JSON field")
                result[key] = convert(child, child_path)
            return result
        if isinstance(value, list):
            return [convert(child, f"{location}[{index}]") for index, child in enumerate(value)]
        return value

    try:
        decoded = json.loads(text, object_pairs_hook=_ObjectPairs)
        result = convert(decoded, path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    except ValueError as exc:
        raise ValueError(f"{path}: invalid JSON ({exc})") from exc
    except (TypeError, RecursionError) as exc:
        raise ValueError(f"{path}: invalid JSON input ({exc})") from exc
    validate_json_value(result, path)
    return result


def validate_grader(grader: Any, path: str = "grader") -> None:
    """Validate a complete grader declaration before scoring or API calls."""
    if not isinstance(grader, dict):
        raise ValueError(f"{path} must be an object")
    kind = grader.get("type")
    if not isinstance(kind, str) or kind not in {"exact", "json"}:
        raise ValueError(f"{path}.type must be 'exact' or 'json'")
    allowed = {"type", "answers", *EXACT_OPTIONS} if kind == "exact" else {"type", "value"}
    for key in grader:
        if key not in allowed:
            raise ValueError(f"{path}.{key} is not a supported {kind} grader field")
    if kind == "exact":
        answers = grader.get("answers")
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"{path}.answers must be a non-empty list")
        for index, answer in enumerate(answers):
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"{path}.answers[{index}] must be a non-empty string")
        for option in EXACT_OPTIONS:
            if option in grader and not isinstance(grader[option], bool):
                raise ValueError(f"{path}.{option} must be a boolean")
    else:
        if "value" not in grader:
            raise ValueError(f"{path}.value is required (null is allowed)")
        validate_json_value(grader["value"], f"{path}.value")


def normalize_text(
    value: str,
    *,
    case_sensitive: bool = False,
    normalize_whitespace: bool = True,
    normalize_unicode: bool = True,
) -> str:
    """Apply explicit exact-grader rules; disabled whitespace rules preserve all spaces."""
    if normalize_unicode:
        value = unicodedata.normalize("NFKC", value)
    if not case_sensitive:
        value = value.casefold()
    if normalize_whitespace:
        value = re.sub(r"\s+", " ", value.strip())
    return value


def _json_equal(actual: Any, expected: Any) -> bool:
    # JSON has one number category, but booleans must not compare equal to 0/1.
    if type(actual) in (int, float) and type(expected) in (int, float):
        return actual == expected
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            _json_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _json_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def grade(output: str, grader: dict[str, Any]) -> tuple[bool, str]:
    """Return (passed, normalized_output) for a validated grader declaration."""
    validate_grader(grader)
    if not isinstance(output, str):
        raise ValueError("output must be a string")
    if grader["type"] == "exact":
        options = {key: grader.get(key, default) for key, default in EXACT_OPTIONS.items()}
        normalized = normalize_text(output, **options)
        expected = {normalize_text(answer, **options) for answer in grader["answers"]}
        return normalized in expected, normalized

    try:
        actual = strict_json_loads(output, path="output")
    except ValueError:
        return False, normalize_text(output)
    normalized = json.dumps(actual, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _json_equal(actual, grader["value"]), normalized
