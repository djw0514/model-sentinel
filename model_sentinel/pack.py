"""Probe-pack loading and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .grading import strict_json_loads, validate_grader, validate_json_value


DEFAULT_PACK = Path(__file__).resolve().parent / "packs" / "core_v1.json"
SUPPORTED_GRADERS = {"exact", "json"}


def load_pack(path: str | Path | None = None) -> dict[str, Any]:
    pack_path = Path(path) if path else DEFAULT_PACK
    pack = strict_json_loads(pack_path.read_text(encoding="utf-8"), path="pack")
    validate_pack(pack)
    return pack


def pack_digest(pack: dict[str, Any]) -> str:
    validate_pack(pack)
    canonical = json.dumps(pack, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _check_fields(value: dict[str, Any], allowed: set[str], path: str) -> None:
    for key in value:
        if key not in allowed:
            raise ValueError(f"{path}.{key} is not a supported field")


def _nonempty_string(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")


def validate_pack(pack: Any) -> None:
    if not isinstance(pack, dict):
        raise ValueError("pack must be an object")
    validate_json_value(pack, "pack")
    _check_fields(pack, {"schema_version", "name", "version", "description", "cases"}, "pack")
    if type(pack.get("schema_version")) is not int or pack["schema_version"] != 1:
        raise ValueError("pack.schema_version must be the integer 1")
    _nonempty_string(pack.get("name"), "pack.name")
    _nonempty_string(pack.get("version"), "pack.version")
    if "description" in pack and not isinstance(pack["description"], str):
        raise ValueError("pack.description must be a string")
    cases = pack.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("pack.cases must be a non-empty list")

    seen: set[str] = set()
    for index, case in enumerate(cases):
        prefix = f"pack.cases[{index}]"
        if not isinstance(case, dict):
            raise ValueError(f"{prefix} must be an object")
        _check_fields(case, {"id", "category", "prompt", "grader", "max_tokens"}, prefix)
        case_id = case.get("id")
        _nonempty_string(case_id, f"{prefix}.id")
        if case_id in seen:
            raise ValueError(f"{prefix}.id is a duplicate case id: {case_id}")
        seen.add(case_id)
        _nonempty_string(case.get("category"), f"{prefix}.category")
        _nonempty_string(case.get("prompt"), f"{prefix}.prompt")
        validate_grader(case.get("grader"), f"{prefix}.grader")
        max_tokens = case.get("max_tokens", 16)
        if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
            raise ValueError(f"{prefix}.max_tokens must be an integer in [1, 4096]")
