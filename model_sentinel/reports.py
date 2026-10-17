"""Versioned run reports, compatible evaluation settings, and observations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from .grading import GRADER_VERSION, grade, validate_json_value
from .pack import pack_digest


SCHEMA_VERSION = 2


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(configuration: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(configuration).encode("utf-8")).hexdigest()


def evaluation_configuration(client: Any, pack: dict[str, Any], max_tokens: int | None) -> dict[str, Any]:
    provider = getattr(client, "configuration", None)
    configuration = dict(provider()) if callable(provider) else {
        "surface": "custom_completion",
        "model": client.model,
    }
    configuration["grader_version"] = GRADER_VERSION
    configuration["max_tokens_by_case"] = {
        case["id"]: max_tokens if max_tokens is not None else case.get("max_tokens", 16)
        for case in pack["cases"]
    }
    return configuration


def metadata(pack: dict[str, Any], model: str, mode: str, configuration: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "pack": {"name": pack["name"], "version": pack["version"], "digest": pack_digest(pack)},
        "configuration": configuration,
        "configuration_fingerprint": fingerprint(configuration),
    }


def configuration_differences(baseline: dict[str, Any], current: dict[str, Any], allow: bool) -> dict[str, Any]:
    differences = {
        key: {"baseline": baseline.get(key), "current": current.get(key)}
        for key in sorted(baseline.keys() | current.keys())
        if key not in baseline or key not in current or _canonical(baseline[key]) != _canonical(current[key])
    }
    if differences and not allow:
        raise ValueError("evaluation configuration differs: " + ", ".join(differences) + "; use allow_config_mismatch explicitly to compare")
    return differences


def aggregate_case(case: dict[str, Any], attempts: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    errors = sum(attempt["passed"] is None for attempt in attempts)
    passed = sum(attempt["passed"] is True for attempt in attempts)
    complete = len(attempts) == repeats
    outcome = None if errors or not complete else passed == repeats
    result = {
        "id": case["id"], "category": case["category"], "attempts": attempts,
        "passed": outcome, "pass_rate": passed / repeats, "errors": errors,
    }
    # Keep the convenient single-run output fields while attempts remains canonical.
    if repeats == 1 and attempts:
        result.update({key: value for key, value in attempts[0].items() if key != "passed"})
    return result


def observation(attempts: list[dict[str, Any]], confirm_failures: bool) -> dict[str, Any]:
    if not attempts:
        return {"counted": False, "reason": "missing_candidate"}
    first = attempts[0]["passed"]
    if first is None:
        return {"counted": False, "reason": "request_error"}
    if first is True:
        return {"counted": True, "regressed": False, "flaky": False}
    if not confirm_failures:
        return {"counted": True, "regressed": True, "flaky": False}
    if len(attempts) < 2:
        return {"counted": False, "reason": "missing_confirmation"}
    second = attempts[1]["passed"]
    if second is None:
        return {"counted": False, "reason": "confirmation_error"}
    return {"counted": True, "regressed": second is False, "flaky": second is True}


def usage_total(results: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {}

    def merge(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict):
                if key not in target:
                    target[key] = {}
                if isinstance(target[key], dict):
                    merge(target[key], value)
            elif type(value) is int or (type(value) is float and math.isfinite(value)):
                if key not in target:
                    target[key] = 0
                if not isinstance(target[key], dict):
                    target[key] += value

    for result in results:
        for attempt in result.get("attempts", []):
            merge(totals, attempt.get("usage", {}))
    return totals


def execution_summary(results: list[dict[str, Any]], repeats: int | None = None) -> dict[str, Any]:
    attempts = [attempt for item in results for attempt in item.get("attempts", [])]
    summary: dict[str, Any] = {
        "total": len(results),
        "passed": sum(item.get("passed") is True for item in results),
        "failed": sum(item.get("passed") is False for item in results),
        "errored_probes": sum(any(a["passed"] is None for a in item.get("attempts", [])) for item in results),
        "errors": sum(attempt["passed"] is None for attempt in attempts),
        "attempts": len(attempts),
        "attempt_passed": sum(attempt["passed"] is True for attempt in attempts),
        "attempt_failed": sum(attempt["passed"] is False for attempt in attempts),
        "categories": {},
    }
    if repeats is not None:
        summary["repeats"] = repeats
    if any("counted" in item for item in results):
        summary["counted"] = sum(item.get("counted") is True for item in results)
        summary["regressions"] = sum(item.get("regressed") is True for item in results)
        summary["flaky"] = sum(item.get("flaky") is True for item in results)
    for category in sorted({item["category"] for item in results}):
        selected = [item for item in results if item["category"] == category]
        counts = {
            "total": len(selected), "passed": sum(item.get("passed") is True for item in selected),
            "failed": sum(item.get("passed") is False for item in selected),
            "errors": sum(a["passed"] is None for item in selected for a in item.get("attempts", [])),
        }
        if any("counted" in item for item in selected):
            counts["counted"] = sum(item.get("counted") is True for item in selected)
            counts["regressions"] = sum(item.get("regressed") is True for item in selected)
        summary["categories"][category] = counts
    return summary


def availability(results: list[dict[str, Any]], max_error_rate: float, aborted: bool = False) -> dict[str, Any]:
    attempts = [attempt for item in results for attempt in item.get("attempts", [])]
    errors = sum(attempt["passed"] is None for attempt in attempts)
    rate = errors / len(attempts) if attempts else 1.0
    return {
        "status": "unavailable" if aborted or not attempts else ("degraded" if rate > max_error_rate else "available"),
        "attempts": len(attempts), "errors": errors, "error_rate": rate,
        "unknown_usage": any(attempt.get("unknown_usage", False) for attempt in attempts),
    }


def coverage(pack: dict[str, Any], eligible: list[dict[str, Any]], results: list[dict[str, Any]], *, offline: bool, min_category_samples: int) -> dict[str, Any]:
    counted = sum(item.get("counted") is True for item in results)
    observed = sum(bool(item.get("attempts")) for item in results)
    denominator = len(eligible) if offline else len(results)
    categories: dict[str, Any] = {}
    for category in sorted({case["category"] for case in pack["cases"]}):
        categories[category] = {
            "eligible": sum(case["category"] == category for case in eligible),
            "observed": sum(item["category"] == category and bool(item.get("attempts")) for item in results),
            "counted": sum(item["category"] == category and item.get("counted") is True for item in results),
            "required": min_category_samples,
        }
    return {
        "pack_total": len(pack["cases"]), "eligible": len(eligible), "observed": observed,
        "counted": counted, "missing": max(0, len(eligible) - observed),
        "unresolved": denominator - counted, "valid_rate": counted / denominator if denominator else 0.0,
        "category_coverage_met": all(item["counted"] >= min_category_samples for item in categories.values()),
        "categories": categories,
    }


def _validate_usage(value: Any, path: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    for key, child in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{path} keys must be strings")
        if isinstance(child, dict):
            _validate_usage(child, f"{path}.{key}")
        elif type(child) not in (int, float) or child < 0 or (type(child) is float and not math.isfinite(child)):
            raise ValueError(f"{path}.{key} must be a non-negative finite number")


def validate_run_report(pack: dict[str, Any], report: dict[str, Any]) -> None:
    """Validate full-run reports before trusting their baseline or candidate data."""
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != SCHEMA_VERSION:
        raise ValueError("report.schema_version must be 2; recreate legacy baselines")
    if report.get("mode") != "baseline":
        raise ValueError("report.mode must be baseline; smoke and check reports cannot be used as baselines/candidates")
    for key in ("created_at", "model"):
        if not isinstance(report.get(key), str) or not report[key].strip():
            raise ValueError(f"report.{key} must be a non-empty string")
    try:
        created_at = datetime.fromisoformat(report["created_at"])
    except ValueError as exc:
        raise ValueError("report.created_at must be an ISO timestamp") from exc
    if created_at.tzinfo is None:
        raise ValueError("report.created_at must include a timezone")
    expected_pack = {"name": pack["name"], "version": pack["version"], "digest": pack_digest(pack)}
    if report.get("pack") != expected_pack:
        raise ValueError("report and current probe pack have different digests or metadata")
    configuration = report.get("configuration")
    if not isinstance(configuration, dict) or not isinstance(configuration.get("surface"), str) or configuration.get("model") != report["model"]:
        raise ValueError("report.configuration must identify its evaluation surface and model")
    validate_json_value(configuration, path="report.configuration")
    if configuration["surface"] not in {"custom_completion", "chat_completions"}:
        raise ValueError("report.configuration.surface is unsupported")
    if configuration["surface"] == "chat_completions":
        required = {"endpoint", "system_prompt_digest", "temperature", "seed", "token_param", "reasoning_effort"}
        if not required.issubset(configuration):
            raise ValueError("report.configuration is missing chat_completions settings")
        endpoint = configuration["endpoint"]
        if not isinstance(endpoint, str):
            raise ValueError("report.configuration.endpoint must be an HTTP URL")
        try:
            parsed = urlsplit(endpoint)
            parsed.port
        except ValueError as exc:
            raise ValueError("report.configuration.endpoint is invalid") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment or not parsed.path.endswith("/chat/completions"):
            raise ValueError("report.configuration.endpoint must be a sanitized chat-completions URL")
        digest = configuration["system_prompt_digest"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("report.configuration.system_prompt_digest must be a SHA-256 digest")
        temperature = configuration["temperature"]
        if temperature is not None and (type(temperature) not in (int, float) or not 0 <= temperature <= 2 or (type(temperature) is float and not math.isfinite(temperature))):
            raise ValueError("report.configuration.temperature must be null or a number in [0, 2]")
        if configuration["seed"] is not None and type(configuration["seed"]) is not int:
            raise ValueError("report.configuration.seed must be an integer or null")
        if not isinstance(configuration["token_param"], str) or configuration["token_param"] not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("report.configuration.token_param is unsupported")
        effort = configuration["reasoning_effort"]
        if effort is not None and (not isinstance(effort, str) or not effort.strip()):
            raise ValueError("report.configuration.reasoning_effort must be a string or null")
    if configuration.get("grader_version") != GRADER_VERSION:
        raise ValueError("report uses an incompatible grader_version; recreate the report")
    caps = configuration.get("max_tokens_by_case")
    if not isinstance(caps, dict) or set(caps) != {case["id"] for case in pack["cases"]} or any(type(cap) is not int or cap < 1 for cap in caps.values()):
        raise ValueError("report.configuration.max_tokens_by_case must contain a positive token cap for every probe")
    if report.get("configuration_fingerprint") != fingerprint(configuration):
        raise ValueError("report.configuration_fingerprint does not match its configuration")
    results = report.get("results")
    summary = report.get("summary")
    if not isinstance(results, list) or not isinstance(summary, dict):
        raise ValueError("report requires a results list and summary object")
    repeats = summary.get("repeats")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("report.summary.repeats must be a positive integer")
    if type(report.get("aborted")) is not bool:
        raise ValueError("report.aborted must be boolean")
    cases = {case["id"]: case for case in pack["cases"]}
    seen: set[str] = set()
    for index, item in enumerate(results):
        path = f"report.results[{index}]"
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or item["id"] not in cases:
            raise ValueError(f"{path}.id must belong to the probe pack")
        if item["id"] in seen:
            raise ValueError(f"{path}.id is duplicated")
        seen.add(item["id"])
        case = cases[item["id"]]
        if item.get("category") != case["category"]:
            raise ValueError(f"{path}.category does not match the probe pack")
        if "passed" not in item or (item["passed"] is not None and type(item["passed"]) is not bool):
            raise ValueError(f"{path}.passed must be true, false, or null")
        attempts = item.get("attempts")
        if not isinstance(attempts, list) or not 1 <= len(attempts) <= repeats:
            raise ValueError(f"{path}.attempts must contain 1..repeats attempts")
        if len(attempts) < repeats and not report["aborted"]:
            raise ValueError(f"{path}.attempts is incomplete without an aborted run")
        for attempt_index, attempt in enumerate(attempts):
            attempt_path = f"{path}.attempts[{attempt_index}]"
            if not isinstance(attempt, dict) or "passed" not in attempt or (attempt["passed"] is not None and type(attempt["passed"]) is not bool):
                raise ValueError(f"{attempt_path}.passed must be true, false, or null")
            if attempt["passed"] is None:
                if not isinstance(attempt.get("error"), str) or not attempt["error"]:
                    raise ValueError(f"{attempt_path}.error is required for an invalid attempt")
                if attempt.get("status") == "ok":
                    raise ValueError(f"{attempt_path}.status is inconsistent with an invalid attempt")
            elif attempt.get("error") is not None:
                raise ValueError(f"{attempt_path} cannot have both a scored result and an error")
            else:
                if not isinstance(attempt.get("output"), str) or attempt.get("status") != "ok" or attempt.get("finish_reason") != "stop" or attempt.get("refusal"):
                    raise ValueError(f"{attempt_path} cannot score an unfinished or invalid response")
                actual_passed, normalized = grade(attempt["output"], case["grader"])
                if attempt["passed"] is not actual_passed or attempt.get("normalized_output") != normalized:
                    raise ValueError(f"{attempt_path}.passed/normalized_output does not match the grader")
            if not isinstance(attempt.get("status"), str) or not attempt["status"] or type(attempt.get("fatal")) is not bool:
                raise ValueError(f"{attempt_path} requires status and boolean fatal")
            _validate_usage(attempt.get("usage"), f"{attempt_path}.usage")
        expected = aggregate_case(case, attempts, repeats)
        for key in ("passed", "pass_rate", "errors"):
            if key not in item or _canonical(item[key]) != _canonical(expected[key]):
                raise ValueError(f"{path}.{key} is inconsistent with attempts")
    if _canonical(summary) != _canonical(execution_summary(results, repeats)):
        raise ValueError("report.summary is inconsistent with attempts/results")
    if _canonical(report.get("usage")) != _canonical(usage_total(results)):
        raise ValueError("report.usage is inconsistent with attempts")
    recorded_coverage = report.get("coverage")
    if not isinstance(recorded_coverage, dict) or recorded_coverage.get("pack_total") != len(cases):
        raise ValueError("report.coverage must identify pack_total")
    selected = recorded_coverage.get("selected_ids")
    if not isinstance(selected, list) or any(not isinstance(case_id, str) or case_id not in cases for case_id in selected) or len(set(selected)) != len(selected):
        raise ValueError("report.coverage.selected_ids must be unique members of the probe pack")
    if not seen.issubset(selected) or (not report["aborted"] and seen != set(selected)):
        raise ValueError("report results do not match selected coverage")
    if type(recorded_coverage.get("observed")) is not int or recorded_coverage["observed"] != len(results):
        raise ValueError("report.coverage.observed is inconsistent with results")
    expected_complete = len(results) == len(selected) and not report["aborted"]
    if recorded_coverage.get("complete") is not expected_complete:
        raise ValueError("report.coverage.complete is inconsistent with results")
    expected_categories = {
        category: {
            "selected": sum(cases[case_id]["category"] == category for case_id in selected),
            "observed": sum(item["category"] == category for item in results),
            "healthy": sum(item["category"] == category and item["passed"] is True for item in results),
        }
        for category in sorted({case["category"] for case in cases.values()})
    }
    if _canonical(recorded_coverage.get("categories")) != _canonical(expected_categories):
        raise ValueError("report.coverage.categories is inconsistent with results")
    fatal = any(attempt.get("fatal") for item in results for attempt in item["attempts"])
    if fatal is not report["aborted"]:
        raise ValueError("report.aborted is inconsistent with fatal attempts")
