"""Full runs, adaptive checks, and complete offline comparisons."""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Protocol

from .api import APIError, APIResponse
from .grading import grade, strict_json_loads
from .pack import validate_pack
from .reports import (
    aggregate_case, availability, configuration_differences, coverage,
    evaluation_configuration, execution_summary, metadata, observation,
    usage_total, validate_run_report,
)
from .sampling import sample_order
from .sequential import SequentialConfig, SequentialTest, fixed_sample_verdict


class CompletionClient(Protocol):
    model: str

    def complete(self, prompt: str, max_tokens: int) -> APIResponse: ...


def _positive_integer(value: Any, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_options(max_tokens: int | None, max_error_rate: float = 0.2) -> None:
    if max_tokens is not None:
        _positive_integer(max_tokens, "max_tokens")
    if type(max_error_rate) not in (float, int) or not 0 <= max_error_rate < 1 or (type(max_error_rate) is float and not math.isfinite(max_error_rate)):
        raise ValueError("max_error_rate must be finite and in [0, 1)")


def _attempt(client: CompletionClient, case: dict[str, Any], max_tokens: int | None) -> dict[str, Any]:
    try:
        response = client.complete(case["prompt"], max_tokens if max_tokens is not None else case.get("max_tokens", 16))
    except APIError as exc:
        return {
            "passed": None, "status": getattr(exc, "kind", "api_error"),
            "error": str(exc) or "API request failed", "usage": {},
            "fatal": getattr(exc, "fatal", False), "latency_ms": getattr(exc, "latency_ms", 0),
            "http_attempts": getattr(exc, "http_attempts", []),
            "unknown_usage": getattr(exc, "unknown_usage", False),
        }
    except ValueError as exc:
        return {"passed": None, "status": "protocol_error", "error": str(exc), "usage": {}, "fatal": False}

    finish_reason = getattr(response, "finish_reason", "stop")
    refusal = getattr(response, "refusal", None)
    result: dict[str, Any] = {
        "output": response.text, "latency_ms": response.latency_ms,
        "usage": response.usage, "response_id": response.response_id,
        "finish_reason": finish_reason, "refusal": refusal,
        "response_model": getattr(response, "response_model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "http_attempts": getattr(response, "http_attempts", []),
        "unknown_usage": getattr(response, "unknown_usage", False), "fatal": False,
    }
    if refusal or finish_reason == "content_filter":
        result.update(passed=None, status="refused", error="response was refused or filtered")
    elif finish_reason == "length":
        result.update(passed=None, status="truncated", error="response exceeded the output token limit")
    elif finish_reason != "stop":
        result.update(passed=None, status="protocol_error", error=f"unsupported or missing finish_reason: {finish_reason}")
    else:
        try:
            passed, normalized = grade(response.text, case["grader"])
            result.update(passed=passed, normalized_output=normalized, status="ok")
        except ValueError as exc:
            result.update(passed=None, status="grading_error", error=str(exc))
    return result


def _attach_budget(report: dict[str, Any], client: CompletionClient) -> None:
    snapshot = getattr(client, "budget_snapshot", None)
    if callable(snapshot):
        report["budget"] = snapshot()


def _abort_details(results: list[dict[str, Any]]) -> tuple[bool, str | None]:
    fatal = next((attempt for item in results for attempt in item.get("attempts", []) if attempt.get("fatal")), None)
    return fatal is not None, fatal.get("error") if fatal else None


def run_full(
    pack: dict[str, Any], client: CompletionClient, *, limit: int | None = None,
    max_tokens: int | None = None, repeats: int = 1, workers: int = 1, mode: str = "baseline",
) -> dict[str, Any]:
    validate_pack(pack)
    _positive_integer(repeats, "repeats")
    _positive_integer(workers, "workers")
    if workers > 32:
        raise ValueError("workers must be at most 32")
    if limit is not None:
        _positive_integer(limit, "limit")
    _validate_options(max_tokens)
    if mode not in {"baseline", "smoke"}:
        raise ValueError("full run mode must be baseline or smoke")
    cases = pack["cases"][:limit] if limit is not None else pack["cases"]
    stop = threading.Event()

    def execute(case: dict[str, Any]) -> dict[str, Any] | None:
        attempts = []
        for _ in range(repeats):
            if stop.is_set():
                break
            attempt = _attempt(client, case, max_tokens)
            attempts.append(attempt)
            if attempt.get("fatal"):
                stop.set()
                break
        return aggregate_case(case, attempts, repeats) if attempts else None

    indexed: dict[int, dict[str, Any]] = {}
    if workers == 1:
        for index, case in enumerate(cases):
            result = execute(case)
            if result is not None:
                indexed[index] = result
            if stop.is_set():
                break
    else:
        # Never queue the entire pack: at most workers cases can be in flight.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sentinel") as executor:
            pending: dict[Any, int] = {}
            next_index = 0
            while pending or (next_index < len(cases) and not stop.is_set()):
                while len(pending) < workers and next_index < len(cases) and not stop.is_set():
                    pending[executor.submit(execute, cases[next_index])] = next_index
                    next_index += 1
                if not pending:
                    break
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    index = pending.pop(future)
                    if not future.cancelled():
                        result = future.result()
                        if result is not None:
                            indexed[index] = result
                if stop.is_set():
                    for future in pending:
                        future.cancel()

    results = [indexed[index] for index in sorted(indexed)]
    aborted, abort_reason = _abort_details(results)
    report = metadata(pack, client.model, mode, evaluation_configuration(client, pack, max_tokens))
    report.update({
        "summary": execution_summary(results, repeats), "results": results, "usage": usage_total(results),
        "parameters": {"repeats": repeats, "workers": workers, "limit": limit},
        "aborted": aborted, "abort_reason": abort_reason,
        "availability": availability(results, 0.2, aborted),
        "coverage": {
            "pack_total": len(pack["cases"]), "selected_ids": [case["id"] for case in cases],
            "observed": len(results), "complete": len(results) == len(cases) and not aborted,
            "categories": {
                category: {
                    "selected": sum(case["category"] == category for case in cases),
                    "observed": sum(item["category"] == category for item in results),
                    "healthy": sum(item["category"] == category and item["passed"] is True for item in results),
                }
                for category in sorted({case["category"] for case in pack["cases"]})
            },
        },
    })
    _attach_budget(report, client)
    return report


def validate_baseline(
    pack: dict[str, Any], baseline: dict[str, Any], client: CompletionClient | None = None,
    *, max_tokens: int | None = None, allow_config_mismatch: bool = False,
) -> dict[str, Any]:
    _validate_options(max_tokens)
    validate_run_report(pack, baseline)
    if client is None:
        return {}
    return configuration_differences(
        baseline["configuration"], evaluation_configuration(client, pack, max_tokens), allow_config_mismatch,
    )


def _verify_baseline(pack: dict[str, Any], baseline: dict[str, Any]) -> None:
    validate_baseline(pack, baseline)


def _eligible(pack: dict[str, Any], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    healthy = {item["id"] for item in baseline["results"] if item["passed"] is True}
    return [case for case in pack["cases"] if case["id"] in healthy]


def _default_config(pack: dict[str, Any]) -> SequentialConfig:
    maximum = min(32, len(pack["cases"]))
    return SequentialConfig(min_samples=min(8, maximum), max_samples=maximum)


def _gate_verdict(snapshot: dict[str, Any], report_coverage: dict[str, Any], report_availability: dict[str, Any], max_error_rate: float, *, aborted: bool) -> dict[str, Any]:
    verdict = dict(snapshot)
    reasons = []
    if verdict["decision"] == "running":
        verdict["decision"] = "inconclusive"
    if verdict["decision"] == "stable":
        if aborted:
            reasons.append("run_aborted")
        if report_availability["error_rate"] > max_error_rate:
            reasons.append("error_rate_exceeded")
        if 1 - report_coverage["valid_rate"] > max_error_rate + 1e-12:
            reasons.append("insufficient_valid_coverage")
        if not report_coverage["category_coverage_met"]:
            reasons.append("insufficient_category_coverage")
        if reasons:
            verdict["decision"] = "inconclusive"
    verdict["quality_reasons"] = reasons
    return verdict


def adaptive_check(
    pack: dict[str, Any], baseline: dict[str, Any], client: CompletionClient, *,
    config: SequentialConfig | None = None, confirm_failures: bool = True,
    max_tokens: int | None = None, sample_seed: int | None = None,
    sampling: str = "stratified", min_category_samples: int = 1,
    max_error_rate: float = 0.2, allow_config_mismatch: bool = False,
) -> dict[str, Any]:
    _validate_options(max_tokens, max_error_rate)
    _positive_integer(min_category_samples, "min_category_samples")
    differences = validate_baseline(pack, baseline, client, max_tokens=max_tokens, allow_config_mismatch=allow_config_mismatch)
    config = config or _default_config(pack)
    eligible = _eligible(pack, baseline)
    categories = {case["category"] for case in pack["cases"]}
    effective_min_samples = max(config.min_samples, len(categories) * min_category_samples)
    if effective_min_samples > config.max_samples:
        raise ValueError("max_samples cannot satisfy min_category_samples for every category")
    effective_config = replace(config, min_samples=effective_min_samples)
    preflight_issues = []
    if len(eligible) < effective_min_samples:
        preflight_issues.append("insufficient_healthy_baseline_probes")
    if any(sum(case["category"] == category for case in eligible) < min_category_samples for category in categories):
        preflight_issues.append("insufficient_healthy_baseline_category_coverage")
    ordered, actual_seed = sample_order(eligible, strategy=sampling, seed=sample_seed)
    test = SequentialTest(effective_config)
    results: list[dict[str, Any]] = []
    for case in ([] if preflight_issues else ordered):
        if test.decision() != "running":
            break
        attempts = [_attempt(client, case, max_tokens)]
        if attempts[0]["passed"] is False and confirm_failures:
            attempts.append(_attempt(client, case, max_tokens))
        item = aggregate_case(case, attempts, len(attempts))
        item.update(observation(attempts, confirm_failures))
        item["passed"] = not item["regressed"] if item["counted"] else None
        if item["counted"]:
            item["decision_after"] = test.add(item["regressed"])
        results.append(item)
        if any(attempt.get("fatal") for attempt in attempts):
            break

    aborted, abort_reason = _abort_details(results)
    snapshot = test.snapshot()
    if snapshot["decision"] == "running":
        snapshot["decision"] = "inconclusive"
    report_coverage = coverage(pack, eligible, results, offline=False, min_category_samples=min_category_samples)
    report_availability = availability(results, max_error_rate, aborted)
    report = metadata(pack, client.model, "adaptive_check", evaluation_configuration(client, pack, max_tokens))
    report.update({
        "baseline_model": baseline["model"], "config_mismatches": differences,
        "parameters": {**asdict(config), "effective_min_samples": effective_min_samples, "confirm_failures": confirm_failures, "max_error_rate": max_error_rate, "min_category_samples": min_category_samples, "allow_config_mismatch": allow_config_mismatch},
        "sampling": {"strategy": sampling, "seed": actual_seed, "order": [case["id"] for case in ordered], "observed_order": [item["id"] for item in results]},
        "capability_verdict": snapshot,
        "verdict": _gate_verdict(snapshot, report_coverage, report_availability, max_error_rate, aborted=aborted),
        "availability": report_availability, "coverage": report_coverage,
        "flaky_probes": sum(item.get("flaky") is True for item in results),
        "summary": execution_summary(results), "usage": usage_total(results), "results": results,
        "aborted": aborted, "abort_reason": abort_reason,
        "preflight_issues": preflight_issues,
    })
    report["verdict"]["quality_reasons"].extend(preflight_issues)
    _attach_budget(report, client)
    return report


def compare_runs(
    pack: dict[str, Any], baseline: dict[str, Any], candidate: dict[str, Any], *,
    config: SequentialConfig | None = None, confirm_failures: bool = True,
    replay: bool = False, allow_config_mismatch: bool = False, max_error_rate: float = 0.2,
) -> dict[str, Any]:
    _validate_options(None, max_error_rate)
    validate_baseline(pack, baseline)
    validate_run_report(pack, candidate)
    differences = configuration_differences(baseline["configuration"], candidate["configuration"], allow_config_mismatch)
    config = config or _default_config(pack)
    eligible = _eligible(pack, baseline)
    current = {item["id"]: item for item in candidate["results"]}
    compared: list[dict[str, Any]] = []
    for case in eligible:
        attempts = current.get(case["id"], {}).get("attempts", [])
        item = {"id": case["id"], "category": case["category"], "attempts": attempts}
        item.update(observation(attempts, confirm_failures))
        item["passed"] = not item["regressed"] if item["counted"] else None
        compared.append(item)

    counted = [item for item in compared if item["counted"]]
    complete_snapshot = fixed_sample_verdict(len(counted), sum(item["regressed"] for item in counted), config)
    snapshot = complete_snapshot
    replay_ids = []
    if replay:
        test = SequentialTest(config)
        for item in counted:
            if test.decision() != "running":
                break
            item["decision_after"] = test.add(item["regressed"])
            replay_ids.append(item["id"])
        snapshot = test.snapshot()
        if snapshot["decision"] == "running":
            snapshot["decision"] = "inconclusive"
    aborted = candidate["aborted"]
    report_coverage = coverage(pack, eligible, compared, offline=True, min_category_samples=1)
    report_availability = availability(candidate["results"], max_error_rate, aborted)
    comparison_summary = execution_summary(compared)
    summary = execution_summary(candidate["results"], candidate["summary"]["repeats"])
    for key in ("counted", "regressions", "flaky"):
        summary[key] = comparison_summary.get(key, 0)
    report = metadata(pack, candidate["model"], "offline_compare", candidate["configuration"])
    report.update({
        "baseline_model": baseline["model"], "candidate_model": candidate["model"], "config_mismatches": differences,
        "parameters": {**asdict(config), "confirm_failures": confirm_failures, "replay": replay, "max_error_rate": max_error_rate, "allow_config_mismatch": allow_config_mismatch},
        "capability_verdict": snapshot, "full_verdict": complete_snapshot,
        "verdict": _gate_verdict(snapshot, report_coverage, report_availability, max_error_rate, aborted=aborted),
        "availability": report_availability, "coverage": report_coverage,
        "summary": summary, "comparison_summary": comparison_summary,
        "results": compared, "usage": usage_total(candidate["results"]),
        "flaky_probes": sum(item.get("flaky") is True for item in compared),
        "aborted": aborted, "abort_reason": candidate.get("abort_reason"),
    })
    if replay:
        report["replay"] = {"order": replay_ids, "ignored_observations": len(counted) - len(replay_ids)}
    if "budget" in candidate:
        report["budget"] = candidate["budget"]
    return report


def load_report(path: str | Path) -> dict[str, Any]:
    data = strict_json_loads(Path(path).read_text(encoding="utf-8"), path="report")
    if not isinstance(data, dict):
        raise ValueError("report root must be an object")
    return data


def save_report(path: str | Path, report: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
