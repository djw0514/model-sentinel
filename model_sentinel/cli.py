"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

from . import __version__
from .api import ChatCompletionsClient
from .calibration import calibrate_manifest
from .pack import DEFAULT_PACK, load_pack, pack_digest
from .runner import adaptive_check, compare_runs, load_report, run_full, save_report, validate_baseline
from .sequential import SequentialConfig


VERDICT_EXIT = {"stable": 0, "regressed": 2, "inconclusive": 3}


class SentinelParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=os.getenv("MODEL_SENTINEL_MODEL"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("MODEL_SENTINEL_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="explicit key variable; otherwise MODEL_SENTINEL_API_KEY then OPENAI_API_KEY",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--omit-temperature",
        action="store_true",
        help="omit temperature for APIs/models that do not support this parameter",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reasoning-effort", help="optional provider-supported reasoning effort")
    parser.add_argument("--max-requests", type=int, help="total HTTP request budget, including retries")
    parser.add_argument("--deadline", type=float, help="total run time budget in seconds")
    parser.add_argument(
        "--token-param",
        choices=["max_tokens", "max_completion_tokens"],
        default="max_tokens",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="override the per-probe output cap (reasoning models may need a larger value)",
    )


def _add_stat_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--p0", type=float, default=0.02, help="acceptable regression rate")
    parser.add_argument("--p1", type=float, default=0.20, help="material regression rate")
    parser.add_argument("--alpha", type=float, default=0.05, help="false-alarm target")
    parser.add_argument("--beta", type=float, default=0.10, help="miss target at p1")
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=32)


def _config(args: argparse.Namespace) -> SequentialConfig:
    return SequentialConfig(
        p0=args.p0,
        p1=args.p1,
        alpha=args.alpha,
        beta=args.beta,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
    )


def _client(args: argparse.Namespace, parser: argparse.ArgumentParser) -> ChatCompletionsClient:
    if not args.model:
        parser.error("--model is required (or set MODEL_SENTINEL_MODEL)")
    if args.api_key_env is not None:
        key = os.getenv(args.api_key_env)
        if not key:
            raise ValueError(f"API key environment variable {args.api_key_env!r} is empty or missing")
    else:
        key = os.getenv("MODEL_SENTINEL_API_KEY") or os.getenv("OPENAI_API_KEY")
    return ChatCompletionsClient(
        base_url=args.base_url,
        api_key=key,
        model=args.model,
        timeout=args.timeout,
        retries=args.retries,
        temperature=None if args.omit_temperature else args.temperature,
        seed=args.seed,
        token_param=args.token_param,
        reasoning_effort=args.reasoning_effort,
        max_requests=args.max_requests,
        deadline=args.deadline,
    )


def _print_baseline(report: dict[str, Any], output: str) -> None:
    summary = report["summary"]
    usage = report.get("usage", {})
    print(
        f"{report['mode']}: {summary['passed']}/{summary['total']} passed, "
        f"{summary['errors']} errors -> {output}"
    )
    if usage:
        print("usage:", json.dumps(usage, ensure_ascii=False, sort_keys=True))
    if report.get("aborted"):
        print(f"stopped: {report.get('abort_reason', 'run interrupted')}")


def _print_verdict(report: dict[str, Any], output: str | None = None) -> None:
    verdict = report["verdict"]
    line = (
        f"verdict: {verdict['decision']} | samples={verdict['samples']} | "
        f"confirmed_regressions={verdict['confirmed_regressions']} | "
        f"llr={verdict['llr']}"
    )
    if output:
        line += f" -> {output}"
    print(line)
    usage = report.get("usage", {})
    if usage:
        print("usage:", json.dumps(usage, ensure_ascii=False, sort_keys=True))
    availability = report.get("availability", {})
    coverage = report.get("coverage", {})
    if availability or coverage:
        print(
            f"availability: {availability.get('status', 'unknown')} | "
            f"errors={availability.get('errors', 0)} | "
            f"unresolved={coverage.get('unresolved', 0)}"
        )
    if report.get("config_mismatches"):
        print("configuration differences:", ", ".join(report["config_mismatches"]))
    if verdict.get("quality_reasons"):
        print("quality:", ", ".join(verdict["quality_reasons"]))
    if report.get("aborted"):
        print("stopped:", report.get("abort_reason") or "run interrupted")


def _exit_code(report: dict[str, Any]) -> int:
    """Execution failures take precedence; the saved report retains capability evidence."""
    if (report.get("aborted") or report.get("availability", {}).get("errors", 0)
            or report.get("execution_errors", 0)):
        return 1
    return VERDICT_EXIT[report["verdict"]["decision"]]


def _add_comparison_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--allow-config-mismatch", action="store_true",
                        help="explicit cross-configuration comparison; differences are recorded")
    parser.add_argument("--max-error-rate", type=float, default=0.2,
                        help="maximum unresolved observation fraction for a stable verdict")
    parser.add_argument("--no-confirm", action="store_true",
                        help="use first answers only; thresholds apply to unconfirmed failures")


def build_parser() -> argparse.ArgumentParser:
    parser = SentinelParser(
        prog="model-sentinel",
        description="Detect model capability regressions with an adaptive, low-token probe suite.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run the full pack and save a baseline/candidate report")
    _add_api_arguments(run_parser)
    run_parser.add_argument("--pack", default=str(DEFAULT_PACK))
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--workers", type=int, default=1, help="bounded full-run concurrency")
    run_parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="repeat every baseline probe; 3 is recommended for a high-confidence baseline",
    )

    check_parser = sub.add_parser("check", help="adaptively compare a live endpoint with a baseline")
    _add_api_arguments(check_parser)
    _add_stat_arguments(check_parser)
    check_parser.add_argument("--pack", default=str(DEFAULT_PACK))
    check_parser.add_argument("--baseline", help="required for standard and deep modes")
    check_parser.add_argument("--output", required=True)
    check_parser.add_argument("--mode", choices=["quick", "standard", "deep"], default="standard")
    check_parser.add_argument("--sample-seed", type=int, help="probe ordering seed, separate from model seed")
    check_parser.add_argument("--sampling", choices=["stratified", "ordered"], default="stratified")
    check_parser.add_argument("--min-category-samples", type=int, default=1)
    check_parser.add_argument("--workers", type=int, default=1, help="concurrency for quick/deep only")
    check_parser.add_argument("--repeats", type=int, default=2, help="attempts per probe in deep mode")
    _add_comparison_arguments(check_parser)

    compare_parser = sub.add_parser("compare", help="compare two saved full-run reports")
    _add_stat_arguments(compare_parser)
    compare_parser.add_argument("baseline")
    compare_parser.add_argument("candidate")
    compare_parser.add_argument("--pack", default=str(DEFAULT_PACK))
    compare_parser.add_argument("--output")
    compare_parser.add_argument("--replay", action="store_true", help="replay SPRT while retaining full summary")
    _add_comparison_arguments(compare_parser)

    validate_parser = sub.add_parser("validate-pack", help="validate a custom probe pack")
    validate_parser.add_argument("pack")

    calibration_parser = sub.add_parser("calibrate", help="evaluate thresholds against labeled saved runs")
    calibration_parser.add_argument("manifest", help="JSON manifest with labeled baseline/candidate pairs")
    calibration_parser.add_argument("--output", required=True)
    calibration_parser.add_argument("--replay", action="store_true")
    _add_stat_arguments(calibration_parser)
    _add_comparison_arguments(calibration_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command in {"check", "compare", "calibrate"}:
            if not math.isfinite(args.max_error_rate) or not 0 <= args.max_error_rate < 1:
                raise ValueError("--max-error-rate must be finite and in [0, 1)")
        if args.command == "validate-pack":
            pack = load_pack(args.pack)
            print(
                f"valid: {pack['name']} {pack['version']} | "
                f"cases={len(pack['cases'])} | sha256={pack_digest(pack)}"
            )
            return 0

        if args.command == "calibrate":
            report = calibrate_manifest(
                args.manifest, config=_config(args), confirm_failures=not args.no_confirm,
                replay=args.replay, allow_config_mismatch=args.allow_config_mismatch,
                max_error_rate=args.max_error_rate,
            )
            save_report(args.output, report)
            print(f"calibration: {report['total']} labeled comparisons -> {args.output}")
            print("rates:", json.dumps({key: report[key] for key in
                  ("false_alarm_rate", "miss_rate", "inconclusive_rate")}, sort_keys=True))
            return 0

        pack = load_pack(args.pack)
        if args.command == "run":
            report = run_full(
                pack,
                _client(args, parser),
                limit=args.limit,
                max_tokens=args.max_tokens,
                repeats=args.repeats,
                workers=args.workers,
            )
            save_report(args.output, report)
            _print_baseline(report, args.output)
            return 1 if report["summary"]["errors"] or report.get("aborted") else 0

        if args.command == "check":
            config = _config(args)
            if args.mode == "quick":
                if args.baseline:
                    raise ValueError("quick is an absolute smoke test; use standard/deep with a baseline")
                report = run_full(pack, _client(args, parser), limit=min(12, config.max_samples),
                                  max_tokens=args.max_tokens, repeats=1, workers=args.workers, mode="smoke")
                summary = report["summary"]
                outcome = ("smoke_error" if summary["errors"] or report.get("aborted") else
                           "smoke_fail" if summary["failed"] else "smoke_pass")
                report["smoke_verdict"] = outcome
                save_report(args.output, report)
                _print_baseline(report, args.output)
                print("smoke verdict:", outcome)
                return {"smoke_pass": 0, "smoke_fail": 2, "smoke_error": 1}[outcome]
            if not args.baseline:
                raise ValueError("--baseline is required for standard and deep modes")
            baseline = load_report(args.baseline)
            client = _client(args, parser)
            if args.mode == "deep":
                if not args.no_confirm and args.repeats < 2:
                    raise ValueError("deep mode requires --repeats >= 2 for failure confirmation")
                validate_baseline(pack, baseline, client, max_tokens=args.max_tokens,
                                  allow_config_mismatch=args.allow_config_mismatch)
                healthy = [item for item in baseline["results"] if item["passed"] is True]
                if len(healthy) < config.min_samples:
                    raise ValueError("baseline has insufficient healthy probes for deep comparison; rebuild it")
                if {item["category"] for item in healthy} != {case["category"] for case in pack["cases"]}:
                    raise ValueError("baseline lacks healthy probes in one or more categories; rebuild it")
                candidate = run_full(pack, client, max_tokens=args.max_tokens,
                                     repeats=args.repeats, workers=args.workers)
                report = compare_runs(pack, baseline, candidate, config=config,
                                      confirm_failures=not args.no_confirm,
                                      allow_config_mismatch=args.allow_config_mismatch,
                                      max_error_rate=args.max_error_rate)
                report["evaluation_mode"] = "deep"
                report["candidate_run"] = candidate
                report["usage"] = candidate["usage"]
                report["execution_errors"] = candidate["summary"]["errors"]
                report["aborted"] = candidate.get("aborted", False)
                if "budget" in candidate:
                    report["budget"] = candidate["budget"]
            else:
                if args.workers != 1:
                    raise ValueError("standard mode is sequential; --workers must be 1")
                report = adaptive_check(
                    pack, baseline, client, config=config,
                    confirm_failures=not args.no_confirm, max_tokens=args.max_tokens,
                    sample_seed=args.sample_seed, sampling=args.sampling,
                    min_category_samples=args.min_category_samples,
                    max_error_rate=args.max_error_rate,
                    allow_config_mismatch=args.allow_config_mismatch,
                )
            save_report(args.output, report)
            _print_verdict(report, args.output)
            return _exit_code(report)

        if args.command == "compare":
            report = compare_runs(
                pack,
                load_report(args.baseline),
                load_report(args.candidate),
                config=_config(args),
                confirm_failures=not args.no_confirm,
                replay=args.replay,
                allow_config_mismatch=args.allow_config_mismatch,
                max_error_rate=args.max_error_rate,
            )
            if args.output:
                save_report(args.output, report)
            _print_verdict(report, args.output)
            return _exit_code(report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error("unknown command")
    return 1
