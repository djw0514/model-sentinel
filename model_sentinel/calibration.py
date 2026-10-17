"""Offline evaluation of decision settings against a fixed, labeled dataset.

These empirical rates describe the supplied reports, not a generalization
guarantee. Calibration never calls a model or changes decision thresholds.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import math
from pathlib import Path
import re
from typing import Any

from .grading import strict_json_loads
from .pack import load_pack, pack_digest
from .reports import SCHEMA_VERSION
from .runner import compare_runs, load_report
from .sequential import SequentialConfig


def _local_path(value: Any, parent: Path | None, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{field} must be a non-empty local file path")
    raw = str(value)
    windows_drive = re.match(r"^[A-Za-z]:[\\/]", raw)
    if raw.replace("\\", "/").startswith("//") or (
        re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", raw) and not windows_drive
    ):
        raise ValueError(f"{field} must be a local file path; URLs and network paths are unsupported")
    result = Path(raw)
    return ((parent / result) if parent is not None and not result.is_absolute() else result).resolve()


def _fields(value: dict[str, Any], allowed: set[str], path: str) -> None:
    for key in value:
        if key not in allowed:
            raise ValueError(f"{path}.{key} is not a supported field")


def calibrate_manifest(
    path: str | Path,
    *,
    config: SequentialConfig | None = None,
    confirm_failures: bool = True,
    replay: bool = False,
    allow_config_mismatch: bool = False,
    max_error_rate: float = 0.2,
) -> dict[str, Any]:
    """Compare local report pairs and retain inconclusive cases in rate denominators."""
    for key, value in (
        ("confirm_failures", confirm_failures), ("replay", replay),
        ("allow_config_mismatch", allow_config_mismatch),
    ):
        if type(value) is not bool:
            raise ValueError(f"parameters.{key} must be a boolean")
    if config is not None and not isinstance(config, SequentialConfig):
        raise ValueError("parameters.config must be a SequentialConfig")
    if type(max_error_rate) not in (int, float) or not math.isfinite(max_error_rate) or not 0 <= max_error_rate < 1:
        raise ValueError("parameters.max_error_rate must be finite and in [0, 1)")

    manifest_path = _local_path(path, None, "manifest")
    manifest = strict_json_loads(manifest_path.read_text(encoding="utf-8"), path="manifest")
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    _fields(manifest, {"cases", "pack"}, "manifest")
    entries = manifest.get("cases")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest.cases must be a non-empty list")

    names: set[str] = set()
    resolved: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        location = f"manifest.cases[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{location} must be an object")
        _fields(entry, {"name", "baseline", "candidate", "expected"}, location)
        name = entry.get("name", f"case-{index + 1}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{location}.name must be a non-empty string")
        if name in names:
            raise ValueError(f"{location}.name is duplicated: {name}")
        names.add(name)
        expected = entry.get("expected")
        if not isinstance(expected, str) or expected not in {"stable", "regressed"}:
            raise ValueError(f"{location}.expected must be 'stable' or 'regressed'")
        resolved.append({
            "name": name, "expected": expected, "location": location,
            "baseline": _local_path(entry.get("baseline"), manifest_path.parent, f"{location}.baseline"),
            "candidate": _local_path(entry.get("candidate"), manifest_path.parent, f"{location}.candidate"),
        })

    pack_path = _local_path(manifest["pack"], manifest_path.parent, "manifest.pack") if "pack" in manifest else None
    pack = load_pack(pack_path)
    if config is None:
        maximum = min(32, len(pack["cases"]))
        config = SequentialConfig(min_samples=min(8, maximum), max_samples=maximum)
    parameters = {
        **asdict(config), "confirm_failures": confirm_failures, "replay": replay,
        "allow_config_mismatch": allow_config_mismatch, "max_error_rate": max_error_rate,
    }
    confusion = {
        expected: {decision: 0 for decision in ("stable", "regressed", "inconclusive")}
        for expected in ("stable", "regressed")
    }
    results: list[dict[str, Any]] = []
    for entry in resolved:
        reports = {}
        for role in ("baseline", "candidate"):
            try:
                reports[role] = load_report(entry[role])
            except (OSError, ValueError) as exc:
                raise ValueError(f"{entry['location']}.{role}: {exc}") from exc
        try:
            comparison = compare_runs(
                pack, reports["baseline"], reports["candidate"], config=config,
                confirm_failures=confirm_failures, replay=replay,
                allow_config_mismatch=allow_config_mismatch, max_error_rate=max_error_rate,
            )
        except ValueError as exc:
            raise ValueError(f"{entry['location']}: {exc}") from exc
        verdict = comparison["verdict"]
        decision = verdict["decision"]
        confusion[entry["expected"]][decision] += 1
        results.append({
            "name": entry["name"], "baseline": str(entry["baseline"]), "candidate": str(entry["candidate"]),
            "expected": entry["expected"], "decision": decision, "matched": decision == entry["expected"],
            "samples": verdict["samples"], "quality_reasons": verdict.get("quality_reasons", []),
        })

    healthy = sum(confusion["stable"].values())
    regressed = sum(confusion["regressed"].values())
    inconclusive = sum(row["inconclusive"] for row in confusion.values())
    return {
        "schema_version": SCHEMA_VERSION, "mode": "calibration",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": str(manifest_path),
        "pack": {"name": pack["name"], "version": pack["version"], "digest": pack_digest(pack)},
        "parameters": parameters, "total": len(results),
        "false_alarm_rate": confusion["stable"]["regressed"] / healthy if healthy else None,
        "miss_rate": confusion["regressed"]["stable"] / regressed if regressed else None,
        "inconclusive_rate": inconclusive / len(results),
        "confusion": confusion, "results": results,
        "limitations": "Rates describe this fixed labeled dataset, not a generalization guarantee. Thresholds were not modified.",
    }
