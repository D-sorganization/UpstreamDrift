#!/usr/bin/env python3
"""Validate the ratcheted budget for mypy path exclusions.

Preconditions:
    The configured pyproject file must contain ``[tool.mypy].exclude`` as a
    list of strings. The budget file must be JSON with ``schema_version``,
    ``schedule``, ``coverage_gates``, and ``exclusions`` keys.
Postconditions:
    Returns 0 only when pyproject exclusions match the budget exactly and the
    active exclusion count does not exceed the ratchet cap. Coverage gate
    metadata must also define accountable per-package thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

DEFAULT_BUDGET = Path("scripts/config/mypy_exclusion_budget.json")
DEFAULT_PYPROJECT = Path("pyproject.toml")
SCHEMA_VERSION = 1
REQUIRED_COVERAGE_GATES = frozenset(
    {
        "api-routes",
        "data-io",
        "execution-checkpointing",
        "deployment",
        "optimization",
        "engine-adapters",
    }
)


@dataclass(frozen=True)
class BudgetEntry:
    """A single accountable mypy exclusion."""

    path: str
    owner: str
    reason: str
    expires_on: date


@dataclass(frozen=True)
class ScheduleEntry:
    """A dated maximum exclusion count."""

    effective_on: date
    max_exclusions: int


@dataclass(frozen=True)
class CoverageGate:
    """An accountable per-package coverage ratchet."""

    name: str
    path: str
    min_coverage: float
    owner: str
    reason: str
    ratchet_to: float
    ratchet_on: date


@dataclass(frozen=True)
class MypyOverride:
    """A normalized mypy module override block."""

    modules: tuple[str, ...]
    ignore_errors: bool


def _normalize_path(raw_path: str) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("path must be a non-empty string")
    return raw_path.strip().replace("\\", "/")


def _parse_iso_date(raw_value: Any, field_name: str) -> date:
    if not isinstance(raw_value, str):
        raise ValueError(f"{field_name} must be an ISO date string")
    try:
        return date.fromisoformat(raw_value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date string") from exc


def load_pyproject_exclusions(path: Path) -> list[str]:
    """Return normalized mypy exclude entries from ``pyproject.toml``."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    mypy_config = data.get("tool", {}).get("mypy", {})
    exclusions = mypy_config.get("exclude")
    if not isinstance(exclusions, list):
        raise ValueError("[tool.mypy].exclude must be a list")
    return [_normalize_path(entry) for entry in exclusions]


def load_mypy_overrides(path: Path) -> list[MypyOverride]:
    """Return normalized mypy override metadata from ``pyproject.toml``."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    mypy_config = data.get("tool", {}).get("mypy", {})
    raw_overrides = mypy_config.get("overrides", [])
    if not isinstance(raw_overrides, list):
        raise ValueError("[tool.mypy].overrides must be a list")
    return [_parse_mypy_override(raw_override) for raw_override in raw_overrides]


def _parse_mypy_override(raw_override: Any) -> MypyOverride:
    if not isinstance(raw_override, dict):
        raise ValueError("each mypy override must be an object")
    raw_modules = raw_override.get("module", ())
    if isinstance(raw_modules, str):
        modules: tuple[str, ...] = (raw_modules.strip(),)
    elif isinstance(raw_modules, list):
        modules = tuple(str(module).strip() for module in raw_modules)
    else:
        raise ValueError("mypy override module must be a string or list")
    return MypyOverride(
        modules=tuple(module for module in modules if module),
        ignore_errors=raw_override.get("ignore_errors") is True,
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("budget must be a JSON object")
    return data


def _parse_budget_entry(raw_entry: Any) -> BudgetEntry:
    if not isinstance(raw_entry, dict):
        raise ValueError("each exclusion must be an object")
    return BudgetEntry(
        path=_normalize_path(raw_entry.get("path", "")),
        owner=str(raw_entry.get("owner", "")).strip(),
        reason=str(raw_entry.get("reason", "")).strip(),
        expires_on=_parse_iso_date(raw_entry.get("expires_on"), "expires_on"),
    )


def _parse_schedule_entry(raw_entry: Any) -> ScheduleEntry:
    if not isinstance(raw_entry, dict):
        raise ValueError("each schedule entry must be an object")
    max_exclusions = raw_entry.get("max_exclusions")
    if not isinstance(max_exclusions, int) or max_exclusions < 0:
        raise ValueError("max_exclusions must be a non-negative integer")
    return ScheduleEntry(
        effective_on=_parse_iso_date(raw_entry.get("effective_on"), "effective_on"),
        max_exclusions=max_exclusions,
    )


def _parse_coverage_percent(raw_value: Any, field_name: str) -> float:
    if not isinstance(raw_value, int | float):
        raise ValueError(f"{field_name} must be a number")
    value = float(raw_value)
    if not 0.0 <= value <= 100.0:
        raise ValueError(f"{field_name} must be between 0 and 100")
    return value


def _parse_coverage_gate(raw_entry: Any) -> CoverageGate:
    if not isinstance(raw_entry, dict):
        raise ValueError("each coverage gate must be an object")
    name = str(raw_entry.get("name", "")).strip()
    if not name:
        raise ValueError("coverage gate name must be a non-empty string")
    return CoverageGate(
        name=name,
        path=_normalize_path(raw_entry.get("path", "")),
        min_coverage=_parse_coverage_percent(
            raw_entry.get("min_coverage"), "min_coverage"
        ),
        owner=str(raw_entry.get("owner", "")).strip(),
        reason=str(raw_entry.get("reason", "")).strip(),
        ratchet_to=_parse_coverage_percent(raw_entry.get("ratchet_to"), "ratchet_to"),
        ratchet_on=_parse_iso_date(raw_entry.get("ratchet_on"), "ratchet_on"),
    )


def load_budget(path: Path) -> tuple[list[BudgetEntry], list[ScheduleEntry]]:
    """Return budget exclusions and ratchet schedule entries."""
    data = _load_json_object(path)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    raw_exclusions = data.get("exclusions")
    raw_schedule = data.get("schedule")
    if not isinstance(raw_exclusions, list):
        raise ValueError("exclusions must be a list")
    if not isinstance(raw_schedule, list) or not raw_schedule:
        raise ValueError("schedule must be a non-empty list")
    return (
        [_parse_budget_entry(entry) for entry in raw_exclusions],
        [_parse_schedule_entry(entry) for entry in raw_schedule],
    )


def load_coverage_gates(path: Path) -> list[CoverageGate]:
    """Return accountable per-package coverage gates from the budget."""
    data = _load_json_object(path)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    raw_gates = data.get("coverage_gates")
    if not isinstance(raw_gates, list) or not raw_gates:
        raise ValueError("coverage_gates must be a non-empty list")
    return [_parse_coverage_gate(entry) for entry in raw_gates]


def active_cap(schedule: list[ScheduleEntry], today: date) -> int:
    """Return the newest ratchet cap effective on or before ``today``."""
    active_entries = [entry for entry in schedule if entry.effective_on <= today]
    if not active_entries:
        raise ValueError("schedule has no entry effective today")
    return max(active_entries, key=lambda entry: entry.effective_on).max_exclusions


def _duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def validate_budget(
    pyproject_exclusions: list[str],
    budget_entries: list[BudgetEntry],
    schedule: list[ScheduleEntry],
    today: date,
    overrides: list[MypyOverride] | None = None,
) -> list[str]:
    """Return validation errors for mypy exclusion budget drift."""
    errors: list[str] = []
    budget_paths = [entry.path for entry in budget_entries]
    errors.extend(_validate_duplicates(pyproject_exclusions, "pyproject"))
    errors.extend(_validate_duplicates(budget_paths, "budget"))
    errors.extend(_validate_metadata(budget_entries, today))
    errors.extend(_validate_no_ignore_errors(overrides or []))
    errors.extend(_validate_path_sets(pyproject_exclusions, budget_paths))
    errors.extend(_validate_schedule_ratchet(schedule))
    cap = active_cap(schedule, today)
    if len(budget_entries) > cap:
        errors.append(
            f"exclusions exceed active cap: {len(budget_entries)} configured, {cap} allowed"
        )
    return errors


def validate_coverage_gates(
    gates: list[CoverageGate],
    today: date,
) -> list[str]:
    """Return validation errors for per-package coverage gate metadata."""
    errors: list[str] = []
    gate_names = [gate.name for gate in gates]
    errors.extend(_validate_duplicates(gate_names, "coverage gate"))
    missing_gates = REQUIRED_COVERAGE_GATES - set(gate_names)
    for gate_name in sorted(missing_gates):
        errors.append(f"coverage gate missing required package: {gate_name}")
    for gate in gates:
        if not gate.owner:
            errors.append(f"{gate.name}: missing owner")
        if not gate.reason:
            errors.append(f"{gate.name}: missing reason")
        if gate.ratchet_to <= gate.min_coverage:
            errors.append(
                f"{gate.name}: ratchet_to must exceed min_coverage "
                f"({gate.ratchet_to:g} <= {gate.min_coverage:g})"
            )
        if gate.ratchet_on < today:
            errors.append(f"{gate.name}: coverage ratchet expired on {gate.ratchet_on}")
    return errors


def _validate_duplicates(values: list[str], label: str) -> list[str]:
    return [
        f"{label} has duplicate exclusion: {value}"
        for value in _duplicate_values(values)
    ]


def _validate_metadata(entries: list[BudgetEntry], today: date) -> list[str]:
    errors: list[str] = []
    for entry in entries:
        if not entry.owner:
            errors.append(f"{entry.path}: missing owner")
        if not entry.reason:
            errors.append(f"{entry.path}: missing reason")
        if entry.expires_on < today:
            errors.append(f"{entry.path}: expired on {entry.expires_on.isoformat()}")
    return errors


def _validate_schedule_ratchet(schedule: list[ScheduleEntry]) -> list[str]:
    errors: list[str] = []
    previous_cap: int | None = None
    for entry in sorted(schedule, key=lambda item: item.effective_on):
        if previous_cap is not None and entry.max_exclusions > previous_cap:
            errors.append(
                f"{entry.effective_on.isoformat()}: max_exclusions increases "
                f"from {previous_cap} to {entry.max_exclusions}"
            )
        previous_cap = entry.max_exclusions
    return errors


def _validate_no_ignore_errors(overrides: list[MypyOverride]) -> list[str]:
    errors: list[str] = []
    for override in overrides:
        if override.ignore_errors:
            modules = ", ".join(override.modules) or "<unknown>"
            errors.append(
                "mypy override uses ignore_errors=true; move debt into "
                f"mypy_exclusion_budget.json instead: {modules}"
            )
    return errors


def _validate_path_sets(
    pyproject_paths: list[str], budget_paths: list[str]
) -> list[str]:
    errors: list[str] = []
    budget_set = set(budget_paths)
    pyproject_set = set(pyproject_paths)
    for path in sorted(pyproject_set - budget_set):
        errors.append(f"{path}: pyproject exclusion is not present in budget")
    for path in sorted(budget_set - pyproject_set):
        errors.append(f"{path}: budget exclusion is not present in pyproject")
    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", default=str(DEFAULT_PYPROJECT))
    parser.add_argument("--budget", default=str(DEFAULT_BUDGET))
    parser.add_argument("--today", default=date.today().isoformat())
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the mypy exclusion budget check."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        today = _parse_iso_date(args.today, "today")
        exclusions = load_pyproject_exclusions(Path(args.pyproject))
        overrides = load_mypy_overrides(Path(args.pyproject))
        budget_entries, schedule = load_budget(Path(args.budget))
        coverage_gates = load_coverage_gates(Path(args.budget))
        errors = validate_budget(exclusions, budget_entries, schedule, today, overrides)
        errors.extend(validate_coverage_gates(coverage_gates, today))
    except (OSError, ValueError, tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"mypy exclusion budget failed: {exc}\n")
        return 1

    if errors:
        sys.stderr.write("mypy exclusion budget failed:\n")
        for error in errors:
            sys.stderr.write(f"  {error}\n")
        return 1

    cap = active_cap(schedule, today)
    sys.stdout.write(
        "mypy exclusion budget passed "
        f"({len(budget_entries)} exclusions, active cap {cap}; "
        f"{len(coverage_gates)} coverage gates)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
