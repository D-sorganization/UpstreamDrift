#!/usr/bin/env python3
"""Per-module coverage gate checker for UpstreamDrift.

Reads coverage gates from ``scripts/config/mypy_exclusion_budget.json`` and enforces
minimum coverage thresholds using Cobertura XML or JSON coverage reports.

Usage:
    python3 -m pytest --cov=src --cov-report=xml:coverage.xml -q
    python3 scripts/check_coverage_gates.py --report coverage.xml
    python3 scripts/check_coverage_gates.py --report coverage.json

Design by Contract:
    Pre:  A valid coverage report (.xml or .json) and budget file exist.
          Each budget entry has 0 <= min_coverage <= 100 and path ending with '/'.
    Post: Returns 0 if all gates meet or exceed min_coverage.
          Returns 1 if one or more gates fall below min_coverage.
          Returns 2 on file/config error, or if any gate matches zero files.

Exit codes:
    0 — all coverage gates passed
    1 — one or more gates below floor
    2 — configuration / file error (or gate matching zero files)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import defusedxml.ElementTree as ET  # XXE-safe (bandit B314, #10989)

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element

DEFAULT_BUDGET = Path("scripts/config/mypy_exclusion_budget.json")


@dataclass(frozen=True)
class CoverageGate:
    """An accountable per-package coverage gate."""

    name: str
    path: str
    min_coverage: float


@dataclass(frozen=True)
class GateResult:
    """Evaluation result for a single coverage gate."""

    name: str
    path: str
    covered: int
    total: int
    percent: float
    min_coverage: float
    matching_files: int
    status: str


def load_budget_gates(budget_path: Path | str) -> list[CoverageGate]:
    """Load and validate coverage gate configurations from the budget file.

    Preconditions:
        budget_path points to an existing valid JSON file.
        Each coverage gate in the budget must have non-empty name,
        path ending with '/', and 0.0 <= min_coverage <= 100.0.
    Postconditions:
        Returns a list of validated CoverageGate objects.
    """
    path = Path(budget_path)
    if not path.is_file():
        raise FileNotFoundError(f"Budget file not found: {path}")

    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in budget file: {path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Budget file must contain a JSON object: {path}")

    raw_gates = data.get("coverage_gates")
    if not isinstance(raw_gates, list) or not raw_gates:
        raise ValueError("Budget file must contain a non-empty 'coverage_gates' list")

    gates: list[CoverageGate] = []
    for entry in raw_gates:
        if not isinstance(entry, dict):
            raise ValueError("Each coverage gate must be an object")

        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError("Coverage gate name must be a non-empty string")

        raw_path = str(entry.get("path", "")).strip().replace("\\", "/")
        if not raw_path:
            raise ValueError(f"Gate '{name}' path must be a non-empty string")
        if not raw_path.endswith("/"):
            raise ValueError(f"Gate '{name}' path must end with '/': {raw_path}")

        raw_min = entry.get("min_coverage")
        if not isinstance(raw_min, int | float) or isinstance(raw_min, bool):
            raise ValueError(f"Gate '{name}' min_coverage must be a number: {raw_min}")

        min_cov = float(raw_min)
        if not (0.0 <= min_cov <= 100.0):
            raise ValueError(
                f"Gate '{name}' min_coverage must be between 0 and 100: {min_cov}"
            )

        gates.append(CoverageGate(name=name, path=raw_path, min_coverage=min_cov))

    return gates


def normalize_repo_path(path_str: str, repo_root: Path | None = None) -> str:
    """Normalize a filepath to be repository-relative using posix separators."""
    norm = path_str.replace("\\", "/").removeprefix("./")
    if repo_root is not None:
        root_str = str(repo_root.resolve()).replace("\\", "/") + "/"
        norm = norm.removeprefix(root_str)
    return norm


def _source_prefix(source: str, repo_root: Path) -> str:
    """Repository-relative prefix of a Cobertura ``<source>`` directory.

    Raises ``ValueError`` when an absolute source lies outside ``repo_root``, since
    its files cannot be matched to repository gate paths.
    """
    src = Path(source)
    if src.is_absolute():
        try:
            src = src.resolve().relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Coverage source {source} is outside repository root {repo_root}"
            ) from exc
    prefix = src.as_posix().strip("/")
    return "" if prefix in ("", ".") else prefix + "/"


def _xml_source_prefixes(root: Element, repo_root: Path) -> list[str]:
    """Prefixes for every ``<source>``; no sources means names are already repo-relative."""
    sources = [s.text.strip() for s in root.findall("./sources/source") if s.text]
    if not sources:
        return [""]
    return [_source_prefix(source, repo_root) for source in sources]


def _resolve_xml_filename(filename: str, prefixes: list[str], repo_root: Path) -> str:
    """Map a Cobertura filename to a repo path via the first source that contains it."""
    norm = filename.replace("\\", "/").removeprefix("./")
    if len(prefixes) == 1:
        return prefixes[0] + norm
    for prefix in prefixes:
        if (repo_root / prefix / norm).is_file():
            return prefix + norm
    raise ValueError(f"Coverage file {filename} not found under any <source>")


def load_coverage_xml(
    path: Path | str, repo_root: Path | None = None
) -> dict[str, tuple[int, int]]:
    """Load per-file coverage from a Cobertura XML report."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Coverage report not found: {path}")

    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Failed to parse coverage XML: {path}") from exc

    base = repo_root if repo_root is not None else Path.cwd()
    prefixes = _xml_source_prefixes(root, base)
    file_line_hits: dict[str, dict[int, bool]] = {}
    for class_node in root.findall(".//class"):
        raw_filename = class_node.attrib.get("filename", "")
        if not raw_filename:
            continue
        norm_path = _resolve_xml_filename(raw_filename, prefixes, base)
        line_map = file_line_hits.setdefault(norm_path, {})
        for line in class_node.findall("./lines/line"):
            line_no = int(line.attrib.get("number", 0))
            hits = int(line.attrib.get("hits", "0"))
            line_map[line_no] = line_map.get(line_no, False) or (hits > 0)

    files_cov: dict[str, tuple[int, int]] = {}
    for norm_path, line_map in file_line_hits.items():
        total = len(line_map)
        covered = sum(1 for hit in line_map.values() if hit)
        files_cov[norm_path] = (covered, total)

    return files_cov


def load_coverage_json(
    path: Path | str, repo_root: Path | None = None
) -> dict[str, tuple[int, int]]:
    """Load per-file coverage from a coverage.py JSON report."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Coverage report not found: {path}")

    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Failed to parse coverage JSON: {path}") from exc

    if not isinstance(data, dict) or "files" not in data:
        raise ValueError(f"Coverage JSON missing 'files' key: {path}")

    files_cov: dict[str, tuple[int, int]] = {}
    for raw_filename, file_info in data["files"].items():
        norm_path = normalize_repo_path(raw_filename, repo_root)
        summary = file_info.get("summary", {})
        total = summary.get("num_statements", 0)
        covered = summary.get("covered_lines", 0)
        files_cov[norm_path] = (covered, total)

    return files_cov


def load_coverage_report(
    path: Path | str, repo_root: Path | None = None
) -> dict[str, tuple[int, int]]:
    """Load coverage data from XML or JSON chosen by file extension."""
    report_path = Path(path)
    ext = report_path.suffix.lower()
    if ext == ".xml":
        return load_coverage_xml(report_path, repo_root)
    if ext == ".json":
        return load_coverage_json(report_path, repo_root)
    raise ValueError(
        f"Unsupported coverage report extension '{ext}': {report_path} (must be .xml or .json)"
    )


def evaluate_gates(
    gates: list[CoverageGate],
    files_coverage: dict[str, tuple[int, int]],
) -> tuple[dict[str, GateResult], int]:
    """Evaluate coverage data against gates.

    Returns:
        tuple of (results_dict, exit_code)
        exit_code 0: all passed
        exit_code 1: one or more gates below floor
        exit_code 2: one or more gates matched zero files (misconfigured)
    """
    results: dict[str, GateResult] = {}
    has_misconfigured = False
    has_failure = False

    for gate in gates:
        matching_files = [f for f in files_coverage if f.startswith(gate.path)]
        if not matching_files:
            results[gate.name] = GateResult(
                name=gate.name,
                path=gate.path,
                covered=0,
                total=0,
                percent=0.0,
                min_coverage=gate.min_coverage,
                matching_files=0,
                status="MISCONFIGURED",
            )
            has_misconfigured = True
            continue

        covered = sum(files_coverage[f][0] for f in matching_files)
        total = sum(files_coverage[f][1] for f in matching_files)
        pct = (covered / total * 100.0) if total > 0 else 0.0
        passed = pct >= gate.min_coverage
        status = "PASS" if passed else "FAIL"
        if not passed:
            has_failure = True

        results[gate.name] = GateResult(
            name=gate.name,
            path=gate.path,
            covered=covered,
            total=total,
            percent=pct,
            min_coverage=gate.min_coverage,
            matching_files=len(matching_files),
            status=status,
        )

    if has_misconfigured:
        exit_code = 2
    elif has_failure:
        exit_code = 1
    else:
        exit_code = 0

    return results, exit_code


def format_summary_table(results: dict[str, GateResult]) -> str:
    """Format gate results into a human-readable summary table."""
    lines = [
        "Coverage Gate Summary:",
        f"  {'Gate':<26} {'Path':<36} {'Files':>5} {'Covered':>8} {'Total':>8} {'Percent':>8} {'Min %':>8} {'Status':<14}",
        f"  {'-' * 26} {'-' * 36} {'-' * 5} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 14}",
    ]
    for res in results.values():
        if res.status == "MISCONFIGURED":
            pct_str = "N/A"
            status_str = "MISCONFIGURED"
        else:
            pct_str = f"{res.percent:.1f}%"
            status_str = res.status
        lines.append(
            f"  {res.name:<26} {res.path:<36} {res.matching_files:>5} {res.covered:>8} {res.total:>8} {pct_str:>8} {res.min_coverage:>7.1f}% {status_str:<14}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the coverage gate checker."""
    parser = argparse.ArgumentParser(
        description="Check per-module coverage gates for UpstreamDrift."
    )
    parser.add_argument(
        "--report",
        default="coverage.xml",
        help="Path to coverage report (XML or JSON, default: coverage.xml)",
    )
    parser.add_argument(
        "--budget",
        default=None,
        help="Path to budget JSON (default: scripts/config/mypy_exclusion_budget.json)",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root that report paths are relative to (default: cwd)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Maintained for backward compatibility.",
    )
    args = parser.parse_args(argv)

    budget_path = Path(args.budget) if args.budget else DEFAULT_BUDGET
    if not budget_path.is_file():
        repo_budget = (
            Path(__file__).resolve().parent / "config" / "mypy_exclusion_budget.json"
        )
        if repo_budget.is_file():
            budget_path = repo_budget

    try:
        gates = load_budget_gates(budget_path)
        repo_root = Path(args.repo_root) if args.repo_root else Path.cwd()
        coverage_data = load_coverage_report(args.report, repo_root)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"Configuration or file error: {exc}\n")
        return 2

    results, exit_code = evaluate_gates(gates, coverage_data)
    print(format_summary_table(results))

    if exit_code == 2:
        misconfigured = [
            r.name for r in results.values() if r.status == "MISCONFIGURED"
        ]
        sys.stderr.write(
            f"\nERROR: Coverage gate(s) matched zero files (misconfigured): {', '.join(misconfigured)}\n"
        )
    elif exit_code == 1:
        failed = [
            f"{r.name} ({r.percent:.1f}% < {r.min_coverage:.1f}%)"
            for r in results.values()
            if r.status == "FAIL"
        ]
        sys.stderr.write(f"\nFAIL: Coverage gate(s) below floor: {', '.join(failed)}\n")
    else:
        print(f"\nAll {len(gates)} coverage gates passed.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
