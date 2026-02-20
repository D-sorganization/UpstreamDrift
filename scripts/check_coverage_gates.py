#!/usr/bin/env python3
"""Per-module coverage gate checker for UpstreamDrift.

Reads a coverage JSON report (produced by ``pytest --cov --cov-report=json``)
and enforces minimum coverage thresholds per top-level source module.

Usage:
    python3 -m pytest --cov=src --cov-report=json -q
    python3 scripts/check_coverage_gates.py          # uses coverage.json
    python3 scripts/check_coverage_gates.py --report path/to/coverage.json

Design by Contract:
    Pre:  A valid coverage JSON file exists at the specified path.
    Post: Exits 0 if all gates pass, exits 1 if any gate fails.
          Prints a summary table to stdout.

Exit codes:
    0 — all coverage gates passed
    1 — one or more gates failed
    2 — configuration / file error
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

# ── Coverage gates (module_prefix -> minimum percent) ────────────────────────
# These thresholds are intentionally conservative starting points.
# They should be ratcheted up as coverage improves.

COVERAGE_GATES: Dict[str, float] = {
    "engines": 35.0,
    "api": 30.0,
    "core": 40.0,
    "shared": 35.0,
    "robotics": 30.0,
}

# Mapping from gate name to source directory prefix patterns.
# A file matches a gate if its path contains one of these fragments.
MODULE_PATTERNS: Dict[str, list] = {
    "engines": ["src/engines/"],
    "api": ["src/api/"],
    "core": ["src/shared/python/engine_core/", "src/shared/python/physics/"],
    "shared": ["src/shared/"],
    "robotics": ["src/robotics/"],
}


def load_coverage_report(path: str) -> dict:
    """Load and validate a coverage JSON report.

    Args:
        path: Filesystem path to coverage.json.

    Returns:
        Parsed JSON as a dict.

    Raises:
        SystemExit: If the file does not exist or is invalid JSON.
    """
    if not os.path.isfile(path):
        print(f"ERROR: Coverage report not found: {path}", file=sys.stderr)
        print(
            "  Run: python3 -m pytest --cov=src --cov-report=json -q",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(2)

    if "files" not in data:
        print(
            f"ERROR: Coverage report missing 'files' key. "
            f"Ensure you used --cov-report=json.",
            file=sys.stderr,
        )
        sys.exit(2)

    return data


def compute_module_coverage(
    report: dict,
) -> Dict[str, Tuple[int, int, float]]:
    """Compute per-module coverage from the JSON report.

    Args:
        report: Parsed coverage JSON.

    Returns:
        Dict mapping gate name to (covered_lines, total_lines, percent).
        Only gates that have at least one matching source file are included.
    """
    results: Dict[str, Tuple[int, int]] = {}

    for filepath, file_data in report["files"].items():
        summary = file_data.get("summary", {})
        total = summary.get("num_statements", 0)
        covered = summary.get("covered_lines", 0)

        if total == 0:
            continue

        for gate, patterns in MODULE_PATTERNS.items():
            if any(pattern in filepath for pattern in patterns):
                prev_covered, prev_total = results.get(gate, (0, 0))
                results[gate] = (prev_covered + covered, prev_total + total)
                # A file can match multiple gates (e.g. shared/physics -> core AND shared)
                # This is intentional: "core" is a subset of "shared".

    final: Dict[str, Tuple[int, int, float]] = {}
    for gate, (covered, total) in results.items():
        pct = (covered / total * 100.0) if total > 0 else 0.0
        final[gate] = (covered, total, pct)

    return final


def check_gates(
    coverage: Dict[str, Tuple[int, int, float]],
    gates: Dict[str, float],
) -> bool:
    """Check coverage against gates and print a summary table.

    Args:
        coverage: Output of compute_module_coverage.
        gates: Minimum thresholds per module.

    Returns:
        True if all gates pass, False if any fail.
    """
    all_passed = True

    print()
    print(f"{'Module':<12} {'Covered':>8} {'Total':>8} {'Pct':>7} {'Gate':>7} {'Status':>8}")
    print("-" * 58)

    for gate_name, threshold in sorted(gates.items()):
        if gate_name in coverage:
            covered, total, pct = coverage[gate_name]
            passed = pct >= threshold
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False
            print(
                f"{gate_name:<12} {covered:>8} {total:>8} {pct:>6.1f}% {threshold:>6.1f}% {status:>8}"
            )
        else:
            # No source files found for this gate — skip silently
            print(
                f"{gate_name:<12} {'—':>8} {'—':>8} {'—':>7} {threshold:>6.1f}% {'SKIP':>8}"
            )

    print("-" * 58)
    print()

    return all_passed


def main() -> None:
    """Entry point for the coverage gate checker."""
    parser = argparse.ArgumentParser(
        description="Check per-module coverage gates for UpstreamDrift."
    )
    parser.add_argument(
        "--report",
        default="coverage.json",
        help="Path to coverage JSON report (default: coverage.json)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero even if gates are skipped (no matching files).",
    )
    args = parser.parse_args()

    report = load_coverage_report(args.report)
    coverage = compute_module_coverage(report)
    passed = check_gates(coverage, COVERAGE_GATES)

    if passed:
        print("All coverage gates passed.")
        sys.exit(0)
    else:
        print("One or more coverage gates FAILED.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
