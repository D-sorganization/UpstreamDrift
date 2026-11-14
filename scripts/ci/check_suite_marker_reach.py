#!/usr/bin/env python3
"""Enforce suite-marker reach and selectability floor in CI (#14947).

Verifies that:
1. Mapped directory test files (tests/unit/**, tests/integration/**) have 100%
   reach under directory-based suite marker inference (floor >= 95.0%).
2. The overall suite selectability after directory auto-marking meets or exceeds
   the target floor (>= 85.0%).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.ci.audit_suite_marker_reach import audit_suite_markers
from scripts.ci.check_suite_marker_ratchet import iter_test_files
from tests.helpers.suite_marker_reach import infer_directory_suite_marker
from tests.support.suite_markers import BASELINE_PATH

SELECTABILITY_FLOOR_PCT = 85.0
DIRECTORY_REACH_FLOOR_PCT = 95.0


def check_directory_reach(
    tests_root: pathlib.Path, floor_pct: float = DIRECTORY_REACH_FLOOR_PCT
) -> dict[str, Any]:
    """Verify test files in mapped directories are covered by directory auto-marking."""
    unit_files = list(iter_test_files(tests_root / "unit"))
    int_files = list(iter_test_files(tests_root / "integration"))

    u_matched = sum(
        1 for f in unit_files if infer_directory_suite_marker(f.as_posix()) == "unit"
    )
    i_matched = sum(
        1
        for f in int_files
        if infer_directory_suite_marker(f.as_posix()) == "integration"
    )

    u_pct = (u_matched / len(unit_files) * 100.0) if unit_files else 100.0
    i_pct = (i_matched / len(int_files) * 100.0) if int_files else 100.0

    passed = bool(u_pct >= floor_pct and i_pct >= floor_pct)
    return {
        "unit_files": len(unit_files),
        "unit_matched": u_matched,
        "unit_pct": round(u_pct, 2),
        "integration_files": len(int_files),
        "integration_matched": i_matched,
        "integration_pct": round(i_pct, 2),
        "floor_pct": floor_pct,
        "passed": passed,
    }


def check_suite_marker_reach(
    tests_root: pathlib.Path,
    baseline_path: pathlib.Path | None = None,
    selectability_floor: float = SELECTABILITY_FLOOR_PCT,
    directory_floor: float = DIRECTORY_REACH_FLOOR_PCT,
) -> tuple[bool, str]:
    """Run all suite-marker reach checks and return (passed, message)."""
    dir_res = check_directory_reach(tests_root, directory_floor)
    if not dir_res["passed"]:
        msg = (
            f"Directory reach check FAILED:\n"
            f"  tests/unit: {dir_res['unit_matched']}/{dir_res['unit_files']} ({dir_res['unit_pct']}%, floor: {directory_floor}%)\n"
            f"  tests/integration: {dir_res['integration_matched']}/{dir_res['integration_files']} ({dir_res['integration_pct']}%, floor: {directory_floor}%)"
        )
        return False, msg

    audit_res = audit_suite_markers(tests_root, baseline_path)
    census = audit_res["census"]
    after_pct = float(census["after"]["selected_pct"])
    if after_pct < selectability_floor:
        msg = (
            f"Selectability floor check FAILED:\n"
            f"  After auto-marking: {census['after']['summary_line']}\n"
            f"  Required floor: {selectability_floor:.1f}%\n"
            f"  Target met: False"
        )
        return False, msg

    msg = (
        f"Suite marker reach checks PASSED:\n"
        f"  Directory reach: tests/unit {dir_res['unit_pct']}%, tests/integration {dir_res['integration_pct']}% (floor: {directory_floor}%)\n"
        f"  Selectability: {census['after']['summary_line']} (floor: {selectability_floor}%)\n"
        f"  Explicitly marked: {census['explicitly_marked']} | Auto-marked: {census['auto_marked_unit'] + census['auto_marked_integration']}"
    )
    return True, msg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-root", type=pathlib.Path, default=REPO_ROOT / "tests")
    parser.add_argument("--baseline", type=pathlib.Path, default=BASELINE_PATH)
    parser.add_argument(
        "--selectability-floor", type=float, default=SELECTABILITY_FLOOR_PCT
    )
    parser.add_argument(
        "--directory-floor", type=float, default=DIRECTORY_REACH_FLOOR_PCT
    )
    args = parser.parse_args(argv)

    passed, msg = check_suite_marker_reach(
        args.tests_root,
        args.baseline,
        args.selectability_floor,
        args.directory_floor,
    )
    print(msg)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
