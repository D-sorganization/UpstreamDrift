#!/usr/bin/env python3
"""CLI entrypoint to check engine test floors against JUnit XML (issue #14948)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "scripts" / "config" / "engine_test_floor.json"

from src.shared.python.engine_core.engine_lane_guard import (
    verify_junit_engine_floor,
)  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check engine test floor.")
    parser.add_argument(
        "--junit",
        "--junit-xml",
        "--junitxml",
        dest="junit",
        required=True,
        type=Path,
        help="Path to JUnit XML",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to engine_test_floor.json (default: scripts/config/engine_test_floor.json)",
    )
    parser.add_argument(
        "--lane",
        "--engine",
        dest="lane",
        required=True,
        help="Engine lane name (e.g. mujoco, pinocchio, drake, opensim, jaxsim)",
    )
    parser.add_argument(
        "--min-executed",
        dest="min_executed",
        type=int,
        default=None,
        help="Optional override for min_executed floor",
    )
    args = parser.parse_args(argv)

    try:
        report = verify_junit_engine_floor(args.junit, args.config, args.lane)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.min_executed is not None and report.executed < args.min_executed:
        print(
            f"ERROR: executed {report.executed} < required --min-executed {args.min_executed}",
            file=sys.stderr,
        )
        return 1

    print(report.message)
    return 0 if report.success else 1


if __name__ == "__main__":
    sys.exit(main())
