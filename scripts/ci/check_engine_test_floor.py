#!/usr/bin/env python3
"""CLI entrypoint to check engine test floors against JUnit XML (issue #14948)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.shared.python.engine_core.engine_lane_guard import verify_junit_engine_floor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check engine test floor.")
    parser.add_argument("--junit", required=True, type=Path, help="Path to JUnit XML")
    parser.add_argument(
        "--config", required=True, type=Path, help="Path to engine_test_floor.json"
    )
    parser.add_argument("--lane", required=True, help="Engine lane name")
    args = parser.parse_args(argv)

    try:
        report = verify_junit_engine_floor(args.junit, args.config, args.lane)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(report.message)
    return 0 if report.success else 1


if __name__ == "__main__":
    sys.exit(main())
