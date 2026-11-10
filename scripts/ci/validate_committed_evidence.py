#!/usr/bin/env python3
"""CI entry: fail when committed evidence violates #14935 invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.shared.python.motion_matching.evidence_validator import (  # noqa: E402
    validate_committed_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: auto-detected)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable report",
    )
    args = parser.parse_args()

    report = validate_committed_evidence(args.repo_root)
    if args.json:
        print(
            json.dumps(
                {
                    "failed": report.failed,
                    "issues": [issue.__dict__ for issue in report.issues],
                    "waived": [issue.__dict__ for issue in report.waived],
                },
                indent=2,
            )
        )
    else:
        for issue in report.issues:
            print(f"FAIL [{issue.rule}] {issue.path}: {issue.message}")
        for issue in report.waived:
            print(
                f"WAIVED [{issue.rule}] {issue.path} ({issue.waiver_id}): "
                f"{issue.message}"
            )
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
