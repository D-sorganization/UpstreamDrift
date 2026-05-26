#!/usr/bin/env python3
"""Enforce repo-local coverage thresholds for production-critical surfaces."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoveragePolicy:
    """Coverage threshold for a named group of source files."""

    name: str
    threshold: float
    prefixes: tuple[str, ...] = ()
    exact_paths: tuple[str, ...] = ()

    def matches(self, filename: str) -> bool:
        normalized = filename.replace("\\", "/")
        return normalized in self.exact_paths or normalized.startswith(self.prefixes)


DEFAULT_POLICIES: tuple[CoveragePolicy, ...] = (
    CoveragePolicy(
        name="api_routes_services",
        threshold=85.0,
        prefixes=("src/api/routes/", "src/api/services/"),
    ),
    CoveragePolicy(
        name="engine_core_control_interface",
        threshold=85.0,
        prefixes=("src/shared/python/engine_core/",),
        exact_paths=("src/shared/python/control_interface.py",),
    ),
    CoveragePolicy(
        name="task_management",
        threshold=80.0,
        exact_paths=("src/api/task_manager.py", "src/api/task_manager_durable.py"),
    ),
    CoveragePolicy(
        name="shared_utilities",
        threshold=70.0,
        prefixes=("src/shared/python/",),
    ),
)


def parse_coverage_report(report_path: Path) -> dict[str, dict[int, int]]:
    """Return per-file line hit data from a coverage.py Cobertura XML report."""
    root = ElementTree.parse(report_path).getroot()
    report: dict[str, dict[int, int]] = {}

    for class_node in root.findall(".//class"):
        filename = class_node.attrib.get("filename", "").replace("\\", "/")
        if not filename:
            continue
        line_hits = report.setdefault(filename, {})
        for line_node in class_node.findall("./lines/line"):
            number = int(line_node.attrib["number"])
            hits = int(line_node.attrib.get("hits", "0"))
            line_hits[number] = max(line_hits.get(number, 0), hits)

    return report


def find_policy_failures(
    report: dict[str, dict[int, int]],
    policies: tuple[CoveragePolicy, ...],
) -> list[str]:
    """Return human-readable failures for missing or under-threshold groups."""
    totals = {
        policy.name: {
            "covered": 0,
            "total": 0,
            "files": 0,
            "threshold": policy.threshold,
        }
        for policy in policies
    }

    for filename, line_hits in report.items():
        for policy in policies:
            if not policy.matches(filename):
                continue
            totals[policy.name]["files"] += 1
            totals[policy.name]["total"] += len(line_hits)
            totals[policy.name]["covered"] += sum(
                1 for hits in line_hits.values() if hits > 0
            )
            break

    failures: list[str] = []
    for policy in policies:
        totals_for_policy = totals[policy.name]
        files = int(totals_for_policy["files"])
        if files == 0:
            failures.append(
                f"{policy.name}: no matching files found in coverage report"
            )
            continue

        covered = int(totals_for_policy["covered"])
        total = int(totals_for_policy["total"])
        percent = (covered / total) * 100 if total else 0.0
        threshold = float(totals_for_policy["threshold"])
        if percent < threshold:
            failures.append(
                f"{policy.name}: {percent:.1f}% covered ({covered}/{total} lines), "
                f"threshold {threshold:.1f}% across {files} file(s)"
            )

    return failures


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("Usage: coverage_enforcer.py <coverage.xml>", file=sys.stderr)
        return 2

    report_path = Path(args[0])
    if not report_path.is_file():
        print(f"Coverage report not found: {report_path}", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    failures = find_policy_failures(
        parse_coverage_report(report_path), DEFAULT_POLICIES
    )

    if failures:
        logger.error("FAIL: coverage policy violations detected:\n")
        for failure in failures:
            logger.error("  %s", failure)
        return 1

    logger.info("OK: coverage policy thresholds satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
