#!/usr/bin/env python3
"""Enforce repo-local coverage thresholds for production-critical surfaces."""

from __future__ import annotations

import argparse
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
        filename = _normalize_path(class_node.attrib.get("filename", ""))
        if not filename:
            continue
        line_hits = report.setdefault(filename, {})
        for line_node in class_node.findall("./lines/line"):
            number = int(line_node.attrib["number"])
            hits = int(line_node.attrib.get("hits", "0"))
            line_hits[number] = max(line_hits.get(number, 0), hits)

    return report


def load_changed_files(path: Path) -> list[str]:
    """Return normalized changed-file paths from a newline-delimited file."""
    if not path.is_file():
        raise FileNotFoundError(f"changed-file list not found: {path}")

    return [
        normalized
        for line in path.read_text(encoding="utf-8").splitlines()
        if (normalized := _normalize_path(line.strip()))
    ]


def find_changed_file_failures(
    report: dict[str, dict[int, int]],
    policies: tuple[CoveragePolicy, ...],
    changed_files: list[str],
) -> list[str]:
    """Return policy failures for changed files covered by a policy.

    This PR-mode ratchet is intentionally narrower than full-suite group
    enforcement: it ignores changed files outside policy groups, but fails when
    a changed policy file is missing from targeted coverage or falls below its
    policy threshold.
    """
    normalized_report = {
        _normalize_path(filename): hits for filename, hits in report.items()
    }
    failures: list[str] = []

    for changed_file in sorted({_normalize_path(path) for path in changed_files}):
        policy = _first_matching_policy(changed_file, policies)
        if policy is None:
            continue

        line_hits = normalized_report.get(changed_file)
        if line_hits is None:
            failures.append(
                f"{changed_file}: missing from coverage report (policy {policy.name})"
            )
            continue

        total = len(line_hits)
        if total == 0:
            continue

        covered = sum(1 for hits in line_hits.values() if hits > 0)
        percent = (covered / total) * 100
        if percent < policy.threshold:
            failures.append(
                f"{changed_file}: {percent:.1f}% covered ({covered}/{total} lines), "
                f"threshold {policy.threshold:.1f}% for policy {policy.name}"
            )

    return failures


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


def _first_matching_policy(
    filename: str,
    policies: tuple[CoveragePolicy, ...],
) -> CoveragePolicy | None:
    normalized = _normalize_path(filename)
    for policy in policies:
        if policy.matches(normalized):
            return policy
    return None


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Enforce repo-local coverage thresholds."
    )
    parser.add_argument("coverage_xml", help="Path to Cobertura coverage XML.")
    parser.add_argument(
        "--changed-files",
        type=Path,
        help=(
            "Newline-delimited changed-file list for PR-targeted coverage. "
            "Only changed files matching coverage policies are enforced."
        ),
    )
    namespace = parser.parse_args(args)

    report_path = Path(namespace.coverage_xml)
    if not report_path.is_file():
        print(f"Coverage report not found: {report_path}", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = parse_coverage_report(report_path)
    if namespace.changed_files is None:
        failures = find_policy_failures(report, DEFAULT_POLICIES)
    else:
        try:
            changed_files = load_changed_files(namespace.changed_files)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        failures = find_changed_file_failures(
            report,
            DEFAULT_POLICIES,
            changed_files,
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
