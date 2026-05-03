#!/usr/bin/env python3
"""Validate pip-audit waiver metadata and emit ignore flags."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

SCHEMA_VERSION = 1
REQUIRED_FIELDS = ("vuln", "package", "reason", "tracked_in", "expires_on")


@dataclass(frozen=True)
class Waiver:
    """Structured representation of one pip-audit waiver."""

    vuln: str
    package: str
    reason: str
    tracked_in: str
    expires_on: date


def _load_manifest(path: Path) -> dict[str, object]:
    """Read and validate the top-level waiver manifest object."""
    if not path.exists():
        raise FileNotFoundError(f"pip-audit waiver file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8") or "{}")
    if not isinstance(raw, dict):
        raise ValueError("waiver file must contain a JSON object")
    schema_version = raw.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {schema_version}")
    return raw


def _parse_waiver(item: object) -> Waiver:
    """Parse and validate one waiver entry."""
    if not isinstance(item, dict):
        raise ValueError("each waiver entry must be a mapping")
    missing_fields = [field for field in REQUIRED_FIELDS if field not in item]
    if missing_fields:
        raise ValueError(f"missing waiver field: {missing_fields[0]}")
    try:
        waiver = Waiver(
            vuln=str(item["vuln"]).strip(),
            package=str(item["package"]).strip(),
            reason=str(item["reason"]).strip(),
            tracked_in=str(item["tracked_in"]).strip(),
            expires_on=date.fromisoformat(str(item["expires_on"]).strip()),
        )
    except ValueError as exc:
        raise ValueError("expires_on must be an ISO date") from exc

    _validate_waiver_metadata(waiver)
    return waiver


def _validate_waiver_metadata(waiver: Waiver) -> None:
    """Validate required waiver metadata after parsing."""
    if (
        not waiver.vuln
        or not waiver.package
        or not waiver.reason
        or not waiver.tracked_in
    ):
        raise ValueError("waiver vuln, package, reason, and tracked_in are required")
    if not waiver.tracked_in.startswith("#"):
        raise ValueError("tracked_in must reference a GitHub issue")


def load_waivers(path: Path) -> list[Waiver]:
    """Load waiver definitions from JSON.

    Preconditions:
        path points to a JSON object with a top-level ``waivers`` list.

    Postconditions:
        Every returned waiver has non-empty issue metadata and an expiry date.
    """
    raw = _load_manifest(path)
    items = raw.get("waivers", [])
    if not isinstance(items, list):
        raise ValueError("waivers must be a list")

    return [_parse_waiver(item) for item in items]


def find_expired_waivers(
    waivers: list[Waiver], today: date | None = None
) -> list[Waiver]:
    """Return waivers whose expiry date has passed."""
    current = today or date.today()
    return [waiver for waiver in waivers if waiver.expires_on < current]


def load_reported_vulns(path: Path) -> set[tuple[str, str]]:
    """Load ``(package, vuln)`` pairs from a pip-audit JSON report."""
    raw = json.loads(path.read_text(encoding="utf-8") or "{}")
    dependencies = raw.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise ValueError("pip-audit report dependencies must be a list")

    reported: set[tuple[str, str]] = set()
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            continue
        package = str(dependency.get("name", "")).strip()
        vulns = dependency.get("vulns", [])
        if not package or not isinstance(vulns, list):
            continue
        for vuln in vulns:
            if isinstance(vuln, dict):
                vuln_id = str(vuln.get("id", "")).strip()
                if vuln_id:
                    reported.add((package.lower(), vuln_id))
    return reported


def find_stale_waivers(
    waivers: list[Waiver], reported_vulns: set[tuple[str, str]]
) -> list[Waiver]:
    """Return waivers that no longer match pip-audit findings."""
    return [
        waiver
        for waiver in waivers
        if (waiver.package.lower(), waiver.vuln) not in reported_vulns
    ]


def build_ignore_flags(waivers: list[Waiver]) -> list[str]:
    """Build CLI flags for pip-audit."""
    flags: list[str] = []
    for waiver in waivers:
        flags.extend(["--ignore-vuln", waiver.vuln])
    return flags


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--waiver-file",
        type=Path,
        default=Path("scripts/config/pip_audit_waivers.json"),
        help="Path to the pip-audit waiver manifest.",
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        help="Optional pip-audit JSON report used to reject stale waivers.",
    )
    return parser.parse_args()


def main() -> int:
    """Validate waivers and print ignore flags for shell consumption."""
    args = parse_args()
    waivers = load_waivers(args.waiver_file)
    expired = find_expired_waivers(waivers)
    if expired:
        for waiver in expired:
            print(
                f"Expired pip-audit waiver: {waiver.vuln} for {waiver.package} "
                f"(expired {waiver.expires_on.isoformat()})"
            )
        return 1

    if args.audit_report is not None:
        stale = find_stale_waivers(waivers, load_reported_vulns(args.audit_report))
        if stale:
            for waiver in stale:
                print(
                    f"Stale pip-audit waiver: {waiver.vuln} for {waiver.package} "
                    "is not present in the current audit report"
                )
            return 1

    for flag in build_ignore_flags(waivers):
        print(flag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
