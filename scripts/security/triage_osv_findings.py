#!/usr/bin/env python3
"""Summarize OSV findings with tier-aware vulnerability SLA deadlines."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

TIER_SLA_DAYS = {
    "core": {"critical": 1, "high": 7, "medium": 30, "low": 0},
    "extended": {"critical": 2, "high": 14, "medium": 60, "low": 0},
    "experimental": {"critical": 7, "high": 30, "medium": 90, "low": 0},
    "archived": {"critical": 0, "high": 0, "medium": 0, "low": 0},
}

PACKAGE_TIERS = {
    "drake": "extended",
    "meshcat": "extended",
    "pinocchio": "extended",
    "gymnasium": "experimental",
    "mediapipe": "experimental",
    "myosuite": "experimental",
    "opensim": "experimental",
    "stable-baselines3": "experimental",
}

CVSS_SCORE_PATTERN = re.compile(r"(?<!\d)(\d{1,2}(?:\.\d)?)(?!\d)")


@dataclass(frozen=True)
class FindingSummary:
    """Tier-aware OSV finding summary.

    Postconditions:
        ``sla_deadline`` is computed from severity and affected component tier.
    """

    vulnerability_id: str
    package: str
    tier: str
    severity: str
    sla_deadline: date


def compute_sla_deadline(tier: str, severity: str, detected_on: date) -> date:
    """Compute the response SLA deadline for a vulnerability finding."""
    if not isinstance(detected_on, date):
        raise TypeError("detected_on must be a datetime.date")
    normalized_tier = _normalize_required("tier", tier)
    normalized_severity = _normalize_required("severity", severity)
    if normalized_tier not in TIER_SLA_DAYS:
        raise ValueError(f"unsupported tier: {tier}")
    if normalized_severity not in TIER_SLA_DAYS[normalized_tier]:
        raise ValueError(f"unsupported severity: {severity}")
    days = TIER_SLA_DAYS[normalized_tier][normalized_severity]
    return detected_on + timedelta(days=days)


def summarize_finding(
    finding: dict[str, Any], detected_on: date | None = None
) -> FindingSummary:
    """Classify one OSV JSON finding and attach the triage SLA deadline."""
    if not isinstance(finding, dict):
        raise TypeError("finding must be a mapping")
    current_date = detected_on or date.today()
    package = _package_name(finding)
    tier = tier_for_package(package)
    severity = severity_for_finding(finding)
    vulnerability_id = _require_text("id", str(finding.get("id", "")))
    return FindingSummary(
        vulnerability_id=vulnerability_id,
        package=package,
        tier=tier,
        severity=severity,
        sla_deadline=compute_sla_deadline(tier, severity, current_date),
    )


def tier_for_package(package: str) -> str:
    """Return the dependency tier that owns a package."""
    normalized_package = _normalize_required("package", package)
    return PACKAGE_TIERS.get(normalized_package, "core")


def severity_for_finding(finding: dict[str, Any]) -> str:
    """Return SLA severity bucket for an OSV finding."""
    severity_entries = finding.get("severity", [])
    if isinstance(severity_entries, list):
        for entry in severity_entries:
            if isinstance(entry, dict):
                score = str(entry.get("score", ""))
                severity = severity_from_score(score)
                if severity != "low":
                    return severity
    database_specific = finding.get("database_specific", {})
    if isinstance(database_specific, dict):
        severity = str(database_specific.get("severity", "")).lower()
        if severity in {"critical", "high", "medium", "low"}:
            return severity
    return "low"


def severity_from_score(score: str) -> str:
    """Convert numeric or CVSS-vector score text into an SLA severity bucket."""
    normalized_score = _normalize_required("score", score)
    upper_score = normalized_score.upper()
    if upper_score.startswith("CVSS:") and _is_common_critical_vector(upper_score):
        return "critical"
    match = CVSS_SCORE_PATTERN.search(normalized_score)
    if match:
        return severity_from_cvss(float(match.group(1)))
    if "/C:H" in upper_score or "/I:H" in upper_score or "/A:H" in upper_score:
        return "high"
    return "low"


def severity_from_cvss(score: float) -> str:
    """Convert a CVSS numeric score into the repository SLA severity bucket."""
    if score < 0 or score > 10:
        raise ValueError("CVSS score must be between 0 and 10")
    if score >= 9:
        return "critical"
    if score >= 7:
        return "high"
    if score >= 4:
        return "medium"
    return "low"


def load_osv_findings(path: Path) -> list[dict[str, Any]]:
    """Load OSV JSON output and return vulnerability mappings."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise ValueError("OSV output must be a JSON object or list")
    results = payload.get("results", [])
    findings: list[dict[str, Any]] = []
    if isinstance(results, list):
        for result in results:
            if isinstance(result, dict):
                findings.extend(_packages_from_result(result))
    return findings


def _packages_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    packages = result.get("packages", [])
    findings: list[dict[str, Any]] = []
    if not isinstance(packages, list):
        return findings
    for package_entry in packages:
        if not isinstance(package_entry, dict):
            continue
        package = package_entry.get("package", {})
        vulnerabilities = package_entry.get("vulnerabilities", [])
        if not isinstance(vulnerabilities, list):
            continue
        for vulnerability in vulnerabilities:
            if isinstance(vulnerability, dict):
                finding = dict(vulnerability)
                finding["package"] = package
                findings.append(finding)
    return findings


def _package_name(finding: dict[str, Any]) -> str:
    package = finding.get("package", {})
    if isinstance(package, dict):
        return _normalize_required("package.name", str(package.get("name", "")))
    return _normalize_required("package", str(package))


def _normalize_required(name: str, value: str) -> str:
    return _require_text(name, value).lower()


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _is_common_critical_vector(score: str) -> bool:
    return all(
        part in score for part in ("AV:N", "AC:L", "PR:N", "UI:N", "C:H", "I:H", "A:H")
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osv-json", type=Path, required=True)
    parser.add_argument("--detected-on", default=date.today().isoformat())
    return parser.parse_args()


def main() -> int:
    """Print tiered OSV finding summaries as JSON lines."""
    args = parse_args()
    detected_on = date.fromisoformat(args.detected_on)
    summaries = [
        summarize_finding(finding, detected_on=detected_on)
        for finding in load_osv_findings(args.osv_json)
    ]
    for summary in summaries:
        print(
            json.dumps(
                {
                    "vulnerability_id": summary.vulnerability_id,
                    "package": summary.package,
                    "tier": summary.tier,
                    "severity": summary.severity,
                    "sla_deadline": summary.sla_deadline.isoformat(),
                },
                sort_keys=True,
            )
        )
    return 1 if summaries else 0


if __name__ == "__main__":
    raise SystemExit(main())
