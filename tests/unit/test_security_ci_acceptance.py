"""Issue #3844 security CI acceptance contract tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.unit


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_pip_audit_waivers_use_issue_schema() -> None:
    manifest = json.loads(_read("scripts/config/pip_audit_waivers.json"))

    assert manifest["schema_version"] == 1
    for waiver in manifest["waivers"]:
        assert {"vuln", "package", "reason", "tracked_in", "expires_on"} <= set(waiver)
        assert waiver["tracked_in"].startswith("#")


def test_standard_ci_runs_blocking_semgrep_and_trivy() -> None:
    workflow = _read(".github/workflows/ci-standard.yml")
    trivy_test = _read("tests/security/test_no_secrets_in_tree.py")

    assert "semgrep --config p/python --config p/security-audit" in workflow
    assert "--config p/owasp-top-ten --error" in workflow
    assert "aquasecurity/trivy-action@" in workflow
    assert "scan-type: fs" in workflow
    # Accept either YAML quote style — prettier normalises scalars and the
    # behaviour of exit-code 1 (fail the build on findings) is identical.
    assert ('exit-code: "1"' in workflow) or ("exit-code: '1'" in workflow)
    assert "trivy" in trivy_test
    assert '"fs"' in trivy_test


def test_docker_security_scan_blocks_high_and_critical() -> None:
    workflow = _read(".github/workflows/docker-security-scan.yml")

    assert 'severity: "CRITICAL,HIGH"' in workflow
    assert 'exit-code: "1"' in workflow
    assert "Fail the build on HIGH or CRITICAL vulnerabilities." in workflow
    assert "ignore-unfixed: true" not in workflow


def test_codeowners_has_two_owners_for_critical_paths() -> None:
    codeowners = _read(".github/CODEOWNERS")
    required_paths = {".github/workflows/", "scripts/", "/src/"}
    owners_by_path: dict[str, set[str]] = {}

    for raw_line in codeowners.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        owners_by_path[parts[0]] = {part for part in parts[1:] if part.startswith("@")}

    for path in required_paths:
        assert len(owners_by_path.get(path, set())) >= 2


def test_branch_protection_contract_is_documented() -> None:
    doc = _read("docs/operations/branch-protection.md")

    required_terms = [
        "CODEOWNERS",
        "ci-standard",
        "up-to-date with `main`",
        "linear history",
        "Tags `v*.*.*`",
        "Force-push to `main` is denied",
    ]
    for term in required_terms:
        assert term in doc

    assert re.search(r"at least one approving review", doc, re.IGNORECASE)
