"""Issue #3844 security CI acceptance contract tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

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


def test_standard_ci_uses_locked_python_dependencies_for_dev_and_audit() -> None:
    workflow = _read(".github/workflows/ci-standard.yml")
    makefile = _read("Makefile")
    dev_lock = _read("requirements-dev.lock")

    assert "requirements*.lock" in workflow
    assert (
        "python3 -m piptools compile --extra dev --extra gui-test "
        "-o requirements-dev.lock pyproject.toml"
    ) in makefile
    assert (
        "python -m pip install --no-cache-dir --ignore-installed -r requirements-dev.lock"
        in workflow
    )
    assert (
        "python3 -m pip install --no-cache-dir --ignore-installed -r requirements.lock"
        in workflow
    )
    assert 'python -m pip install --no-cache-dir --no-deps -e ".[dev]"' in workflow
    assert (
        '"$python" -m pip install --no-cache-dir --no-deps -e ".[gui-test]"' in workflow
    )
    assert "-c requirements-dev.lock" not in workflow

    assert (
        '"$audit_python" -m pip_audit -r requirements-dev.lock --format json '
        "--output pip-audit-report.json"
    ) in workflow
    assert (
        '"$audit_python" -m pip_audit -r requirements-dev.lock '
        '"${pip_audit_waiver_flags[@]}"'
    ) in workflow
    assert "python -m pip_audit -r requirements.lock" in workflow

    assert 'pip install --no-cache-dir -e ".[dev]"' not in workflow
    assert 'pip install --no-cache-dir -e ".[dev,gui-test]"' not in workflow
    assert "pip-audit\n" not in workflow

    for package in ("pyqt6==", "pyqt6-qt6==", "pyqt6-sip==", "pytest-qt=="):
        assert package in dev_lock.lower()

    assert "idna==3.18" in dev_lock
    assert "idna==3.11" not in dev_lock
    assert "pygments==2.20.0" in dev_lock
    assert "pygments==2.19.2" not in dev_lock


def test_ci_long_running_gates_emit_heartbeat_logs() -> None:
    standard = _read(".github/workflows/ci-standard.yml")
    optional = _read(".github/workflows/ci-optional-stack.yml")

    assert "- name: Setup Node for UI audit" in standard
    assert 'node-version: "24"' in standard
    assert "npm run test:run -- --testTimeout=15000" in standard
    assert 'run_with_heartbeat "pip-audit JSON report"' in standard
    assert 'run_with_heartbeat "pip-audit waiver-enforced scan"' in standard
    assert 'run_with_heartbeat "core pytest lane"' in standard
    assert 'run_with_heartbeat "optional-stack unit target $target"' in optional
    assert "still running at $(date -u +%H:%M:%SZ)" in standard
    assert "still running at $(date -u +%H:%M:%SZ)" in optional


def test_standard_ci_shell_continuations_are_not_split_by_blank_lines() -> None:
    workflow = _read(".github/workflows/ci-standard.yml")
    lines = workflow.splitlines()

    blank_continuations = [
        line_number
        for line_number, line in enumerate(lines[:-1], start=1)
        if line.rstrip().endswith("\\") and not lines[line_number].strip()
    ]

    assert blank_continuations == []


def test_docker_security_scan_blocks_high_and_critical() -> None:
    workflow = _read(".github/workflows/docker-security-scan.yml")
    parsed = yaml.safe_load(workflow)
    steps = parsed["jobs"]["trivy-scan"]["steps"]
    sarif_scan = next(
        step for step in steps if step["name"] == "Run Trivy vulnerability scanner"
    )
    table_scan = next(
        step
        for step in steps
        if step["name"] == "Run Trivy (table output for PR comments)"
    )

    assert 'severity: "CRITICAL,HIGH"' in workflow
    assert "SARIF upload above still reports unfixed OS findings" in workflow
    assert sarif_scan["with"]["exit-code"] == "0"
    assert "ignore-unfixed" not in sarif_scan["with"]
    assert table_scan["with"]["exit-code"] == "1"
    assert str(table_scan["with"]["ignore-unfixed"]).lower() == "true"


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
