"""Tests for security workflow silent-failure guards."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script_module(name: str):
    script_path = Path(__file__).resolve().parents[3] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_detects_security_continue_on_error_and_masked_scanner(tmp_path):
    module = _load_script_module("check_workflows_no_silent_failures")
    workflow = tmp_path / "security.yml"
    workflow.write_text(
        """
name: security
jobs:
  scan:
    steps:
      - name: Security Audit
        continue-on-error: true
        run: |
          pip-audit --ignore-vuln CVE-2024-0001 || echo "[]" > report.json
""",
        encoding="utf-8",
    )

    violations = module.find_violations([workflow])

    assert [violation.pattern for violation in violations] == [
        "continue-on-error",
        "inline-pip-audit-waiver",
        "masked-scanner-failure",
    ]


def test_allows_explicit_waiver_helper_invocation(tmp_path):
    module = _load_script_module("check_workflows_no_silent_failures")
    workflow = tmp_path / "security.yml"
    workflow.write_text(
        """
name: security
jobs:
  scan:
    steps:
      - name: Security Audit
        run: |
          mapfile -t flags < <(python3 scripts/ci/check_pip_audit_waivers.py)
          pip-audit "${flags[@]}"
""",
        encoding="utf-8",
    )

    assert module.find_violations([workflow]) == []
