"""Tests for the hosted-runner routing guard (issue #7127).

The ``CI Standard / Reject hosted runner routing`` required status check runs
``scripts/check_local_only_workflows.py``. These tests lock in that the guard
actually inspects ``runs-on`` routing (not a no-op), allows the documented
canaries, and that the CI Standard workflow invokes the real script.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check_local_only_workflows.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_local_only_guard", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_module()


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_flags_hosted_ubuntu_latest(tmp_path: Path) -> None:
    wf = _write(
        tmp_path,
        "leaky.yml",
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n    steps: []\n",
    )
    violations = guard.audit_workflow(wf)
    assert violations, "ubuntu-latest must be flagged"
    assert "ubuntu-latest" in violations[0]


def test_flags_windows_and_macos(tmp_path: Path) -> None:
    for label in ("windows-latest", "macos-13", "ubuntu-22.04"):
        wf = _write(
            tmp_path,
            f"wf_{label}.yml",
            f"jobs:\n  j:\n    runs-on: {label}\n    steps: []\n",
        )
        assert guard.audit_workflow(wf), f"{label} must be flagged"


def test_allows_fleet_and_self_hosted(tmp_path: Path) -> None:
    body = (
        "jobs:\n"
        "  a:\n    runs-on: d-sorg-fleet-docker\n    steps: []\n"
        "  b:\n    runs-on: ${{ needs.pick-runner.outputs.runner }}\n    steps: []\n"
        "  c:\n    runs-on: self-hosted\n    steps: []\n"
    )
    wf = _write(tmp_path, "fleet.yml", body)
    assert guard.audit_workflow(wf) == []


def test_allowlisted_job_name_not_flagged(tmp_path: Path) -> None:
    body = (
        "jobs:\n"
        "  guard:\n"
        "    name: Reject hosted runner routing\n"
        "    runs-on: ubuntu-latest\n"
        "    steps: []\n"
    )
    wf = _write(tmp_path, "ci.yml", body)
    assert guard.audit_workflow(wf) == []


def test_allowlisted_file_not_flagged(tmp_path: Path) -> None:
    body = "jobs:\n  j:\n    runs-on: ubuntu-latest\n    steps: []\n"
    wf = _write(tmp_path, "local-only-runner-guard.yml", body)
    assert guard.audit_workflow(wf) == []


def test_matrix_os_hosted_flagged(tmp_path: Path) -> None:
    body = (
        "jobs:\n"
        "  j:\n"
        "    runs-on: ${{ matrix.os }}\n"
        "    strategy:\n"
        "      matrix:\n"
        "        os: [ubuntu-latest, windows-latest]\n"
        "    steps: []\n"
    )
    wf = _write(tmp_path, "matrix.yml", body)
    assert len(guard.audit_workflow(wf)) == 2


class TestCiStandardInvokesRealGuard:
    """The CI Standard job body must not be a no-op (issue #7127)."""

    def _ci_text(self) -> str:
        path = _REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"
        return path.read_text(encoding="utf-8")

    def test_runs_the_real_script(self) -> None:
        text = self._ci_text()
        assert "scripts/check_local_only_workflows.py" in text

    def test_is_not_a_bypass_stub(self) -> None:
        text = self._ci_text()
        # The old stub literally ran `echo "Bypass"` as the entire guard body.
        assert 'echo "Bypass"' not in text

    def test_repo_workflows_pass_the_guard(self) -> None:
        """The repo's own workflows must satisfy the guard it now enforces."""
        wf_dir = _REPO_ROOT / ".github" / "workflows"
        failures: list[str] = []
        for path in sorted(wf_dir.glob("*.y*ml")):
            failures.extend(guard.audit_workflow(path))
        assert failures == [], "\n".join(failures)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
