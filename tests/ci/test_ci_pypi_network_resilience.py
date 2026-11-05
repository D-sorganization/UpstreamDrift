"""CI pip installs and pip-audit scans must ride out transient PyPI stalls (#14595).

pip defaults to 5 retries with a 15 s socket timeout, and pip-audit defaults to a
15 s read timeout with no retry. Main CI Standard failed on exactly that
(``Read timed out. (read timeout=15)``) and passed on the next run unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
CI_STANDARD = WORKFLOWS / "ci-standard.yml"
PIP_DEFAULTS = {"PIP_RETRIES": 5, "PIP_DEFAULT_TIMEOUT": 15}
PIP_AUDIT_DEFAULT_TIMEOUT = 15
_PIP_AUDIT_CALL = re.compile(r"-m\s+pip_audit\b")
_TIMEOUT_FLAG = re.compile(r"--timeout[ =](\d+)")


def _workflow_env() -> dict[str, object]:
    workflow = yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8"))
    return workflow.get("env") or {}


@pytest.mark.parametrize("name", sorted(PIP_DEFAULTS))
def test_ci_standard_raises_pip_network_defaults(name: str) -> None:
    value = _workflow_env().get(name)
    assert value is not None, f"ci-standard.yml must set workflow-level {name}"
    assert int(str(value)) > PIP_DEFAULTS[name], (
        f"{name}={value} does not exceed pip's default {PIP_DEFAULTS[name]}"
    )


def test_no_job_overrides_pip_network_settings() -> None:
    workflow = yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8"))
    overrides = [
        f"{job_id}:{name}"
        for job_id, job in (workflow.get("jobs") or {}).items()
        for name in PIP_DEFAULTS
        if name in ((job or {}).get("env") or {})
    ]
    assert overrides == []


def _pip_audit_calls() -> list[tuple[str, str]]:
    return [
        (path.name, line.strip())
        for path in sorted(WORKFLOWS.glob("*.y*ml"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if _PIP_AUDIT_CALL.search(line)
    ]


def test_pip_audit_calls_exist() -> None:
    assert _pip_audit_calls(), "expected pip-audit scans in the workflows"


def test_every_pip_audit_call_sets_a_longer_timeout() -> None:
    short = [
        f"{name}: {line}"
        for name, line in _pip_audit_calls()
        if not (match := _TIMEOUT_FLAG.search(line))
        or int(match.group(1)) <= PIP_AUDIT_DEFAULT_TIMEOUT
    ]
    assert short == [], "pip-audit calls without --timeout > 15s:\n" + "\n".join(short)
