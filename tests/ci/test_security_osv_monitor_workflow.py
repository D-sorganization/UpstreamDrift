"""OSV Monitor must reference a runnable osv-scanner action (#14137)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github/workflows/security-osv-monitor.yml"
)
# The repository root action.yml is a metadata stub with no `runs:`; the action lives here.
RUNNABLE_PREFIX = "google/osv-scanner-action/osv-scanner-action@"


def _scanner_step() -> dict[str, Any]:
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["osv-monitor"]["steps"]
    return next(
        s
        for s in steps
        if str(s.get("uses", "")).startswith("google/osv-scanner-action")
    )


def test_scanner_step_uses_the_runnable_subpath_action() -> None:
    assert _scanner_step()["uses"].startswith(RUNNABLE_PREFIX)


def test_findings_do_not_fail_the_monitor() -> None:
    step = _scanner_step()
    assert step.get("continue-on-error") is True
    assert "fail-on-vuln" not in step.get("with", {})
