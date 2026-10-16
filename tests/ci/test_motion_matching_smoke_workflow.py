"""The slow motion-matching smokes run on a schedule (#12784).

``tests/integration/motion_matching`` is ``integration`` + ``slow``: the
default addopts deselect it and no PR lane runs it, so #12702 went unnoticed.
A dedicated scheduled workflow runs it, with a failure surfacing as an issue.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/motion-matching-smoke.yml"
SCRIPT = ROOT / "scripts/ci/run_motion_matching_smoke.sh"
PIN = re.compile(r"@[0-9a-f]{40}$")


def _doc() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _triggers() -> dict[str, Any]:
    doc = _doc()
    return doc.get("on") or doc[True]  # PyYAML parses a bare ``on`` as True


def test_workflow_is_scheduled_and_dispatchable() -> None:
    triggers = _triggers()
    assert triggers["schedule"][0]["cron"]
    assert "workflow_dispatch" in triggers
    assert "pull_request" not in triggers  # a 10-minute smoke never gates PRs


def test_runs_the_slow_smoke_via_the_shared_script() -> None:
    steps = _doc()["jobs"]["smoke"]["steps"]
    assert any(
        "scripts/ci/run_motion_matching_smoke.sh" in s.get("run", "") for s in steps
    )
    script = SCRIPT.read_text(encoding="utf-8")
    assert "tests/integration/motion_matching" in script
    assert '-m "slow' in script  # overrides the addopts "not slow" deselection
    assert "--timeout=0" in script  # IK alone outlasts the default per-test timeout


def test_job_is_bounded_and_fleet_routed() -> None:
    doc = _doc()
    job = doc["jobs"]["smoke"]
    assert job["needs"] == "pick-runner"
    assert "needs.pick-runner.outputs.runner" in job["runs-on"]
    assert 45 <= job["timeout-minutes"] <= 90
    run_step = next(
        s for s in job["steps"] if "run_motion_matching_smoke" in s.get("run", "")
    )
    assert 20 <= run_step["timeout-minutes"] <= 40  # --timeout=0 needs a hard stop
    assert doc["permissions"]["issues"] == "write"
    assert "concurrency" in doc


def test_all_actions_are_pinned_and_failure_reports_an_issue() -> None:
    doc = _doc()
    uses = [
        s["uses"]
        for job in doc["jobs"].values()
        for s in job.get("steps", ())
        if "uses" in s and not s["uses"].startswith("./")
    ]
    assert uses and all(PIN.search(u) for u in uses), uses
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "issues.create" in text
    assert "Motion-matching smoke FAILED" in text


@pytest.mark.parametrize(
    "doc",
    [
        ".github/WORKFLOWS.md",
        "docs/operations/workflow_inventory.md",
        "docs/development/ci_workflow_map.md",
    ],
)
def test_workflow_is_inventoried(doc: str) -> None:
    assert "motion-matching-smoke.yml" in (ROOT / doc).read_text(encoding="utf-8")
