"""Lighthouse mobile baseline runs in its own fleet job (#14698, follow-up of #14270).

``frontend-tests`` follows ``pick-runner``, which routes public-repo runs to a
hosted image, so the Lighthouse steps live in ``lighthouse-mobile`` pinned to
``d-sorg-fleet-docker`` behind the fork-PR guard (RM#1989). Scores stay
informational: the job never feeds ``quality-gate``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/ci-standard.yml"
JOB = "lighthouse-mobile"
FORK_GUARD = "github.event.pull_request.head.repo.full_name == github.repository"


def _jobs() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def _lighthouse_steps(job: dict[str, Any]) -> list[str]:
    return [
        step.get("name", "")
        for step in job.get("steps", [])
        if "lighthouse" in (step.get("name", "") + str(step.get("uses", ""))).lower()
    ]


def test_lighthouse_job_is_pinned_to_the_fleet_docker_runner() -> None:
    assert _jobs()[JOB]["runs-on"] == "d-sorg-fleet-docker"


def test_lighthouse_job_keeps_fork_prs_off_the_fleet_and_runs_on_prs_only() -> None:
    condition = " ".join(_jobs()[JOB]["if"].split())
    assert FORK_GUARD in condition
    assert "github.event_name == 'pull_request'" in condition


def test_every_lighthouse_step_moved_out_of_frontend_tests() -> None:
    jobs = _jobs()
    assert _lighthouse_steps(jobs["frontend-tests"]) == []
    assert len(_lighthouse_steps(jobs[JOB])) >= 5


def test_lighthouse_job_uploads_the_sanitized_receipt() -> None:
    uploads = [
        step["with"]["path"]
        for step in _jobs()[JOB]["steps"]
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]
    assert uploads == [".lighthouseci/baseline-receipt.json"]


def test_lighthouse_scores_stay_informational() -> None:
    jobs = _jobs()
    assert JOB not in jobs["quality-gate"]["needs"]
    assert all(JOB not in (job.get("needs") or []) for job in jobs.values())


def test_editing_this_workflow_runs_the_frontend_and_lighthouse_jobs() -> None:
    detect = next(
        step
        for step in _jobs()["frontend-tests"]["steps"]
        if step.get("id") == "frontend-changes"
    )
    assert "HEAD -- ui .github/workflows/ci-standard.yml" in detect["run"]
    outputs = _jobs()["frontend-tests"]["outputs"]
    assert (
        outputs["ui-changed"] == "${{ steps.frontend-changes.outputs.skip != 'true' }}"
    )
