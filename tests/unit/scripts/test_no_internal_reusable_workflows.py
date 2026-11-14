"""Public UD must not call internal Repository_Management reusable workflows (#14206).

GitHub refuses a public repository's call to an internal repository's
reusable workflow before any job is created, so such a workflow fails every
run with 0 jobs. Conflict Disarm did exactly that; it now calls the public
fleet-workflows provider instead (fleet-workflows#1).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
CONFLICT_DISARM = WORKFLOWS / "conflict-disarm.yml"
PR_BODY_LINT = WORKFLOWS / "pr-body-lint.yml"
INTERNAL_PREFIX = "D-sorganization/Repository_Management/.github/workflows/"
PR_BODY_LINT_REUSABLE = (
    "D-sorganization/Repository_Management/.github/workflows/reusable-pr-body-lint.yml"
)
PR_BODY_LINT_PIN = "3b5e9f54febbd730bc2a650dc9193d26922ff49f"
FLEET_WORKFLOWS_CONFLICT_DISARM = (
    "D-sorganization/fleet-workflows/.github/workflows/reusable-conflict-disarm.yml"
)
# fleet-workflows commit that scopes the App token to the swept repo (#8).
FLEET_WORKFLOWS_PIN = "367b182c24ae3b01488fc1266130969e775023de"

pytestmark = pytest.mark.unit


def _jobs(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")).get("jobs") or {}


@pytest.mark.parametrize(
    "path",
    sorted(p for p in WORKFLOWS.glob("*.y*ml") if p != PR_BODY_LINT),
    ids=lambda p: p.name,
)
def test_no_job_calls_an_internal_reusable_workflow(path: Path) -> None:
    for name, job in _jobs(path).items():
        assert not str(job.get("uses", "")).startswith(INTERNAL_PREFIX), name


def _conflict_disarm_job() -> dict[str, Any]:
    (job,) = _jobs(CONFLICT_DISARM).values()
    return job


def test_conflict_disarm_calls_the_sha_pinned_public_provider() -> None:
    """fleet-workflows#1: one public implementation, pinned to a reviewed commit."""
    job = _conflict_disarm_job()
    workflow, _, ref = str(job.get("uses", "")).partition("@")

    assert workflow == FLEET_WORKFLOWS_CONFLICT_DISARM
    assert ref == FLEET_WORKFLOWS_PIN
    assert re.fullmatch(r"[0-9a-f]{40}", ref)
    assert "steps" not in job, "the inlined sweep must be gone"


def test_pr_body_lint_calls_the_sha_pinned_rm_reusable() -> None:
    """RM #2427: report-only caller; public repos may call this reviewed reusable."""
    (job,) = _jobs(PR_BODY_LINT).values()
    workflow, _, ref = str(job.get("uses", "")).partition("@")

    assert workflow == PR_BODY_LINT_REUSABLE
    assert ref == PR_BODY_LINT_PIN
    assert re.fullmatch(r"[0-9a-f]{40}", ref)
    assert set(job["with"]) == {"pr_number", "repo_owner", "repo_name"}
    assert "steps" not in job


def test_conflict_disarm_keeps_dry_run_and_the_12634_exclusion() -> None:
    job = _conflict_disarm_job()

    assert job["with"]["dry_run"] == "${{ inputs.dry_run || 'true' }}"
    assert job["with"]["exclude_prs"] == "${{ inputs.exclude_prs || '12634' }}"
    assert set(job["secrets"]) == {"JULES_APP_ID", "JULES_APP_PRIVATE_KEY"}
