"""Public UD must not call internal Repository_Management reusable workflows (#14206).

GitHub refuses a public repository's call to an internal repository's
reusable workflow before any job is created, so such a workflow fails every
run with 0 jobs. Conflict Disarm did exactly that. Read RM scripts through an
App-token checkout instead, as collate-changes.yml does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
CONFLICT_DISARM = WORKFLOWS / "conflict-disarm.yml"
INTERNAL_PREFIX = "D-sorganization/Repository_Management/.github/workflows/"

pytestmark = pytest.mark.unit


def _jobs(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")).get("jobs") or {}


@pytest.mark.parametrize("path", sorted(WORKFLOWS.glob("*.y*ml")), ids=lambda p: p.name)
def test_no_job_calls_an_internal_reusable_workflow(path: Path) -> None:
    for name, job in _jobs(path).items():
        assert not str(job.get("uses", "")).startswith(INTERNAL_PREFIX), name


def test_conflict_disarm_runs_the_rm_script_inline() -> None:
    (job,) = _jobs(CONFLICT_DISARM).values()
    steps = job["steps"]
    checkout = next(
        step
        for step in steps
        if step.get("with", {}).get("repository")
        == "D-sorganization/Repository_Management"
    )
    sweep = next(
        step for step in steps if "scripts.conflict_disarm" in step.get("run", "")
    )

    assert checkout["with"]["token"] == "${{ steps.app-token.outputs.token }}"
    assert checkout["with"]["path"] == ".rm-tools"
    assert sweep["env"]["DRY_RUN_INPUT"] == "${{ inputs.dry_run || 'true' }}"
    assert sweep["env"]["EXCLUDE_PRS"] == "${{ inputs.exclude_prs || '12634' }}"


def test_conflict_disarm_app_token_is_scoped_to_ud_and_rm() -> None:
    """#14214: the token reaches only the repos and permissions the sweep uses."""
    (job,) = _jobs(CONFLICT_DISARM).values()
    token = next(step for step in job["steps"] if step.get("id") == "app-token")
    inputs = token["with"]

    repositories = {
        name.strip()
        for name in str(inputs.get("repositories", "")).replace(",", "\n").split()
    }
    permissions = {
        key: value for key, value in inputs.items() if key.startswith("permission-")
    }

    assert repositories == {"UpstreamDrift", "Repository_Management"}
    assert permissions == {
        "permission-contents": "write",
        "permission-pull-requests": "write",
        "permission-issues": "write",
    }
