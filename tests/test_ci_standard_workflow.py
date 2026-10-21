"""Fast/full workflow wiring acceptance for UD-CI-010b (#13577)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.ci.ci_tier_contract import FULL_TIER_JOBS, ROOT, run_aggregate, workflow

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not shutil.which("bash"), reason="Bash required"),
]


def test_event_classifier_drives_the_changed_paths_tier_output() -> None:
    changes = workflow()["jobs"]["changed-paths"]
    assert changes["outputs"]["tier"] == "${{ steps.tier.outputs.tier }}"
    step = next(step for step in changes["steps"] if step.get("id") == "tier")
    assert "scripts/ci/classify_event_tier.py" in step["run"]
    assert step["env"]["EVENT_NAME"] == "${{ github.event_name }}"


@pytest.mark.parametrize("job", sorted(FULL_TIER_JOBS))
def test_heavy_jobs_run_only_for_full_tier_code_changes(job: str) -> None:
    condition = workflow()["jobs"][job]["if"]
    assert "needs.changed-paths.outputs.tier == 'full'" in condition
    assert "needs.changed-paths.outputs.code == 'true'" in condition


@pytest.mark.parametrize("tier", ["fast", "full"])
@pytest.mark.parametrize("code,fork", [(True, False), (False, False), (True, True)])
def test_expected_skips_and_successes_pass_the_real_aggregate(
    tmp_path: Path, tier: str, code: bool, fork: bool
) -> None:
    result = run_aggregate(tmp_path, tier=tier, code=code, fork=fork)
    assert result.returncode == 0, result.stdout + result.stderr


def test_empty_pr_diff_does_not_silently_skip_code_gates(tmp_path: Path) -> None:
    detect = next(
        step
        for step in workflow()["jobs"]["changed-paths"]["steps"]
        if step.get("id") == "detect"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text("#!/bin/sh\nexit 0\n")
    fake_git.chmod(0o755)
    output = tmp_path / "output"
    env = os.environ | {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "EVENT_NAME": "pull_request",
        "BASE_REF": "main",
        "GITHUB_OUTPUT": str(output),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
    }
    result = subprocess.run(
        ["bash", "-e", "-c", detect["run"]],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "code=true" in output.read_text()


def test_existing_opensim_lane_classifier_remains_available() -> None:
    source = (ROOT / "scripts/classify_ci_changes.py").read_text()
    assert "def should_run_opensim_lane(" in source
