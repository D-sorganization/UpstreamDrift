"""Docker profile smoke removes the images it builds (#13761, #14164)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE = REPO_ROOT / ".github" / "workflows" / "docker-smoke.yml"
BUILD_JOBS = ("profile-smoke", "ud-sim-headless-smoke")
CI_LABEL = "org.upstreamdrift.ci=docker-smoke"

pytestmark = pytest.mark.unit


def _job(name: str) -> dict[str, Any]:
    return yaml.safe_load(SMOKE.read_text(encoding="utf-8"))["jobs"][name]


def _build_steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("docker/build-push-action@")
    ]


@pytest.mark.parametrize("job_name", BUILD_JOBS)
def test_each_build_is_tagged_per_run(job_name: str) -> None:
    job = _job(job_name)
    image = job["env"]["IMAGE"]

    build_steps = _build_steps(job)
    assert build_steps, f"{job_name} must build an image"
    assert "${{ github.run_id }}" in image
    assert "${{ github.run_attempt }}" in image
    for step in build_steps:
        assert step["with"]["tags"] == "${{ env.IMAGE }}"


@pytest.mark.parametrize("job_name", BUILD_JOBS)
def test_each_build_carries_the_cleanup_label(job_name: str) -> None:
    for step in _build_steps(_job(job_name)):
        assert CI_LABEL in step["with"]["labels"]


@pytest.mark.parametrize("job_name", BUILD_JOBS)
def test_each_job_always_removes_only_its_own_images(job_name: str) -> None:
    cleanup_steps = [
        step
        for step in _job(job_name)["steps"]
        if step.get("name", "").startswith("Remove this run's image")
    ]
    assert len(cleanup_steps) == 1
    last = cleanup_steps[0]
    script = "\n".join(
        line for line in last["run"].splitlines() if not line.lstrip().startswith("#")
    )

    assert last["if"] == "always()"
    assert 'docker image rm "$IMAGE"' in script
    assert f"docker image prune --force --filter label={CI_LABEL}" in script


def test_no_step_uses_a_shared_smoke_tag() -> None:
    workflow = SMOKE.read_text(encoding="utf-8")

    assert "tags: upstream-drift-smoke:${{ matrix.profile }}" not in workflow
    assert "ud-sim:latest" not in workflow
