"""The docker size gates remove the images they build (#14164).

Runners on one host share a Docker daemon. Tagging every build
``upstream-drift:<profile>`` left each previous image dangling (~81 GB refilled
the ControlTower-nvme disk within an hour) and let concurrent runs overwrite
each other's tag. Each build job must tag its image per run, label it, and
always clean up only what this workflow built.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SIZE_GATES = REPO_ROOT / ".github" / "workflows" / "docker-size-gates.yml"
BUILD_JOBS = ("docker-smoke-and-size", "profile-size-matrix")
CI_LABEL = "org.upstreamdrift.ci=docker-size-gates"

pytestmark = pytest.mark.unit


def _job(name: str) -> dict[str, Any]:
    return yaml.safe_load(SIZE_GATES.read_text(encoding="utf-8"))["jobs"][name]


def _build_step(job: dict[str, Any]) -> dict[str, Any]:
    (step,) = [
        step
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("docker/build-push-action@")
    ]
    return step


@pytest.mark.parametrize("job_name", BUILD_JOBS)
def test_each_build_is_tagged_per_run(job_name: str) -> None:
    job = _job(job_name)
    image = job["env"]["IMAGE"]

    assert "${{ github.run_id }}" in image
    assert "${{ github.run_attempt }}" in image
    assert _build_step(job)["with"]["tags"] == "${{ env.IMAGE }}"


@pytest.mark.parametrize("job_name", BUILD_JOBS)
def test_each_build_carries_the_cleanup_label(job_name: str) -> None:
    assert CI_LABEL in _build_step(_job(job_name))["with"]["labels"]


@pytest.mark.parametrize("job_name", BUILD_JOBS)
def test_each_job_always_removes_only_its_own_images(job_name: str) -> None:
    last = _job(job_name)["steps"][-1]
    script = "\n".join(
        line for line in last["run"].splitlines() if not line.lstrip().startswith("#")
    )

    assert last["if"] == "always()"
    assert 'docker image rm "$IMAGE"' in script
    assert f"docker image prune --force --filter label={CI_LABEL}" in script
    for forbidden in ("volume", "system prune", "--all", " -a "):
        assert forbidden not in script


def test_no_step_measures_or_runs_a_shared_tag() -> None:
    workflow = SIZE_GATES.read_text(encoding="utf-8")

    assert "upstream-drift:runtime" not in workflow
    assert "upstream-drift:${{ matrix.profile }}" not in workflow
