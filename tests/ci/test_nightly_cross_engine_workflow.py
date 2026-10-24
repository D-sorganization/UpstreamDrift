"""The nightly engine runtime must support the reviewed Drake builds (#13619)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/nightly-cross-engine.yml"


def _jobs() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def test_nightly_runtime_supports_reviewed_drake_versions() -> None:
    job = _jobs()["cross-engine-validation"]
    # Both reviewed Drake releases require Python >=3.12; the project supports
    # 3.11 and 3.12, so this combined-engine lane must use the latter.
    assert job["strategy"]["matrix"]["python-version"] == ["3.12"]
    setup = next(
        step for step in job["steps"] if "actions/setup-python@" in step.get("uses", "")
    )
    assert setup["with"]["python-version"] == "${{ matrix.python-version }}"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["optional-dependencies"]["drake"] == [
        "drake>=1.57.0,<1.59.0"
    ]


def test_public_repo_routes_nightly_validation_to_hosted_runner() -> None:
    step = next(
        step for step in _jobs()["pick-runner"]["steps"] if step.get("id") == "check"
    )
    assert "ubuntu-latest" in step["run"]
    assert "CI_RUNNER_MODE" in step["env"]


def test_runtime_alignment_preserves_engine_coverage_and_failure_reporting() -> None:
    jobs = _jobs()
    job = jobs["cross-engine-validation"]
    assert job["needs"] == "pick-runner"
    install = next(step for step in job["steps"] if step.get("id") == "install_deps")
    assert 'pip install -e ".[dev,drake,pinocchio,opensim,myosuite]"' in install["run"]
    assert 'pip install "mujoco>=3.3.0,<4.0.0"' in install["run"]
    spine = next(
        step for step in job["steps"] if "tests/opensim/spine" in step.get("run", "")
    )
    assert spine["if"] == "steps.install_deps.outcome == 'success'"
    assert spine["env"]["UD_REQUIRE_OPENSIM"] == "1"
    dashboard = jobs["publish-dashboard"]
    assert dashboard["needs"] == ["pick-runner", "cross-engine-validation"]
    assert "always()" in dashboard["if"]
