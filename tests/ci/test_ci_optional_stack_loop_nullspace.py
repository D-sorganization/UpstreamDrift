"""Contract test for loop_nullspace pinocchio optional-stack CI (#14786).

Asserts that .github/workflows/ci-optional-stack.yml probes/installs the
Pinocchio ecosystem and includes tests/unit/loop_nullspace/test_snapshot_vs_spatial_fixture.py
in both its collection and execution steps for real Pinocchio test coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ci-optional-stack.yml"
TARGET_TEST = "tests/unit/loop_nullspace/test_snapshot_vs_spatial_fixture.py"


def _workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _job_steps(job_name: str = "optional-stack-check") -> list[dict[str, Any]]:
    workflow = _workflow()
    job = workflow["jobs"].get(job_name)
    assert isinstance(job, dict), f"missing job {job_name}"
    steps = job.get("steps")
    assert isinstance(steps, list), f"missing steps in {job_name}"
    return steps


def test_optional_stack_installs_real_pinocchio() -> None:
    """The optional-stack job must install real pin robotics library and probe pinocchio import."""
    steps = _job_steps()
    install_step = next(
        (step for step in steps if step.get("id") == "pinocchio_install"),
        None,
    )
    assert install_step is not None, "missing pinocchio_install step"
    run_text = install_step.get("run", "")
    assert "pip_retry install" in run_text
    assert "pin" in run_text.split()
    assert 'python3 -c "import pinocchio"' in run_text


def test_optional_stack_runs_loop_nullspace_fixture() -> None:
    """The optional-stack Pinocchio step must collect and run loop_nullspace spatial fixture test."""
    steps = _job_steps()
    step = next(
        (
            step
            for step in steps
            if step.get("name") == "Run Pinocchio Ecosystem Tests (Optional Stack)"
        ),
        None,
    )
    assert step is not None, "missing Run Pinocchio Ecosystem Tests step"
    run_text = step.get("run", "")

    assert "pytest --collect-only" in run_text
    assert "xvfb-run" in run_text
    collect_part, run_part = run_text.split("xvfb-run", 1)
    assert TARGET_TEST in collect_part, (
        f"{TARGET_TEST} missing from collect-only command"
    )
    assert TARGET_TEST in run_part, f"{TARGET_TEST} missing from test execution command"
