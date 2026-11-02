"""Runtime dependencies required by the heavy force-overlay parity lane."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/cross-engine-equivalence.yml"


def test_pinocchio_force_overlay_installs_openmp_runtime() -> None:
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    parity_job = jobs["force-overlay-parity"]
    steps = parity_job["steps"]

    runtime_setup = next(
        step
        for step in steps
        if step.get("name") == "Install Pinocchio runtime libraries"
    )
    assert runtime_setup["if"] == "matrix.engine == 'pinocchio'"
    assert "libgomp1" in runtime_setup["run"]


def test_jaxsim_parity_guard_requires_each_named_case_exactly() -> None:
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    steps = jobs["equivalence"]["steps"]
    guard = next(
        s["run"] for s in steps if "parity cases actually ran" in s.get("name", "")
    )
    assert guard.count("--required-case") == 3
    assert "--required-classname tests.cross_engine.test_jaxsim_vs_pinocchio" in guard
    assert "--receipt-output" in guard and "--source-revision" in guard
    prerequisites = next(
        s["run"] for s in steps if "prerequisites importable" in s.get("name", "")
    )
    assert "import jax, jaxlib, jaxsim" in prerequisites
    assert "check_pinocchio_dynamics_api.py" in prerequisites
    assert "RUNNER_OS" in prerequisites
    assert "|| true" not in prerequisites
    run_tests = next(
        s["run"]
        for s in steps
        if s.get("name") == "Run cross-engine forward-sim equivalence test"
    )
    assert "tests/unit/cross_engine" not in run_tests
    assert (
        "pytestmark = pytest.mark.unit"
        in (ROOT / "tests/unit/cross_engine/test_engine_output_contract.py").read_text()
    )
    artifact = next(
        s["with"]["path"] for s in steps if s.get("name") == "Upload equivalence report"
    )
    assert (
        artifact.splitlines().count(
            "test-results/cross-engine-equivalence-receipt.json"
        )
        == 1
    )
