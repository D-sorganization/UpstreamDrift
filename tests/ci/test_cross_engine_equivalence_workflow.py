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


def test_jaxsim_parity_guard_requires_exact_cases_and_engine_prerequisites() -> None:
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    steps = jobs["equivalence"]["steps"]

    def step(prefix: str):
        return next(s for s in steps if s.get("name", "").startswith(prefix))

    gate = step("Assert JaxSim parity cases actually ran")["run"]
    required = step("Assert JaxSim parity prerequisites importable")["run"]
    lane = step("Run cross-engine forward-sim equivalence test")["run"]
    artifact = step("Upload equivalence report")["with"]["path"]
    receipt = "test-results/cross-engine-equivalence-receipt.json"
    parity = (ROOT / "tests/cross_engine/test_jaxsim_vs_pinocchio.py").read_text()
    assert (
        gate.count("--required-case") == 3
        and "--required-classname tests.cross_engine.test_jaxsim_vs_pinocchio" in gate
        and receipt in gate
        and "$GITHUB_SHA" in gate
        and '[[ "$RUNNER_OS" == Linux ]]' in required
        and "import jax, jaxlib, jaxsim" in required
        and "check_pinocchio_dynamics_api.py" in required
        and "|| true" not in required
        and "tests/unit/cross_engine" not in lane
        and all(
            marker in parity
            for marker in (
                "pytest.importorskip",
                "requires_jaxsim",
                "requires_pinocchio",
            )
        )
        and "pytestmark = pytest.mark.unit"
        in (ROOT / "tests/unit/cross_engine/test_engine_output_contract.py").read_text()
    )
    assert artifact.splitlines().count(receipt) == 1
