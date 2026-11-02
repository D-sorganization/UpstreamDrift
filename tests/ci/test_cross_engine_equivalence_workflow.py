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


def test_jaxsim_lane_wires_exact_cases_and_evidence_artifacts() -> None:
    steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["equivalence"][
        "steps"
    ]
    by_name = {item.get("name", ""): item for item in steps}
    guard = by_name["Assert JaxSim parity cases actually ran (issue #14290)"]["run"]
    prefix = "Assert JaxSim parity prerequisites importable"
    prerequisites = next(s["run"] for s in steps if s["name"].startswith(prefix))
    artifacts = by_name["Upload equivalence report"]["with"]["path"]
    cases = ("identity-angular-only", "identity-mixed-velocity", "rotated-world-origin")
    assert guard.count("--required-case") == 3
    assert all(f"[{case}]" in guard for case in cases)
    assert "--required-classname tests.cross_engine.test_jaxsim_vs_pinocchio" in guard
    assert (
        "--receipt-output test-results/cross-engine-equivalence-receipt.json" in guard
    )
    assert "$GITHUB_SHA" in guard and 'RUNNER_OS" == Linux' in prerequisites
    assert "test-results/cross-engine-equivalence-receipt.json" in artifacts
