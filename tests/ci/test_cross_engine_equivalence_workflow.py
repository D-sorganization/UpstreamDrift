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
