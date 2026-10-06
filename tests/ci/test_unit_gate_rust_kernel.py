"""The unit-test gate must install the upstream-physics Rust kernel (#9411).

``ball_simulator`` enforces strict Rust parity and raises when the kernel is
missing, so a unit gate without the wheel fails every BunkerShot workbench/GUI
test on Linux. These checks pin the build-then-probe-then-pytest order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"
BUILD_STEP = "Build and install upstream-physics Rust kernel"


def _unit_gate_job() -> dict[str, Any]:
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job: dict[str, Any] = workflow["jobs"]["unit-test-gate"]
    return job


def _step_names(job: dict[str, Any]) -> list[str]:
    return [step.get("name", "") for step in job["steps"]]


@pytest.mark.unit
def test_unit_gate_builds_kernel_after_venv_and_before_pytest() -> None:
    names = _step_names(_unit_gate_job())
    assert BUILD_STEP in names, "unit-test-gate must build the upstream-physics wheel"
    build = names.index(BUILD_STEP)
    assert names.index("Install Rust toolchain") < build
    assert names.index("Install Unit Test Dependencies") < build
    assert build < names.index("Run Green-Suite Unit Gate")


@pytest.mark.unit
def test_unit_gate_kernel_step_installs_into_venv_and_probes_fail_closed() -> None:
    job = _unit_gate_job()
    run = job["steps"][_step_names(job).index(BUILD_STEP)]["run"]
    assert '"$RUNNER_TEMP/unit-gate-venv/bin/python"' in run
    assert "cd rust_core/upstream-physics" in run
    assert "--features python --release" in run
    assert "target/wheels/upstream_physics-*.whl" in run
    assert "scripts/ci/import_built_rust_wheels.py upstream_physics" in run
    assert "|| true" not in run
    assert "|| rm -f Cargo.lock" in run, "untracked Cargo.lock trips root clutter"


@pytest.mark.unit
def test_unit_gate_uses_workspace_cargo_home() -> None:
    env = _unit_gate_job().get("env", {})
    assert env.get("CARGO_HOME") == "${{ github.workspace }}/.cargo-home"


RUST_TOOLCHAIN_ACTION = "dtolnay/rust-toolchain@"
ISOLATED_RUSTUP_HOME = "${{ github.workspace }}/.rustup-home"


def _rust_toolchain_jobs() -> list[tuple[str, str, dict[str, Any]]]:
    yaml = pytest.importorskip("yaml")
    found: list[tuple[str, str, dict[str, Any]]] = []
    for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for job_id, job in (workflow.get("jobs") or {}).items():
            steps = job.get("steps") or []
            if any(
                str(s.get("uses", "")).startswith(RUST_TOOLCHAIN_ACTION) for s in steps
            ):
                found.append((path.name, job_id, job))
    return found


@pytest.mark.unit
def test_every_rust_toolchain_job_uses_an_isolated_rustup_home() -> None:
    """A shared ~/.rustup lets the image's preinstalled toolchain break installs.

    rustup upgrading the runner image's partially recorded ``stable`` in place
    aborts with ``detected conflict: 'bin/cargo'`` before any test runs
    (#11595). A per-job RUSTUP_HOME in the workspace starts empty because
    actions/checkout cleans untracked files; job-level env cannot use the
    runner context.
    """
    jobs = _rust_toolchain_jobs()
    assert jobs, "expected at least one job that installs Rust"
    offenders = [
        f"{workflow}:{job_id}"
        for workflow, job_id, job in jobs
        if (job.get("env") or {}).get("RUSTUP_HOME") != ISOLATED_RUSTUP_HOME
    ]
    assert not offenders, (
        f"jobs without RUSTUP_HOME={ISOLATED_RUSTUP_HOME}: {offenders}"
    )
