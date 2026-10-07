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


ISOLATED_CARGO_HOME = "${{ github.workspace }}/.cargo-home"

# Jobs that deliberately deviate from the shared CARGO_HOME convention. Each is
# a known, separate gap (see the reason); an entry here must be removed as soon
# as the job conforms, which test_cargo_home_exemptions_are_still_needed enforces.
CARGO_HOME_EXEMPT_JOBS = {
    ("tauri-build.yml", "check"): "uses its own .cargo-tauri-check directory",
    (
        "tauri-build.yml",
        "build",
    ): "matrix job whose Windows leg installs rustup under %USERPROFILE%",
}


def _cargo_home(job: dict[str, Any]) -> object:
    return (job.get("env") or {}).get("CARGO_HOME")


@pytest.mark.unit
def test_every_rust_toolchain_job_uses_an_isolated_cargo_home() -> None:
    """Concurrent jobs sharing ~/.cargo can delete binaries from one another.

    Every Rust job must pin both RUSTUP_HOME and CARGO_HOME to the exact
    per-workspace paths (RM#2021).
    """
    jobs = _rust_toolchain_jobs()
    assert jobs, "expected at least one job that installs Rust"
    offenders = [
        f"{workflow}:{job_id}"
        for workflow, job_id, job in jobs
        if (workflow, job_id) not in CARGO_HOME_EXEMPT_JOBS
        and (
            _cargo_home(job) != ISOLATED_CARGO_HOME
            or (job.get("env") or {}).get("RUSTUP_HOME") != ISOLATED_RUSTUP_HOME
        )
    ]
    assert not offenders, (
        f"jobs without RUSTUP_HOME={ISOLATED_RUSTUP_HOME} and "
        f"CARGO_HOME={ISOLATED_CARGO_HOME}: {offenders}"
    )


@pytest.mark.unit
def test_cargo_home_exemptions_are_still_needed() -> None:
    by_key = {(w, j): job for w, j, job in _rust_toolchain_jobs()}
    stale = [
        f"{w}:{j}"
        for (w, j) in CARGO_HOME_EXEMPT_JOBS
        if (w, j) not in by_key or _cargo_home(by_key[(w, j)]) == ISOLATED_CARGO_HOME
    ]
    assert not stale, f"remove stale CARGO_HOME exemptions: {stale}"


@pytest.mark.unit
def test_no_rust_job_caches_the_shared_home_cargo() -> None:
    """Cache paths must follow CARGO_HOME, not the shared ~/.cargo."""
    offenders = [
        f"{workflow}:{job_id}"
        for workflow, job_id, job in _rust_toolchain_jobs()
        if (workflow, job_id) not in CARGO_HOME_EXEMPT_JOBS
        and "~/.cargo" in str(job.get("steps"))
    ]
    assert not offenders, f"jobs still referencing ~/.cargo: {offenders}"
