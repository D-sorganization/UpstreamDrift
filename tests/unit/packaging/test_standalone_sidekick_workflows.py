"""Workflow contract tests for standalone Sidekick packaging and release."""

from __future__ import annotations

from pathlib import Path

import yaml
import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_WORKFLOW = ROOT / ".github" / "workflows" / "package-standalone-sidekick.yml"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release-sidekick-binary.yml"


def _load_workflow(path: Path) -> dict:
    """Return a parsed workflow document."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _checkout_steps(workflow: dict) -> list[dict]:
    """Return every actions/checkout step in *workflow*."""
    return [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if str(step.get("uses", "")).startswith("actions/checkout@")
    ]


def test_package_workflow_builds_distribution_with_ui_bundle_enabled() -> None:
    """Wheel builds must use the checkout that owns the pinned Tools tree."""
    content = PACKAGE_WORKFLOW.read_text(encoding="utf-8")

    assert "python3 -m build --sdist --outdir dist/" in content
    assert "python3 -m build --wheel --outdir dist/" in content
    assert "python3 -m build --outdir dist/" not in content
    assert "SKIP_UI_BUILD" not in content


def test_package_workflow_checks_out_exact_recursive_tools_tree() -> None:
    """Every source checkout must materialize the pinned Tools submodule."""
    workflow = _load_workflow(PACKAGE_WORKFLOW)
    checkout_steps = _checkout_steps(workflow)

    assert len(checkout_steps) == 2
    for checkout in checkout_steps:
        assert checkout["with"]["submodules"] == "recursive"
        assert checkout["with"]["persist-credentials"] is False


def test_package_workflow_tracks_canonical_and_extension_sources() -> None:
    """Packaging runs when the Tools pin or classified UD extensions change."""
    content = PACKAGE_WORKFLOW.read_text(encoding="utf-8")

    assert '"vendor/ud-tools"' in content
    assert '"src/shared/python/sidekick/**"' in content
    assert '"src/__init__.py"' in content
    assert '"src/launchers/sidekick_extension_overlay.py"' in content
    assert '"launch_upstream_drift.py"' in content
    assert '"build_hooks.py"' in content
    assert '"scripts/config/shared_python_ownership_exceptions.yaml"' in content
    assert '"src/shared/python/sidekick/standalone/**"' not in content
    assert '"src/shared/python/sidekick/__main__.py"' not in content


def test_package_workflow_selects_one_exact_wheel() -> None:
    """Wheel consumers must use the single artifact selected by the build."""
    workflow = _load_workflow(PACKAGE_WORKFLOW)
    build_job = workflow["jobs"]["build-wheel"]
    build_step = next(
        step for step in build_job["steps"] if step.get("name") == "Build sdist + wheel"
    )
    smoke_job = workflow["jobs"]["smoke-test-wheel"]
    install_step = next(
        step
        for step in smoke_job["steps"]
        if step.get("name") == "Install wheel in clean venv"
    )

    assert build_job["outputs"]["wheel_filename"] == (
        "${{ steps.build-dist.outputs.wheel_filename }}"
    )
    assert build_step["id"] == "build-dist"
    assert "rm -rf dist" in build_step["run"]
    assert "mapfile -t wheels" in build_step["run"]
    assert 'test "${#wheels[@]}" -eq 1' in build_step["run"]
    assert "wheel_filename=$(basename" in build_step["run"]
    assert install_step["env"]["UPSTREAM_DRIFT_WHEEL"] == (
        "dist/${{ needs.build-wheel.outputs.wheel_filename }}"
    )
    assert 'pip install "$UPSTREAM_DRIFT_WHEEL"' in install_step["run"]
    assert "dist/*.whl" not in PACKAGE_WORKFLOW.read_text(encoding="utf-8")


def test_package_workflow_preserves_time_for_verified_artifact_upload() -> None:
    """Cold provider builds must not time out after verification but before upload."""
    workflow = _load_workflow(PACKAGE_WORKFLOW)

    assert workflow["jobs"]["build-wheel"]["timeout-minutes"] >= 30


def test_release_workflow_has_queue_protection_and_timeouts() -> None:
    """Release workflow jobs must follow the repo queue-saturation guardrails."""
    content = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "concurrency:" in content
    assert "cancel-in-progress: true" in content
    assert "attach-to-release:" in content
    attach_section = content.split("attach-to-release:", maxsplit=1)[1]
    assert "timeout-minutes:" in attach_section


def test_release_workflow_checks_out_exact_recursive_tools_tree() -> None:
    """Binary builds must use the checked-out gitlink rather than an empty vendor dir."""
    workflow = _load_workflow(RELEASE_WORKFLOW)
    checkout_steps = _checkout_steps(workflow)

    assert len(checkout_steps) == 1
    checkout = checkout_steps[0]
    assert checkout["with"]["submodules"] == "recursive"
    assert checkout["with"]["persist-credentials"] is False


def test_release_workflow_install_is_fail_closed() -> None:
    """A broken project/dependency install must stop the binary release."""
    workflow = _load_workflow(RELEASE_WORKFLOW)
    install_step = next(
        step
        for step in workflow["jobs"]["build-linux"]["steps"]
        if step.get("name") == "Install dependencies"
    )
    command = install_step["run"]

    assert 'python3 -m pip install ".[gui-tools]" pyinstaller' in command
    assert "|| true" not in command
    assert "--no-deps" not in command


def test_release_workflow_only_claims_its_native_linux_artifact() -> None:
    """One Linux runner must not relabel the same binary as macOS and Windows."""
    workflow = _load_workflow(RELEASE_WORKFLOW)
    build_job = workflow["jobs"]["build-linux"]
    upload_step = next(
        step
        for step in build_job["steps"]
        if step.get("name") == "Upload binary artifact"
    )
    build_step = next(
        step for step in build_job["steps"] if step.get("name") == "Build binary"
    )

    assert "matrix" not in build_job
    assert build_job["name"] == "Build binary (Linux)"
    assert build_job["runs-on"] == "d-sorg-fleet-docker"
    assert "--expected-platform linux" in build_step["run"]
    assert upload_step["with"]["name"] == "sidekick-linux"
    assert upload_step["with"]["path"] == "dist/sidekick"
    assert workflow["jobs"]["attach-to-release"]["needs"] == ["build-linux"]
