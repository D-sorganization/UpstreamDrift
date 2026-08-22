"""Contract tests for pinned Tools inputs in modular Docker builds."""

from __future__ import annotations

from pathlib import Path
import re

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(relative: str) -> dict:
    return yaml.safe_load((REPO_ROOT / relative).read_text(encoding="utf-8"))


def test_fetch_action_exports_content_bound_tools_provenance() -> None:
    action = _load_yaml(".github/actions/fetch-pinned-tools/action.yml")
    inputs = action["inputs"]
    outputs = action["outputs"]
    steps = action["runs"]["steps"]
    digest = next(
        step for step in steps if step.get("name") == "Digest pinned Tools sources"
    )

    assert outputs["gitlink-sha"]["value"] == "${{ steps.pin.outputs.sha }}"
    assert outputs["source-sha256"]["value"] == (
        "${{ steps.provenance.outputs.sha256 }}"
    )
    assert inputs["emit-source-provenance"]["default"] == "false"
    assert digest["id"] == "provenance"
    assert digest["if"] == "inputs.emit-source-provenance == 'true'"
    assert "pinned_tools_provenance.py" in digest["run"]
    assert "sha256=" in digest["run"]
    assert '>> "$GITHUB_OUTPUT"' in digest["run"]


@pytest.mark.parametrize(
    ("workflow_path", "job_name"),
    [
        (".github/workflows/docker-size-gates.yml", "profile-size-matrix"),
        (".github/workflows/docker-smoke.yml", "profile-smoke"),
    ],
)
def test_profile_matrix_fetches_and_passes_exact_tools_provenance(
    workflow_path: str,
    job_name: str,
) -> None:
    workflow = _load_yaml(workflow_path)
    steps = workflow["jobs"][job_name]["steps"]
    names = [step.get("name", "") for step in steps]
    fetch_index = names.index("Fetch pinned Tools packages")
    build_index = names.index("Build modular image (profile=${{ matrix.profile }})")
    fetch = steps[fetch_index]
    build_args = steps[build_index]["with"]["build-args"]

    assert fetch_index < build_index
    assert fetch["id"] == "pinned-tools"
    assert fetch["uses"] == "./.github/actions/fetch-pinned-tools"
    assert fetch["with"]["emit-source-provenance"] == "true"
    assert "TOOLS_GITLINK_SHA=${{ steps.pinned-tools.outputs.gitlink-sha }}" in (
        build_args
    )
    assert (
        "TOOLS_SOURCE_SHA256=${{ steps.pinned-tools.outputs.source-sha256 }}"
        in build_args
    )


def test_modular_dockerfile_copies_only_required_pinned_tools_roots() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile.modular").read_text(encoding="utf-8")
    required_copies = (
        "COPY vendor/ud-tools/src/shared/ ./vendor/ud-tools/src/shared/",
        "COPY vendor/ud-tools/src/sidekick/ ./vendor/ud-tools/src/sidekick/",
        "COPY vendor/ud-tools/src/chat/ ./vendor/ud-tools/src/chat/",
        "COPY vendor/ud-tools/src/python/src/utils/ ./vendor/ud-tools/src/python/src/utils/",
        "COPY vendor/ud-tools/src/contracts.py ./vendor/ud-tools/src/contracts.py",
        "COPY scripts/config/shared_python_ownership_exceptions.yaml ./scripts/config/shared_python_ownership_exceptions.yaml",
        "COPY scripts/packaging/pinned_tools_provenance.py ./scripts/packaging/pinned_tools_provenance.py",
    )

    assert "ARG TOOLS_GITLINK_SHA" in dockerfile
    assert "ARG TOOLS_SOURCE_SHA256" in dockerfile
    assert "UPSTREAMDRIFT_TOOLS_GITLINK_SHA=${TOOLS_GITLINK_SHA}" in dockerfile
    assert "UPSTREAMDRIFT_TOOLS_SOURCE_SHA256=${TOOLS_SOURCE_SHA256}" in dockerfile
    for required_copy in required_copies:
        assert required_copy in dockerfile
    assert "COPY vendor/ud-tools/ ./vendor/ud-tools/" not in dockerfile


def test_dockerignore_preserves_exact_pinned_tools_source_boundary() -> None:
    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")

    for relative in (
        "shared/**",
        "sidekick/**",
        "chat/**",
        "python/src/utils/**",
        "contracts.py",
    ):
        assert f"!vendor/ud-tools/src/{relative}" in dockerignore


def test_build_source_includes_provenance_helper_used_by_custom_hook() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"scripts/__init__.py"' in pyproject
    assert '"scripts/packaging/pinned_tools_provenance.py"' in pyproject


def test_runtime_dockerfiles_pin_pip_after_audited_security_floor() -> None:
    pins: set[str] = set()
    for relative in ("Dockerfile", "Dockerfile.modular"):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        pins.update(re.findall(r"\bpip==([0-9.]+)", source))

    assert len(pins) == 1
    version = tuple(int(part) for part in next(iter(pins)).split("."))
    assert version >= (26, 2)
