"""Tests for engine Docker cache contract compliance per CACHE_CONTRACT.md (#13579).

Verifies that all workflow files in .github/workflows that build Docker images
comply with the fleet cache contract:
1. Attach the shared host BuildKit builder (ud-buildkit) via shared_buildkit.py ensure.
2. Build on ${{ steps.buildkit.outputs.builder || steps.buildx-fallback.outputs.name }}.
3. Provide a fallback to setup-buildx-action with continue-on-error so cache infrastructure
   never blocks or fails PRs.
4. Do NOT import or export per-runner type=local or type=gha caches (layer cache is
   managed by BuildKit's internal store on the host).
5. GC settings in scripts/ci/buildkitd.ud.toml match the limits documented in CACHE_CONTRACT.md.
"""

from __future__ import annotations

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
BUILDKITD_CONFIG = REPO_ROOT / "scripts" / "ci" / "buildkitd.ud.toml"

pytestmark = pytest.mark.unit

_SHARED_BUILDER = (
    "${{ steps.buildkit.outputs.builder || steps.buildx-fallback.outputs.name }}"
)


def _get_docker_build_workflows() -> list[Path]:
    """Find all workflows that build Docker images via build-push-action or docker build."""
    docker_workflows: list[Path] = []
    for wf in sorted(WORKFLOWS_DIR.glob("*.yml")):
        content = wf.read_text(encoding="utf-8")
        if "uses: docker/build-push-action@" in content or "docker build " in content:
            docker_workflows.append(wf)
    return docker_workflows


def test_no_workflow_uses_raw_docker_build() -> None:
    """Every Docker build workflow must use build-push-action with BuildKit, never raw docker build."""
    for wf in sorted(WORKFLOWS_DIR.glob("*.yml")):
        content = wf.read_text(encoding="utf-8")
        # Ignore comments or echo statements
        for idx, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#") or "echo " in stripped:
                continue
            assert not stripped.startswith("docker build "), (
                f"{wf.name}:{idx} uses raw 'docker build'; must use docker/build-push-action with shared_buildkit"
            )


def test_docker_build_workflows_use_shared_buildkit_contract() -> None:
    """All workflows building Docker images must adhere to the shared BuildKit contract."""
    workflows = _get_docker_build_workflows()
    assert len(workflows) >= 3, (
        f"Expected at least 3 docker build workflows, found {len(workflows)}"
    )

    for wf in workflows:
        content = wf.read_text(encoding="utf-8")
        build_count = content.count("uses: docker/build-push-action@")
        assert build_count > 0, (
            f"{wf.name} should have at least one build-push-action step"
        )

        # 1. Must ensure shared builder once per build step
        assert (
            content.count("python3 scripts/ci/shared_buildkit.py ensure") == build_count
        ), f"{wf.name} must run shared_buildkit.py ensure for each build"

        # 2. Must use continue-on-error for the ensure step
        assert "continue-on-error: true" in content, (
            f"{wf.name} must set continue-on-error on shared_buildkit.py step"
        )

        # 3. Must use the canonical builder output with fallback
        assert content.count(f"builder: {_SHARED_BUILDER}") == build_count, (
            f"{wf.name} must configure builder: {_SHARED_BUILDER}"
        )

        # 4. Must provide fallback setup-buildx-action
        assert content.count("uses: docker/setup-buildx-action@") == build_count, (
            f"{wf.name} must have fallback setup-buildx-action step"
        )
        assert (
            content.count("if: steps.buildkit.outputs.builder == ''") == build_count
        ), f"{wf.name} must check steps.buildkit.outputs.builder == '' for fallback"


def test_no_workflow_uses_per_runner_type_local_or_gha_cache() -> None:
    """Per CACHE_CONTRACT.md, layer cache is the host builder's volume; no type=local or type=gha."""
    for wf in _get_docker_build_workflows():
        content = wf.read_text(encoding="utf-8")
        assert "type=gha" not in content, f"{wf.name} must not use type=gha cache"
        assert "type=local" not in content, f"{wf.name} must not use type=local cache"
        assert "cache-from:" not in content, f"{wf.name} must not specify cache-from"
        assert "cache-to:" not in content, f"{wf.name} must not specify cache-to"


def test_docker_workflows_do_not_use_per_runner_tool_cache_paths() -> None:
    """#13208: duplicated per-runner buildx local caches must not return."""
    for wf in _get_docker_build_workflows():
        content = wf.read_text(encoding="utf-8")
        assert "runner.tool_cache" not in content, wf.name
        assert "buildx-cache/upstream-drift" not in content, wf.name


def test_buildkit_gc_policy_matches_fleet_cache_contract() -> None:
    """scripts/ci/buildkitd.ud.toml must match the GC numbers documented in CACHE_CONTRACT.md."""
    content = BUILDKITD_CONFIG.read_text(encoding="utf-8")
    assert "gc = true" in content
    assert 'keepDuration = "48h"' in content
    assert 'maxUsedSpace = "5GB"' in content
    assert 'reservedSpace = "5GB"' in content
    assert 'maxUsedSpace = "30GB"' in content
    assert 'minFreeSpace = "20%"' in content


def _get_all_workflows() -> list[Path]:
    """Return all GitHub workflow files in .github/workflows."""
    return sorted(WORKFLOWS_DIR.glob("*.yml"))


def _validate_conda_steps_in_content(content: str, source_name: str) -> None:
    """Validate that any workflow step creating a conda env uses CONDA_PKGS_DIRS on host-local path."""
    lines = content.splitlines()
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if any(
            cmd in stripped
            for cmd in (
                "conda env create",
                "conda create",
                "mamba env create",
                "mamba create",
                "micromamba create",
            )
        ):
            assert "CONDA_PKGS_DIRS" in content, (
                f"{source_name}:{idx} creates a conda environment without CONDA_PKGS_DIRS host-local cache path"
            )


@pytest.mark.parametrize("wf", _get_all_workflows(), ids=lambda p: p.name)
def test_conda_env_steps_use_host_local_pkgs_dir(wf: Path) -> None:
    """Parametrised across all workflows: any conda step must set CONDA_PKGS_DIRS (#13579)."""
    content = wf.read_text(encoding="utf-8")
    _validate_conda_steps_in_content(content, wf.name)


def test_conda_env_guard_fails_when_pkgs_dir_missing() -> None:
    """Guards against uncontracted conda steps: fixture workflow without CONDA_PKGS_DIRS fails (#13579)."""
    fixture_content = (
        "name: Fixture Conda Step\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: self-hosted\n"
        "    steps:\n"
        "      - name: Create Conda Env\n"
        "        run: conda env create -f environment.yml\n"
    )
    with pytest.raises(AssertionError, match="without CONDA_PKGS_DIRS"):
        _validate_conda_steps_in_content(fixture_content, "fixture.yml")
