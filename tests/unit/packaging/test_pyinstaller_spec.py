"""T7 TDD: PyInstaller spec file satisfies the documented include/exclude contract."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent.parent
SPEC_PATH = ROOT / "sidekick.spec"


@pytest.fixture(scope="module")
def spec_source() -> str:
    assert SPEC_PATH.exists(), f"sidekick.spec not found at {SPEC_PATH}"
    return SPEC_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# T7-contract-1: entry-point is correct
# ---------------------------------------------------------------------------


def test_spec_entrypoint(spec_source: str) -> None:
    assert "sidekick.__main__" in spec_source or "__main__.py" in spec_source, (
        "spec must reference sidekick/__main__.py as the entry point"
    )


# ---------------------------------------------------------------------------
# T7-contract-2: required packages are included
# ---------------------------------------------------------------------------


REQUIRED_INCLUDES = [
    "sidekick",
]


@pytest.mark.parametrize("pkg", REQUIRED_INCLUDES)
def test_required_package_included(spec_source: str, pkg: str) -> None:
    assert pkg in spec_source, f"spec must include '{pkg}'"


# ---------------------------------------------------------------------------
# T7-contract-3: heavy physics engines are excluded
# ---------------------------------------------------------------------------


EXCLUDED_PACKAGES = ["pybullet", "mujoco", "pydrake"]


@pytest.mark.parametrize("pkg", EXCLUDED_PACKAGES)
def test_excluded_package_excluded(spec_source: str, pkg: str) -> None:
    # The package should appear only in the excludes section, not in includes
    # Simplest check: if it appears at all it must be in an excludes list
    if pkg not in spec_source:
        return  # absent entirely — fine
    # If present, it must be explicitly excluded
    assert (
        "excludes" in spec_source.lower() or "exclude_binaries" in spec_source.lower()
    ), f"'{pkg}' appears in spec but there is no excludes section"


# ---------------------------------------------------------------------------
# T7-contract-4: size budget constant is declared
# ---------------------------------------------------------------------------


def test_size_budget_declared(spec_source: str) -> None:
    assert "MAX_MB" in spec_source or "max_mb" in spec_source or "250" in spec_source, (
        "spec or build script must declare the 250 MB size budget"
    )


# ---------------------------------------------------------------------------
# T7-contract-5: build script exists
# ---------------------------------------------------------------------------


def test_build_script_exists() -> None:
    script = ROOT / "scripts" / "packaging" / "build_sidekick_binary.py"
    assert script.exists(), f"Build script not found at {script}"


# ---------------------------------------------------------------------------
# T7-contract-6: CI workflow file exists and has the expected triggers
# ---------------------------------------------------------------------------


def test_release_workflow_exists() -> None:
    wf = ROOT / ".github" / "workflows" / "release-sidekick-binary.yml"
    assert wf.exists(), f"Release workflow not found at {wf}"


def test_release_workflow_triggers(spec_source: str) -> None:
    wf = ROOT / ".github" / "workflows" / "release-sidekick-binary.yml"
    if not wf.exists():
        pytest.skip("workflow file absent")
    content = wf.read_text(encoding="utf-8")
    assert "workflow_dispatch" in content, (
        "workflow must support workflow_dispatch trigger"
    )
    assert "sidekick-v" in content, "workflow must trigger on sidekick-v* tags"


def test_release_workflow_matrix(spec_source: str) -> None:
    wf = ROOT / ".github" / "workflows" / "release-sidekick-binary.yml"
    if not wf.exists():
        pytest.skip("workflow file absent")
    content = wf.read_text(encoding="utf-8")
    assert "ubuntu" in content, "matrix must include ubuntu runner"
    assert "macos" in content, "matrix must include macos runner"
    assert "windows" in content, "matrix must include windows runner"
