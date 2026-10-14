"""Packaging and optional extras contract verification (UD #9406, #9417).

Enforces:
1. pyproject.toml defines per-engine and UI optional extras under [project.optional-dependencies].
2. requirements-tools.txt and Cargo.toml (tools-core rev) both pin D-sorganization/Tools to
   the exact same 40-character git SHA.
3. Makefile `install` target references requirements-tools.txt and launch_upstream_drift.py
   references requirements-tools.txt.
4. pyproject.toml `rust` optional extra covers all Maturin extension packages under rust_core.
5. deploy/ directory is populated with README.md and .gitkeep is removed.
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")


def test_pyproject_defines_required_optional_extras() -> None:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    assert pyproject_path.exists(), f"Missing pyproject.toml at {pyproject_path}"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    optional_deps = data.get("project", {}).get("optional-dependencies", {})

    expected_extras = ["mujoco", "drake", "pinocchio", "opensim", "myosuite", "ui"]
    for extra in expected_extras:
        assert extra in optional_deps, (
            f"Expected extra '{extra}' not defined in [project.optional-dependencies] of pyproject.toml"
        )
        assert len(optional_deps[extra]) > 0, f"Extra '{extra}' list must not be empty"

    # Specific package assertions
    assert any("mujoco" in req for req in optional_deps["mujoco"])
    assert any("drake" in req for req in optional_deps["drake"])
    assert any("pin" in req for req in optional_deps["pinocchio"])
    assert any("opensim" in req for req in optional_deps["opensim"])
    assert any("myosuite" in req for req in optional_deps["myosuite"])
    assert any("fastapi" in req for req in optional_deps["ui"])


def test_requirements_tools_and_cargo_pin_parity() -> None:
    req_path = REPO_ROOT / "requirements-tools.txt"
    cargo_path = REPO_ROOT / "Cargo.toml"

    assert req_path.exists(), f"Missing {req_path}"
    assert cargo_path.exists(), f"Missing {cargo_path}"

    # Extract SHA from requirements-tools.txt
    req_content = req_path.read_text(encoding="utf-8")
    req_match = re.search(
        r"git\+https://github\.com/D-sorganization/Tools\.git@([0-9a-fA-F]{40})",
        req_content,
    )
    assert req_match is not None, (
        f"Could not extract 40-character Tools git SHA from {req_path}"
    )
    req_sha = req_match.group(1).lower()
    assert HEX_40_RE.match(req_sha), (
        f"requirements-tools.txt SHA {req_sha} is not 40 hex chars"
    )

    # Extract SHA from Cargo.toml
    with open(cargo_path, "rb") as f:
        cargo_data = tomllib.load(f)

    tools_core = (
        cargo_data.get("workspace", {}).get("dependencies", {}).get("tools-core", {})
    )
    assert "rev" in tools_core, (
        f"tools-core in Cargo.toml missing 'rev' field: {tools_core}"
    )
    cargo_sha = tools_core["rev"].strip().lower()
    assert HEX_40_RE.match(cargo_sha), (
        f"Cargo.toml tools-core rev {cargo_sha} is not 40 hex chars"
    )

    assert req_sha == cargo_sha, (
        f"PIN PARITY MISMATCH: requirements-tools.txt pins Tools@{req_sha} "
        f"but Cargo.toml tools-core pins Tools@{cargo_sha}"
    )


def test_makefile_and_launcher_reference_requirements_tools() -> None:
    makefile_path = REPO_ROOT / "Makefile"
    launcher_path = REPO_ROOT / "launch_upstream_drift.py"

    assert makefile_path.exists(), f"Missing {makefile_path}"
    assert launcher_path.exists(), f"Missing {launcher_path}"

    makefile_content = makefile_path.read_text(encoding="utf-8")
    # Verify install target in Makefile references requirements-tools.txt
    install_target_match = re.search(
        r"install:.*?(?=\n[a-zA-Z0-9_\-\.]+:\s*|\Z)", makefile_content, re.DOTALL
    )
    assert install_target_match is not None, (
        "Could not find 'install:' target in Makefile"
    )
    install_block = install_target_match.group(0)
    assert "requirements-tools.txt" in install_block, (
        "Makefile 'install:' target must reference requirements-tools.txt"
    )

    launcher_content = launcher_path.read_text(encoding="utf-8")
    assert "requirements-tools.txt" in launcher_content, (
        "launch_upstream_drift.py must reference requirements-tools.txt"
    )


def test_rust_extra_covers_all_maturin_crates() -> None:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    assert pyproject_path.exists(), f"Missing pyproject.toml at {pyproject_path}"

    with open(pyproject_path, "rb") as f:
        root_data = tomllib.load(f)

    rust_extra = (
        root_data.get("project", {}).get("optional-dependencies", {}).get("rust", [])
    )
    assert len(rust_extra) > 0, "Optional extra 'rust' must not be empty"

    declared_rust_packages = {
        re.split(r"[><=~!]", req.strip())[0].strip() for req in rust_extra
    }

    rust_core_dir = REPO_ROOT / "rust_core"
    crate_pyprojects = sorted(rust_core_dir.glob("*/pyproject.toml"))
    assert len(crate_pyprojects) > 0, (
        f"No pyproject.toml files found in {rust_core_dir}/*"
    )

    for crate_pyproject in crate_pyprojects:
        with open(crate_pyproject, "rb") as f:
            crate_data = tomllib.load(f)
        crate_name = crate_data.get("project", {}).get("name")
        assert crate_name, f"Missing [project].name in {crate_pyproject}"
        assert crate_name in declared_rust_packages, (
            f"Maturin crate '{crate_name}' from {crate_pyproject.relative_to(REPO_ROOT)} "
            f"is missing from [project.optional-dependencies].rust in pyproject.toml"
        )


def test_deploy_directory_is_populated() -> None:
    deploy_dir = REPO_ROOT / "deploy"
    readme_path = deploy_dir / "README.md"
    gitkeep_path = deploy_dir / ".gitkeep"

    assert readme_path.exists(), f"Missing {readme_path}"
    assert readme_path.stat().st_size > 0, f"{readme_path} must not be empty"
    assert not gitkeep_path.exists(), (
        f"deploy/.gitkeep should be removed once populated: {gitkeep_path}"
    )
