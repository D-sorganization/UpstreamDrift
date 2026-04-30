"""Tests for GH2034: type stub infrastructure validation.

Verifies that the stub packages, py.typed markers, and mixin TYPE_CHECKING
declarations are correctly in place. These tests serve as regression guards
so that stub infrastructure is not accidentally removed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Root of the repo
REPO_ROOT = Path(__file__).parent.parent.parent


@pytest.mark.unit
def test_py_typed_marker_exists_in_core() -> None:
    """PEP 561 py.typed marker must be present in src/shared/python/core/."""
    marker = REPO_ROOT / "src" / "shared" / "python" / "core" / "py.typed"
    assert marker.exists(), (
        f"py.typed marker missing at {marker}. "
        "This marker allows mypy to recognize src.shared.python.core as typed, "
        "eliminating import-untyped errors in signal_toolkit and other consumers."
    )
    # Must be a regular file (may be empty per PEP 561)
    assert marker.is_file(), f"{marker} is not a regular file"


@pytest.mark.unit
def test_pinocchio_stub_file_exists() -> None:
    """Local pinocchio stub must exist at stubs/pinocchio/__init__.pyi."""
    stub = REPO_ROOT / "stubs" / "pinocchio" / "__init__.pyi"
    assert stub.exists(), (
        f"pinocchio stub missing at {stub}. "
        "This stub eliminates import-untyped errors for pinocchio imports. "
        "Re-create it from the execution plan."
    )


@pytest.mark.unit
def test_pinocchio_stub_contains_key_symbols() -> None:
    """Pinocchio stub must declare the core symbols used in the codebase."""
    stub = REPO_ROOT / "stubs" / "pinocchio" / "__init__.pyi"
    assert stub.exists(), "pinocchio stub file not found (see test above)"

    content = stub.read_text()
    required_symbols = [
        "Model",
        "Data",
        "buildModelFromUrdf",
        "forwardKinematics",
        "rnea",
        "aba",
        "crba",
        "computeCoriolisMatrix",
        "computeKineticEnergy",
        "computePotentialEnergy",
        "neutral",
    ]
    for symbol in required_symbols:
        assert symbol in content, (
            f"pinocchio stub is missing symbol '{symbol}'. "
            "Update stubs/pinocchio/__init__.pyi to include it."
        )


@pytest.mark.unit
def test_myosuite_simulation_core_mixin_declares_attrs() -> None:
    """SimulationCoreMixin must declare env and sim as TYPE_CHECKING attributes."""
    source_file = (
        REPO_ROOT
        / "src"
        / "engines"
        / "physics_engines"
        / "myosuite"
        / "python"
        / "_simulation_core.py"
    )
    assert source_file.exists(), f"Source file not found: {source_file}"

    content = source_file.read_text()
    # TYPE_CHECKING guard must be present
    assert "TYPE_CHECKING" in content, (
        "_simulation_core.py is missing TYPE_CHECKING import. "
        "Add `from typing import TYPE_CHECKING` and declare mixin attrs."
    )
    # env and sim must be declared
    assert (
        "env:" in content
    ), "SimulationCoreMixin is missing `env:` declaration under TYPE_CHECKING guard."
    assert (
        "sim:" in content
    ), "SimulationCoreMixin is missing `sim:` declaration under TYPE_CHECKING guard."


@pytest.mark.unit
def test_myosuite_dynamics_mixin_declares_attrs() -> None:
    """DynamicsMixin must declare sim and is_initialized as TYPE_CHECKING attributes."""
    source_file = (
        REPO_ROOT
        / "src"
        / "engines"
        / "physics_engines"
        / "myosuite"
        / "python"
        / "_dynamics.py"
    )
    assert source_file.exists(), f"Source file not found: {source_file}"

    content = source_file.read_text()
    assert "TYPE_CHECKING" in content, "_dynamics.py is missing TYPE_CHECKING import."
    assert "sim:" in content, "DynamicsMixin is missing `sim:` declaration."
    assert (
        "is_initialized:" in content
    ), "DynamicsMixin is missing `is_initialized:` declaration."


@pytest.mark.unit
def test_myosuite_drift_control_mixin_declares_attrs() -> None:
    """DriftControlMixin must declare sim and is_initialized as TYPE_CHECKING attributes."""
    source_file = (
        REPO_ROOT
        / "src"
        / "engines"
        / "physics_engines"
        / "myosuite"
        / "python"
        / "_drift_control.py"
    )
    assert source_file.exists(), f"Source file not found: {source_file}"

    content = source_file.read_text()
    assert (
        "TYPE_CHECKING" in content
    ), "_drift_control.py is missing TYPE_CHECKING import."
    assert "sim:" in content, "DriftControlMixin is missing `sim:` declaration."


@pytest.mark.unit
def test_myosuite_muscle_interface_mixin_declares_attrs() -> None:
    """MuscleInterfaceMixin must declare sim as TYPE_CHECKING attribute."""
    source_file = (
        REPO_ROOT
        / "src"
        / "engines"
        / "physics_engines"
        / "myosuite"
        / "python"
        / "_muscle_interface.py"
    )
    assert source_file.exists(), f"Source file not found: {source_file}"

    content = source_file.read_text()
    assert (
        "TYPE_CHECKING" in content
    ), "_muscle_interface.py is missing TYPE_CHECKING import."
    assert "sim:" in content, "MuscleInterfaceMixin is missing `sim:` declaration."


@pytest.mark.unit
def test_no_type_ignore_import_untyped_in_signal_toolkit() -> None:
    """signal_toolkit files must not have import-untyped ignores for core.contracts."""
    signal_toolkit_dir = REPO_ROOT / "src" / "shared" / "python" / "signal_toolkit"
    assert (
        signal_toolkit_dir.exists()
    ), f"signal_toolkit not found at {signal_toolkit_dir}"

    violations = []
    for py_file in signal_toolkit_dir.rglob("*.py"):
        content = py_file.read_text()
        if "type: ignore[import-untyped]" in content and "core.contracts" in content:
            violations.append(str(py_file.relative_to(REPO_ROOT)))

    assert not violations, (
        f"These signal_toolkit files still have redundant type:ignore[import-untyped] "
        f"for core.contracts (py.typed marker should resolve this): {violations}"
    )


@pytest.mark.unit
def test_no_type_ignore_import_untyped_for_yaml_in_config() -> None:
    """Config package yaml imports must not have import-untyped ignores (types-PyYAML covers it)."""
    config_dir = REPO_ROOT / "src" / "shared" / "python" / "config"
    assert config_dir.exists(), f"config dir not found at {config_dir}"

    violations = []
    for py_file in config_dir.rglob("*.py"):
        content = py_file.read_text()
        lines = content.splitlines()
        for i, line in enumerate(lines, 1):
            if "import yaml" in line and "type: ignore[import-untyped]" in line:
                violations.append(f"{py_file.relative_to(REPO_ROOT)}:{i}")

    assert not violations, (
        f"These files still have `import yaml  # type: ignore[import-untyped]`. "
        f"types-PyYAML in dev deps should eliminate these: {violations}"
    )


@pytest.mark.unit
def test_pyproject_contains_types_pyyaml_dep() -> None:
    """pyproject.toml dev deps must include types-PyYAML for yaml stub coverage."""
    pyproject = REPO_ROOT / "pyproject.toml"
    assert pyproject.exists(), "pyproject.toml not found"

    content = pyproject.read_text()
    assert "types-PyYAML" in content, (
        "types-PyYAML missing from pyproject.toml dev dependencies. "
        "Add `types-PyYAML>=6.0` to [project.optional-dependencies] dev."
    )


@pytest.mark.unit
def test_pyproject_contains_scipy_stubs_dep() -> None:
    """pyproject.toml dev deps must include scipy-stubs for scipy stub coverage."""
    pyproject = REPO_ROOT / "pyproject.toml"
    assert pyproject.exists(), "pyproject.toml not found"

    content = pyproject.read_text()
    assert "scipy-stubs" in content, (
        "scipy-stubs missing from pyproject.toml dev dependencies. "
        "Add `scipy-stubs>=1.13.0` to [project.optional-dependencies] dev."
    )


@pytest.mark.unit
def test_pyproject_mypy_path_includes_stubs() -> None:
    """pyproject.toml [tool.mypy] must set mypy_path = 'stubs' for local stub discovery."""
    pyproject = REPO_ROOT / "pyproject.toml"
    assert pyproject.exists(), "pyproject.toml not found"

    content = pyproject.read_text()
    assert 'mypy_path = "stubs"' in content, (
        "mypy_path = 'stubs' missing from [tool.mypy] in pyproject.toml. "
        "This setting is required for mypy to find local stubs/pinocchio/__init__.pyi."
    )
