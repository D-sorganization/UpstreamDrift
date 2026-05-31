"""Tests for the core-only install isolation guard."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_GUARD_SCRIPT = ROOT / "scripts" / "check_core_install_isolation.py"
_spec = importlib.util.spec_from_file_location(
    "check_core_install_isolation", _GUARD_SCRIPT
)
assert _spec is not None
assert _spec.loader is not None
_guard_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _guard_module
_spec.loader.exec_module(_guard_module)

CORE_IMPORT_MODULES = _guard_module.CORE_IMPORT_MODULES
FORBIDDEN_OPTIONAL_MODULES = _guard_module.FORBIDDEN_OPTIONAL_MODULES
find_import_isolation_violations = _guard_module.find_import_isolation_violations

pytestmark = pytest.mark.core_only


def test_core_install_guard_reports_importable_optional_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must fail when an optional engine package is importable."""

    def fake_find_spec(name: str) -> ModuleSpec | None:
        if name == "drake":
            return ModuleSpec("drake", loader=None)
        return None

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)

    violations = find_import_isolation_violations(import_core_modules=False)

    assert "drake is importable in a core-only environment" in violations


def test_core_install_guard_reports_forbidden_modules_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must fail if a core import loads an optional engine module."""
    fake_module = types.ModuleType("pinocchio")
    monkeypatch.setitem(sys.modules, "pinocchio", fake_module)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    violations = find_import_isolation_violations(import_core_modules=False)

    assert "pinocchio is loaded after core imports" in violations


def test_core_install_guard_has_explicit_contract() -> None:
    """The forbidden and core import surfaces are intentionally small."""
    assert FORBIDDEN_OPTIONAL_MODULES == (
        "drake",
        "pinocchio",
        "opensim",
        "myosuite",
        "jaxsim",
    )
    assert CORE_IMPORT_MODULES == (
        "src.api",
        "src.engines.physics_engines.mujoco",
        "src.shared.python.physics",
        "src.shared.python.spatial_algebra",
    )


def test_core_install_guard_removes_script_directory_from_import_search() -> None:
    """Running the guard as a script must not expose scripts/jaxsim as jaxsim."""
    script_root = Path("scripts").resolve()

    assert script_root not in {Path(entry or ".").resolve() for entry in sys.path}


def test_config_import_does_not_require_pandas() -> None:
    """Config package imports must stay available in the core-only environment."""
    script = """
import builtins

real_import = builtins.__import__

def block_pandas(name, *args, **kwargs):
    if name == "pandas" or name.startswith("pandas."):
        raise ModuleNotFoundError("No module named 'pandas'")
    return real_import(name, *args, **kwargs)

builtins.__import__ = block_pandas

import src.shared.python.config  # noqa: F401
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
