"""Regression tests scoping tests/unit/conftest.py mocks and parity markers (#15108)."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import MagicMock

import pytest

from tests.integration.cross_engine.loop_nullspace_conformance_support import (
    require_real_module,
)
from tests.unit.conftest import _MOCKED_MODULES

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
OPENSIM_PARITY_PATH = (
    REPO_ROOT / "tests/integration/cross_engine/test_opensim_engine_dynamics_parity.py"
)
REGISTRY_UNIT_TEST = REPO_ROOT / "tests/unit/pose_interchange/adapters/test_registry.py"
MOCKED_NAMES = ("pinocchio", "pinocchio.casadi", "pydrake", "pydrake.all", "casadi")


def _decorator_marker_names(node: ast.FunctionDef) -> set[str]:
    """Extract ``pytest.mark.<name>`` marker names from function decorators."""
    names: set[str] = set()
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Attribute)
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "pytest"
            and target.value.attr == "mark"
        ):
            names.add(target.attr)
    return names


@pytest.mark.parametrize("unit_first", [True, False])
def test_unit_conftest_mocks_do_not_leak_outside_unit_tree(
    tmp_path: Path, unit_first: bool
) -> None:
    """Stubs installed by tests/unit/conftest.py must not exist in non-unit tests."""
    blocker_dir = tmp_path / "site_blocker"
    blocker_dir.mkdir()
    (blocker_dir / "sitecustomize.py").write_text(
        "import sys\n"
        "from importlib.abc import MetaPathFinder\n"
        "\n"
        "class _BlockOptionalEngines(MetaPathFinder):\n"
        "    _BLOCKED = ('pinocchio', 'pydrake', 'casadi')\n"
        "\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.partition('.')[0] in self._BLOCKED:\n"
        "            raise ImportError(f'blocked in scope probe: {fullname}')\n"
        "        return None\n"
        "\n"
        "sys.meta_path.insert(0, _BlockOptionalEngines())\n",
        encoding="utf-8",
    )

    outside_dir = tmp_path / "outside_unit"
    outside_dir.mkdir()
    outside_test = outside_dir / "test_outside_unit_scope.py"
    outside_test.write_text(
        "import importlib\n"
        "import sys\n"
        "import pytest\n"
        "\n"
        f"MOCKED = {MOCKED_NAMES!r}\n"
        "\n"
        "def test_no_unit_mocks_in_sys_modules() -> None:\n"
        "    for name in MOCKED:\n"
        "        mod = sys.modules.get(name)\n"
        "        assert mod is None, f'leaked module in sys.modules[{name!r}]: {mod!r}'\n"
        "        with pytest.raises(ImportError):\n"
        "            importlib.import_module(name)\n",
        encoding="utf-8",
    )

    paths = (
        [str(REGISTRY_UNIT_TEST), str(outside_test)]
        if unit_first
        else [str(outside_test), str(REGISTRY_UNIT_TEST)]
    )
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{blocker_dir}{os.pathsep}{REPO_ROOT}{os.pathsep}{existing_pp}"
        if existing_pp
        else f"{blocker_dir}{os.pathsep}{REPO_ROOT}"
    )
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-p",
            "no:randomly",
            "-p",
            "no:xvfb",
            "-v",
            "--tb=short",
            *paths,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"Subprocess pytest failed (unit_first={unit_first}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


def test_unit_conftest_provides_stubs_inside_unit_tree() -> None:
    """Tests inside tests/unit/ still see the unit conftest sys.modules stubs."""
    for name in _MOCKED_MODULES:
        assert name in sys.modules, f"{name} missing inside tests/unit/"


def test_require_real_module_returns_file_backed_and_skips_stubs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """require_real_module must skip MagicMock/stub modules and return real ones."""
    monkeypatch.setitem(sys.modules, "pinocchio", MagicMock())
    with pytest.raises(pytest.skip.Exception):
        require_real_module("pinocchio")

    stub_without_file = types.ModuleType("stub_no_file_mod")
    monkeypatch.setitem(sys.modules, "stub_no_file_mod", stub_without_file)
    with pytest.raises(pytest.skip.Exception):
        require_real_module("stub_no_file_mod")

    real_file_mod = types.ModuleType("real_file_mod")
    real_file_mod.__file__ = str(tmp_path / "real_file_mod.py")
    monkeypatch.setitem(sys.modules, "real_file_mod", real_file_mod)
    assert require_real_module("real_file_mod") is real_file_mod


def test_opensim_dynamics_parity_markers_and_guards() -> None:
    """Cross-engine OpenSim parity must separate Pinocchio/MuJoCo deps and use markers."""
    source = OPENSIM_PARITY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(OPENSIM_PARITY_PATH))

    # Top-level module statements must not importorskip('mujoco') globally.
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "mujoco":
                    pytest.fail(
                        "Module-level mujoco = pytest.importorskip('mujoco') blocks OpenSim-only tests"
                    )

    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    msg = "Replace unguarded _import_pinocchio with shared require_real_module helper"
    assert "_import_pinocchio" not in functions, msg

    assert {"requires_pinocchio", "requires_mujoco"} <= _decorator_marker_names(
        functions["test_bias_gravity_drift_match_mujoco_and_pinocchio"]
    )
    assert "requires_pinocchio" in _decorator_marker_names(
        functions["test_inverse_dynamics_matches_pinocchio"]
    )
    assert "requires_pinocchio" in _decorator_marker_names(
        functions["test_counterfactuals_match_pinocchio"]
    )

    opensim_only = (
        "test_inverse_dynamics_rejects_wrong_length",
        "test_reset_after_step_restores_initial_state",
        "test_joint_names_follow_the_coordinates",
        "test_capabilities_match_what_these_tests_prove",
        "test_golfer_joint_names_name_each_state_coordinate",
    )
    for test_name in opensim_only:
        markers = _decorator_marker_names(functions[test_name])
        assert "requires_pinocchio" not in markers
        assert "requires_mujoco" not in markers
