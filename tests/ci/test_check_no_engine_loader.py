"""Tests for ``scripts/ci/check_no_engine_loader.py`` (issue #4254).

The gate enforces CROSS_ENGINE_PARITY_SPEC.md §2.1: engine-specific target
loaders are forbidden; engines must import the canonical loader at
``src.shared.python.motion_matching.load_club_target``.

Coverage:

* happy-path -- the live repo (run from its own root) passes the gate;
* bad-path -- a fake engine file with ``def load_club_target_excel(...)``
  triggers the gate;
* legacy-import bad-path -- ``from drake_loaders import ...`` triggers the
  gate;
* allow-list -- a file that re-exports ``load_club_target`` from the
  canonical module passes;
* contract -- running outside a directory containing ``pyproject.toml``
  yields exit code 2 (DbC failure), distinct from a content violation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "ci" / "check_no_engine_loader.py"


def _load_module():
    """Load the gate script as a module without polluting sys.modules forever."""
    spec = importlib.util.spec_from_file_location(
        "check_no_engine_loader_under_test", SCRIPT_PATH
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate_module():
    return _load_module()


# -- happy path --------------------------------------------------------------


def test_happy_path_current_repo_passes(monkeypatch, gate_module) -> None:
    """The live repository must pass the gate -- no engine-specific loaders today."""
    monkeypatch.chdir(REPO_ROOT)
    assert gate_module.main([]) == 0


# -- contract failure --------------------------------------------------------


def test_contract_fails_outside_repo_root(tmp_path, monkeypatch, gate_module) -> None:
    """Running with no pyproject.toml in cwd must exit 2 (DbC failure)."""
    monkeypatch.chdir(tmp_path)
    assert gate_module.main([]) == 2


def test_contract_fails_when_engine_tree_missing(
    tmp_path, monkeypatch, gate_module
) -> None:
    """A repo root without src/engines/physics_engines/ must exit 2."""
    (tmp_path / "pyproject.toml").write_text("# stub\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert gate_module.main([]) == 2


# -- bad path: engine-specific loader function ------------------------------


def _make_fake_repo(tmp_path: Path) -> Path:
    """Build a minimal fake repo skeleton with an empty engine tree."""
    (tmp_path / "pyproject.toml").write_text("# stub\n", encoding="utf-8")
    for engine in ("mujoco", "drake", "pinocchio", "opensim"):
        (tmp_path / "src" / "engines" / "physics_engines" / engine).mkdir(
            parents=True, exist_ok=True
        )
    return tmp_path


def test_bad_path_engine_specific_loader_def_fails(
    tmp_path, monkeypatch, gate_module
) -> None:
    """A ``def load_club_target_excel`` in engine code must be rejected."""
    repo = _make_fake_repo(tmp_path)
    bad_file = repo / "src" / "engines" / "physics_engines" / "mujoco" / "bad_loader.py"
    bad_file.write_text(
        "def load_club_target_excel(path):\n    return None\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)
    assert gate_module.main([]) == 1


def test_bad_path_load_target_prefix_caught(tmp_path, monkeypatch, gate_module) -> None:
    """Names starting with ``load_target`` (case-insensitive) are rejected."""
    repo = _make_fake_repo(tmp_path)
    bad_file = repo / "src" / "engines" / "physics_engines" / "drake" / "bad.py"
    bad_file.write_text(
        "async def Load_Target_From_Excel(path):\n    return None\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)
    assert gate_module.main([]) == 1


def test_bad_path_legacy_import_caught(tmp_path, monkeypatch, gate_module) -> None:
    """``from drake_loaders import X`` must be rejected."""
    repo = _make_fake_repo(tmp_path)
    bad_file = repo / "src" / "engines" / "physics_engines" / "drake" / "uses_legacy.py"
    bad_file.write_text(
        "from drake_loaders import load_club_target\nx = load_club_target\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)
    assert gate_module.main([]) == 1


def test_bad_path_legacy_plain_import_caught(
    tmp_path, monkeypatch, gate_module
) -> None:
    """``import opensim_loaders`` must be rejected."""
    repo = _make_fake_repo(tmp_path)
    bad_file = (
        repo / "src" / "engines" / "physics_engines" / "opensim" / "uses_legacy.py"
    )
    bad_file.write_text(
        "import opensim_loaders\n_ = opensim_loaders\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)
    assert gate_module.main([]) == 1


# -- allow-list: thin re-export passes --------------------------------------


def test_allow_list_thin_reexport_passes(tmp_path, monkeypatch, gate_module) -> None:
    """A re-export ``from src.shared.python.motion_matching.load_club_target import ...``
    must NOT trip the gate, even though the imported name matches a reserved
    prefix.
    """
    repo = _make_fake_repo(tmp_path)
    good_file = (
        repo / "src" / "engines" / "physics_engines" / "pinocchio" / "reexport.py"
    )
    good_file.write_text(
        "from src.shared.python.motion_matching.load_club_target import (\n"
        "    load_club_target,\n"
        ")\n"
        "\n"
        "__all__ = ['load_club_target']\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)
    assert gate_module.main([]) == 0


def test_allow_list_aliased_reexport_passes(tmp_path, monkeypatch, gate_module) -> None:
    """Aliased re-exports (``import ... as load_club_target``) also pass."""
    repo = _make_fake_repo(tmp_path)
    good_file = (
        repo / "src" / "engines" / "physics_engines" / "mujoco" / "alias_reexport.py"
    )
    good_file.write_text(
        "from src.shared.python.motion_matching.load_club_target import (\n"
        "    load_club_target_excel as load_club_target,\n"
        ")\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)
    assert gate_module.main([]) == 0


# -- unit-level: helper invariants ------------------------------------------


def test_is_reserved_name_matches_prefixes(gate_module) -> None:
    assert gate_module._is_reserved_loader_name("load_club_target")
    assert gate_module._is_reserved_loader_name("load_club_target_excel")
    assert gate_module._is_reserved_loader_name("Load_Target_Foo")
    assert gate_module._is_reserved_loader_name("load_swing_excel_data")
    assert not gate_module._is_reserved_loader_name("simulate")
    assert not gate_module._is_reserved_loader_name("load_model")


def test_resolve_relative_import_handles_levels(gate_module) -> None:
    resolved = gate_module._resolve_relative_import(
        "loader",
        level=1,
        file_dotted="src.engines.physics_engines.mujoco.foo",
    )
    assert resolved == "src.engines.physics_engines.mujoco.loader"


# -- safety: ensure module is not left registered ---------------------------


def test_module_not_left_in_sys_modules(gate_module) -> None:
    # The fixture loads under a unique sandbox name; assert no stray registration.
    assert "check_no_engine_loader_under_test" not in sys.modules or (
        sys.modules["check_no_engine_loader_under_test"] is gate_module
    )
