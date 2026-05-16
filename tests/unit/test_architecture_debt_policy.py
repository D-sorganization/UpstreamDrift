"""Structural regression tests for issues #2382, #2383, and #2388."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "config" / "architecture_debt_policy.json"


def _load_policy() -> dict[str, list[dict[str, object]]]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _repo_file(relative_path: str) -> Path:
    return REPO_ROOT / relative_path


def _line_count(path: Path) -> int:
    return sum(1 for _ in path.open(encoding="utf-8"))


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }


def test_architecture_policy_tracks_priority_subtargets() -> None:
    tracked_paths = {
        str(entry["path"]) for entry in _load_policy()["tracked_subtargets"]
    }

    assert {
        "src/shared/python/humanoid_character_builder/generators/mesh_generator_makehuman.py",
        "src/shared/python/humanoid_character_builder/generators/mesh_generator_smplx.py",
        "src/shared/python/physics/terrain_representation.py",
        "src/shared/python/pendulum_simulator/gui/equations_popup_reference_content.py",
        "src/shared/python/pendulum_simulator/gui/equations_popup_jacobian_content.py",
        "src/shared/python/sidekick/process_calculators/pressure_drop_calculator/pressure_drop_api.py",
        "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Motion Capture Plotter/motion_capture_plotter_visualization.py",
        "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Simscape Multibody Data Plotters/Python Version/golf_gui_r0/golf_visualizer_renderer.py",
        "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/scripts/dataset_generator/Dataset_GUI.m",
    } <= tracked_paths


def test_budgeted_facades_stay_small_and_debt_free() -> None:
    for entry in _load_policy()["facade_budgets"]:
        path = _repo_file(str(entry["path"]))
        source = path.read_text(encoding="utf-8")

        assert path.exists(), f"Missing budgeted facade: {path}"
        assert _line_count(path) <= int(cast(int | str, entry["max_lines"]))
        assert "# ARCHITECTURE_DEBT:" not in source


def test_tracked_subtargets_exist_and_match_budget() -> None:
    for entry in _load_policy()["tracked_subtargets"]:
        path = _repo_file(str(entry["path"]))
        assert path.exists(), f"Missing tracked subtarget: {path}"
        assert _line_count(path) <= int(cast(int | str, entry["max_lines"]))


def test_budgeted_python_facades_import_required_split_modules() -> None:
    for entry in _load_policy()["facade_budgets"]:
        facade_path = _repo_file(str(entry["path"]))
        required_modules = set(cast(list[str], entry["required_modules"]))
        assert required_modules <= _imported_modules(facade_path)
