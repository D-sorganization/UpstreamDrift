"""Structural regression tests for issue #2382."""

from __future__ import annotations

import ast
import json
from pathlib import Path

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
        "src/shared/python/upstream_drift_tools/process_calculators/pressure_drop_calculator/pressure_drop_api.py",
    } <= tracked_paths


def test_budgeted_facades_stay_small_and_debt_free() -> None:
    for entry in _load_policy()["facade_budgets"]:
        path = _repo_file(str(entry["path"]))
        source = path.read_text(encoding="utf-8")

        assert path.exists(), f"Missing budgeted facade: {path}"
        assert _line_count(path) <= int(entry["max_lines"])
        assert "# ARCHITECTURE_DEBT:" not in source


def test_tracked_subtargets_exist_and_match_budget() -> None:
    for entry in _load_policy()["tracked_subtargets"]:
        path = _repo_file(str(entry["path"]))
        assert path.exists(), f"Missing tracked subtarget: {path}"
        assert _line_count(path) <= int(entry["max_lines"])


def test_mesh_generator_facade_imports_split_backends() -> None:
    facade_path = _repo_file(
        "src/shared/python/humanoid_character_builder/generators/mesh_generator.py"
    )
    assert set(
        _load_policy()["facade_budgets"][0]["required_modules"]
    ) <= _imported_modules(facade_path)


def test_terrain_facade_imports_representation_loading_and_physics() -> None:
    facade_path = _repo_file("src/shared/python/physics/terrain.py")
    assert set(
        _load_policy()["facade_budgets"][1]["required_modules"]
    ) <= _imported_modules(facade_path)


def test_pressure_drop_facade_imports_split_modules() -> None:
    facade_path = _repo_file(
        "src/shared/python/upstream_drift_tools/process_calculators/pressure_drop_calculator/pressure_drop_interface.py"
    )
    assert set(
        _load_policy()["facade_budgets"][2]["required_modules"]
    ) <= _imported_modules(facade_path)
