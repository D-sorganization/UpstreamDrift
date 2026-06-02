"""Regression tests for launcher paths re-homed from archive (#7104)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from src.shared.python.gui_launcher.ud_tool_catalog import UDToolCatalog

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "src" / "config" / "launcher_manifest.json"

REHOMED_MANIFEST_PATHS = {
    "mujoco_unified": "src/launchers/mujoco_unified_launcher.py",
    "matlab_unified": "src/launchers/matlab_launcher_unified.py",
    "motion_capture": "src/launchers/motion_capture_launcher.py",
}

REHOMED_CATALOG_COMMANDS = {
    "motion_capture_launcher": "src.launchers.motion_capture_launcher",
    "matlab_launcher": "src.launchers.matlab_launcher_unified",
}


def _module_from_script_path(script_path: str) -> str:
    if not script_path.endswith(".py"):
        raise ValueError(f"Expected a Python script path, got {script_path!r}")
    return script_path[:-3].replace("/", ".")


def test_rehomed_manifest_launcher_paths_exist_and_import() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    tiles_by_id = {tile["id"]: tile for tile in manifest["tiles"]}

    for tile_id, expected_path in REHOMED_MANIFEST_PATHS.items():
        tile = tiles_by_id[tile_id]
        assert tile["path"] == expected_path
        assert (REPO_ROOT / expected_path).is_file()
        assert importlib.import_module(_module_from_script_path(expected_path))


def test_rehomed_catalog_commands_import() -> None:
    tools_by_id = {tool.tool_id: tool for tool in UDToolCatalog().all_tools()}

    for tool_id, expected_command in REHOMED_CATALOG_COMMANDS.items():
        tool = tools_by_id[tool_id]
        assert tool.command == expected_command
        assert importlib.import_module(expected_command)
