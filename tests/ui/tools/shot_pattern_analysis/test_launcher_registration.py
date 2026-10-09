"""Launcher surfaces for the deterministic shot-pattern analysis tool."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from src.launchers.embedded_tool_bootstrap import FALLBACK_ADAPTER_MODULES

ROOT = Path(__file__).resolve().parents[4]
TOOL_ID = "shot_pattern_analysis"
ADAPTER = "src.tools.shot_pattern_analysis._embed_adapter"
pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_tool_has_a_launcher_tile_and_adapter_registration() -> None:
    models = yaml.safe_load((ROOT / "src/config/models.yaml").read_text())
    tile = next(model for model in models["models"] if model["id"] == TOOL_ID)

    assert tile["type"] == "special_app"
    assert tile["launcher"]["category"] == "tool"
    assert (ROOT / tile["path"]).is_file()
    assert ADAPTER in FALLBACK_ADAPTER_MODULES

    manifest = json.loads(
        (ROOT / "src/config/launcher_manifest.json").read_text(encoding="utf-8")
    )
    public_tile = next(item for item in manifest["tiles"] if item["id"] == TOOL_ID)
    assert public_tile["path"] == tile["path"]
    assert public_tile["category"] == "tool"

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert (
        project["project"]["entry-points"]["upstream_drift.embeddable_tools"][TOOL_ID]
        == ADAPTER
    )


def test_embed_adapter_import_does_not_load_pyqt() -> None:
    code = (
        f"import importlib, sys; importlib.import_module({ADAPTER!r}); "
        "assert 'src.tools.shot_pattern_analysis.gui' not in sys.modules; "
        "assert 'PyQt6' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={
            **os.environ,
            "QT_QPA_PLATFORM": "offscreen",
            "MPLBACKEND": "Agg",
            "MUJOCO_GL": "egl",
            "SDL_VIDEODRIVER": "dummy",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
