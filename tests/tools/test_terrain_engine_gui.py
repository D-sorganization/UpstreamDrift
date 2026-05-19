"""Tests for the dockable terrain engine launcher tab."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.shared.python.physics.terrain_presets import (
    ENVIRONMENT_PRESETS,
    build_environment_preset,
)
from src.tools.terrain_engine.gui import TerrainExplorerWidget, get_dockable_ui

_APP: QApplication | None = None


def _ensure_qapp() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app
    return app


def test_shared_terrain_presets_build_all_known_environments() -> None:
    terrains = [
        build_environment_preset(
            name,
            width=float(info["width"]),
            length=float(info["length"]),
        )
        for name, info in ENVIRONMENT_PRESETS.items()
    ]

    assert "full_hole" in ENVIRONMENT_PRESETS
    assert {terrain.name for terrain in terrains} == set(ENVIRONMENT_PRESETS)


def test_terrain_engine_widget_loads_full_hole_and_queries_surface() -> None:
    _ensure_qapp()
    widget = TerrainExplorerWidget()
    widget.preset_combo.setCurrentIndex(widget.preset_combo.findData("full_hole"))
    widget._load_selected_preset()

    assert widget._terrain is not None
    assert widget._terrain.name == "full_hole"
    assert widget.sample_table.rowCount() == 4
    assert "friction" in widget.query_result.text()


def test_get_dockable_ui_returns_launcher_hosted_widget() -> None:
    _ensure_qapp()
    widget = get_dockable_ui()

    assert isinstance(widget, TerrainExplorerWidget)
    assert widget.objectName() == "TerrainExplorerWidget"
