"""Tests for the dockable terrain engine launcher tab."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ui]

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


def test_terrain_engine_widget_builder_failure_shows_error_without_abort(
    monkeypatch,
) -> None:
    """Issue #8890: exceptions raised during preset building must be caught and show dialog."""
    _ensure_qapp()

    def failing_builder(*args, **kwargs):
        raise ValueError("Invalid slope/direction combination for preset")

    import src.shared.python.physics.terrain_presets as presets_mod

    monkeypatch.setattr(presets_mod, "build_environment_preset", failing_builder)
    for _mod in list(sys.modules.values()):
        if hasattr(_mod, "build_environment_preset"):
            monkeypatch.setattr(_mod, "build_environment_preset", failing_builder)

    widget = TerrainExplorerWidget()
    assert "Failed to Load Terrain" in widget.query_result.text()
    assert widget.query_btn.isEnabled() is False


def test_terrain_engine_widget_query_without_terrain_shows_error_without_abort(
    monkeypatch,
) -> None:
    """Issue #8890: query without loaded terrain must be caught and show error dialog."""
    _ensure_qapp()
    widget = TerrainExplorerWidget()
    widget._terrain = None
    widget._query_surface()

    assert "Terrain Query Error" in widget.query_result.text()
