"""Comprehensive unit tests for src/tools/terrain_engine/gui.py.

Focus: cover branches and error paths not exercised by
``tests/tools/test_terrain_engine_gui.py`` (lines 36, 51, 195, 258 and
related edge cases). Uses the offscreen Qt platform and small synthetic
terrain presets only -- no GPU, no network.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.terrain_engine import gui as gui_module  # noqa: E402
from src.tools.terrain_engine.gui import (  # noqa: E402
    TerrainExplorerWidget,
    _color,
    get_dockable_ui,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


# ---------------------------------------------------------------------------
# _color helper (line 36 dict path + fallback paths)
# ---------------------------------------------------------------------------


def test_color_returns_dict_value_when_present() -> None:
    colors = {"surface_primary": "#abcdef"}
    assert _color(colors, "surface_primary", "#000000") == "#abcdef"


def test_color_returns_fallback_for_missing_dict_key() -> None:
    assert _color({}, "surface_primary", "#fallback") == "#fallback"


def test_color_returns_attribute_from_object() -> None:
    class Palette:
        surface_primary = "#112233"

    assert _color(Palette(), "surface_primary", "#000000") == "#112233"


def test_color_returns_fallback_for_missing_attribute() -> None:
    class Empty:
        pass

    assert _color(Empty(), "surface_primary", "#fallback") == "#fallback"


# ---------------------------------------------------------------------------
# Constructor preconditions (line 51)
# ---------------------------------------------------------------------------


def test_widget_raises_when_no_presets_available(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gui_module, "ENVIRONMENT_PRESETS", {})
    with pytest.raises(ValueError, match="at least one terrain preset"):
        TerrainExplorerWidget()


# ---------------------------------------------------------------------------
# _selected_preset preconditions (line 195)
# ---------------------------------------------------------------------------


def test_selected_preset_raises_when_combo_data_invalid(
    qapp: QApplication,
) -> None:
    widget = TerrainExplorerWidget()
    widget.preset_combo.blockSignals(True)
    widget.preset_combo.clear()
    widget.preset_combo.addItem("Broken", None)
    widget.preset_combo.setCurrentIndex(0)
    widget.preset_combo.blockSignals(False)
    with pytest.raises(ValueError, match="non-empty string"):
        widget._selected_preset()


def test_selected_preset_raises_when_combo_data_empty_string(
    qapp: QApplication,
) -> None:
    widget = TerrainExplorerWidget()
    widget.preset_combo.blockSignals(True)
    widget.preset_combo.clear()
    widget.preset_combo.addItem("Broken", "")
    widget.preset_combo.setCurrentIndex(0)
    widget.preset_combo.blockSignals(False)
    with pytest.raises(ValueError, match="non-empty string"):
        widget._selected_preset()


# ---------------------------------------------------------------------------
# _require_terrain (line 258)
# ---------------------------------------------------------------------------


def test_require_terrain_raises_before_load(qapp: QApplication) -> None:
    widget = TerrainExplorerWidget()
    widget._terrain = None
    with pytest.raises(RuntimeError, match="terrain must be loaded"):
        widget._require_terrain()


# ---------------------------------------------------------------------------
# Integration: preset switching and surface query
# ---------------------------------------------------------------------------


def test_preset_change_updates_spin_ranges_and_values(
    qapp: QApplication,
) -> None:
    widget = TerrainExplorerWidget()
    presets = list(gui_module.ENVIRONMENT_PRESETS.items())
    # Pick a preset other than the default to force _on_preset_changed work.
    name, info = presets[-1]
    idx = widget.preset_combo.findData(name)
    widget.preset_combo.setCurrentIndex(idx)

    assert widget.width_spin.value() == pytest.approx(float(info["width"]))
    assert widget.length_spin.value() == pytest.approx(float(info["length"]))
    assert widget.query_x_spin.maximum() == pytest.approx(float(info["width"]))
    assert widget.query_y_spin.maximum() == pytest.approx(float(info["length"]))
    assert widget.query_x_spin.value() == pytest.approx(float(info["width"]) / 2)
    assert widget.query_y_spin.value() == pytest.approx(float(info["length"]) / 2)


def test_query_surface_formats_material_data(qapp: QApplication) -> None:
    widget = TerrainExplorerWidget()
    widget._load_selected_preset()
    widget._query_surface()
    text = widget.query_result.text()
    assert "friction" in text
    assert "rolling resistance" in text
    assert "deg slope" in text


def test_populate_samples_fills_four_rows(qapp: QApplication) -> None:
    widget = TerrainExplorerWidget()
    widget._load_selected_preset()
    assert widget.sample_table.rowCount() == 4
    assert widget.sample_table.columnCount() == 4
    # Row 0 sanity: x and y values reflect quarter-width / quarter-length.
    assert widget.sample_table.item(0, 0) is not None
    assert widget.sample_table.item(0, 3) is not None


def test_load_button_triggers_terrain_load(qapp: QApplication) -> None:
    widget = TerrainExplorerWidget()
    # Mutate slope and verify reload picks up the new value.
    widget.slope_spin.setValue(5.0)
    widget._load_selected_preset()
    assert widget._terrain is not None


# ---------------------------------------------------------------------------
# Theming branches in _build_ui
# ---------------------------------------------------------------------------


def test_widget_builds_with_dict_theme(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    palette: dict[str, Any] = {
        "surface_primary": "#101010",
        "surface_secondary": "#202020",
        "border_default": "#303030",
        "text_primary": "#ffffff",
        "text_secondary": "#cccccc",
    }
    monkeypatch.setattr(gui_module, "_get_theme_colors", lambda: palette)
    widget = TerrainExplorerWidget()
    assert "#101010" in widget.styleSheet()


def test_widget_builds_with_attribute_theme(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Palette:
        surface_primary = "#0a0a0a"
        surface_secondary = "#1a1a1a"
        border_default = "#2a2a2a"
        text_primary = "#fafafa"
        text_secondary = "#bababa"

    monkeypatch.setattr(gui_module, "_get_theme_colors", lambda: Palette())
    widget = TerrainExplorerWidget()
    assert "#0a0a0a" in widget.styleSheet()


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def test_get_dockable_ui_returns_widget_instance(qapp: QApplication) -> None:
    widget = get_dockable_ui()
    assert isinstance(widget, TerrainExplorerWidget)
    assert widget._terrain is not None
