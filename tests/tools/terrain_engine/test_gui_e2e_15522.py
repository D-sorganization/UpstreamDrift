"""Native headless QTest input -> real terrain builder -> surface queries.

No widgets, terrain factories, elevation maps or materials are mocked.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTEST_QT_API", "pyqt6")
pytest.importorskip("pytestqt")
try:
    import PyQt6.QtCore as qtcore
    import PyQt6.QtTest  # noqa: F401
    import PyQt6.QtWidgets  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(f"PyQt6 not loadable: {exc}", allow_module_level=True)
if not isinstance(qtcore.PYQT_VERSION_STR, str):
    pytest.skip("Real PyQt6 required", allow_module_level=True)

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QDoubleSpinBox, QPushButton  # noqa: E402
from pytestqt.qtbot import QtBot  # noqa: E402

from src.shared.python.physics.terrain_presets import ENVIRONMENT_PRESETS  # noqa: E402
from src.tools.terrain_engine.gui import (  # noqa: E402
    TerrainExplorerWidget,
    get_dockable_ui,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def explorer(qtbot: QtBot) -> Iterator[TerrainExplorerWidget]:
    widget = get_dockable_ui()
    qtbot.addWidget(widget)
    widget.resize(1280, 800)
    widget.show()
    qtbot.waitUntil(widget.isVisible)
    try:
        yield widget
    finally:
        widget.close()
        assert not widget.isVisible()


def _click(widget: TerrainExplorerWidget, text: str) -> None:
    buttons = [b for b in widget.findChildren(QPushButton) if b.text() == text]
    assert len(buttons) == 1
    button = buttons[0]
    assert button.isVisible() and button.isEnabled()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)


def _edit(spin: QDoubleSpinBox, text: str) -> None:
    spin.setFocus()
    QTest.keyClick(spin, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(spin, text)
    QTest.keyClick(spin, Qt.Key.Key_Tab)
    assert spin.value() == pytest.approx(float(text))


def _select(widget: TerrainExplorerWidget, preset: str) -> None:
    combo = widget.preset_combo
    index = combo.findData(preset)
    assert index >= 0
    combo.setFocus()
    QTest.keyClick(combo, Qt.Key.Key_Home)
    for _ in range(index):
        QTest.keyClick(combo, Qt.Key.Key_Down)
    assert combo.currentData() == preset


@pytest.mark.parametrize("preset", list(ENVIRONMENT_PRESETS))
def test_launcher_select_load_and_query_real_presets(
    explorer: TerrainExplorerWidget, preset: str
) -> None:
    assert explorer._terrain is not None
    _select(explorer, preset)
    _click(explorer, "Load Terrain")
    terrain = explorer._terrain
    assert terrain is not None and terrain.name == preset
    assert preset in explorer.summary.text()
    assert explorer.sample_table.rowCount() == 4
    for row in range(4):
        x = float(explorer.sample_table.item(row, 0).text())
        y = float(explorer.sample_table.item(row, 1).text())
        assert float(explorer.sample_table.item(row, 2).text()) == pytest.approx(
            terrain.elevation.get_elevation(x, y), abs=0.0005
        )
        assert explorer.sample_table.item(row, 3).text() == (
            terrain.get_terrain_type(x, y).name.lower()
        )
    _edit(explorer.query_x_spin, "2")
    _edit(explorer.query_y_spin, "2")
    _click(explorer, "Query Surface")
    material = terrain.get_material(2, 2)
    assert (
        f"friction {material.friction_coefficient:.2f}" in explorer.query_result.text()
    )
    assert (
        f"{terrain.elevation.get_elevation(2, 2):.3f} m" in explorer.query_result.text()
    )


@pytest.mark.parametrize("width,length", [(24, 36), (6, 8)])
def test_custom_extent_reload_updates_query_bounds(
    explorer: TerrainExplorerWidget, width: int, length: int
) -> None:
    _select(explorer, "putting_green")
    _edit(explorer.width_spin, str(width))
    _edit(explorer.length_spin, str(length))
    _edit(explorer.slope_spin, "4")
    _click(explorer, "Load Terrain")
    assert explorer.query_x_spin.maximum() == width
    assert explorer.query_y_spin.maximum() == length
    _edit(explorer.query_x_spin, str(width - 1))
    _edit(explorer.query_y_spin, str(length - 1))
    _click(explorer, "Query Surface")
    assert "4.00 deg slope" in explorer.query_result.text()
    assert "friction" in explorer.query_result.text()
    assert f"{width:.1f} m x {length:.1f} m" in explorer.summary.text()

    _select(explorer, "bunker")
    _click(explorer, "Load Terrain")
    assert explorer._terrain.name == "bunker"
    assert explorer.query_x_spin.maximum() == 20
    assert explorer.query_y_spin.maximum() == 20
    assert "0.00 deg slope" in explorer.query_result.text()
