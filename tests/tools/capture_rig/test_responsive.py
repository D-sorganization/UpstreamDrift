"""Tests for responsive window adaptation and minimum width guard (#9847).

Measured offscreen on ``main`` before the layout inversion, the tile reported a
``minimumSizeHint().width()`` of **3276 px** — wider than a 1920 px screen. The
reflowing action grid brought that floor to **1737 px**, and the inverted dock
hierarchy and responsive compaction guarantee **<= 900 px** so the tool comfortably
fits laptop panels (1366 px) and wide monitors alike without horizontal scrollbars.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from src.shared.python.core.contracts import PreconditionError
from src.tools.capture_rig.responsive import (
    DEFAULT_COMPACT_THRESHOLD_PX,
    HISTORICAL_MIN_WIDTH_PX,
    MAX_TILE_WIDTH_PX,
    LayoutMode,
    resolve_layout_mode,
)

pytestmark = [pytest.mark.unit]


# ==============================================================================
# Pure unit tests (no Qt required)
# ==============================================================================


def test_resolve_layout_mode_classifies_widths_cleanly() -> None:
    """The pure resolution function classifies widths without Qt."""
    assert resolve_layout_mode(0) == LayoutMode.COMPACT
    assert resolve_layout_mode(1280) == LayoutMode.COMPACT
    assert resolve_layout_mode(1399) == LayoutMode.COMPACT
    assert resolve_layout_mode(1400) == LayoutMode.ROOMY
    assert resolve_layout_mode(1600) == LayoutMode.ROOMY
    assert resolve_layout_mode(1920) == LayoutMode.ROOMY


def test_resolve_layout_mode_custom_threshold() -> None:
    """Custom thresholds are honored by the resolution function."""
    assert resolve_layout_mode(1000, threshold=1000) == LayoutMode.ROOMY
    assert resolve_layout_mode(999, threshold=1000) == LayoutMode.COMPACT


def test_resolve_layout_mode_preconditions() -> None:
    """Negative widths or non-positive thresholds violate DbC contracts."""
    with pytest.raises(PreconditionError):
        resolve_layout_mode(-1)
    with pytest.raises(PreconditionError):
        resolve_layout_mode(1280, threshold=0)
    with pytest.raises(PreconditionError):
        resolve_layout_mode(1280, threshold=-100)


# ==============================================================================
# Qt-backed responsive integration tests
# ==============================================================================

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QMainWindow, QScrollArea
from src.tools.capture_rig import gui

_APP: QApplication | None = None  # a dropped QApplication aborts the process


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app  # type: ignore[assignment]
    return _APP  # type: ignore[return-value]


def _settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "layouts.ini"), QSettings.Format.IniFormat)


def _shown_at(
    tmp_path: Path, width: int, height: int
) -> tuple[QMainWindow, gui.CaptureRigWidget]:
    """Mount the Capture Rig widget in a window sized to (width, height)."""
    app = _app()
    widget = gui.CaptureRigWidget(settings=_settings(tmp_path))
    window = QMainWindow()
    window.setCentralWidget(widget)
    window.resize(width, height)
    window.show()
    for _ in range(5):
        app.processEvents()
    return window, widget


@pytest.mark.ui
def test_minimum_width_standing_regression_guard(tmp_path: Path) -> None:
    """Assert the tile's minimum width stays at or under 900 px.

    Historical context:
    Measured offscreen on main before the layout inversion and responsive adaptation,
    the whole tile reported a minimumSizeHint().width() of 3276 px (wider than a
    1920 px monitor). The reflowing action grid brought that floor to 1737 px, and
    the inverted dock hierarchy and responsive compaction guarantee <= 900 px
    so the tool easily fits laptop displays (1366 px) without horizontal overflow.
    """
    _app()
    widget = gui.CaptureRigWidget(settings=_settings(tmp_path))
    min_width = widget.minimumSizeHint().width()
    assert min_width <= MAX_TILE_WIDTH_PX, (
        f"Tile minimum width {min_width} px exceeded budget {MAX_TILE_WIDTH_PX} px "
        f"(historical floor was {HISTORICAL_MIN_WIDTH_PX} px)."
    )
    widget.shutdown()


@pytest.mark.ui
def test_responsive_modes_and_preview_width_at_1280_1600_1920(tmp_path: Path) -> None:
    """The preview holds the majority of width across laptop and monitor sizes.

    At 1280 px (laptop), compact mode tabs the control docks together so the
    central live preview holds the majority of the display width and no horizontal
    scrollbar appears on the window. At 1600 px and 1920 px (wide desktop), roomy
    mode keeps the live preview wider than the controls.
    """
    app = _app()
    window, widget = _shown_at(tmp_path, 1280, 800)
    try:
        # 1280 px: Compact mode
        assert widget.layout_mode == LayoutMode.COMPACT
        live_1280 = widget.panes.centralWidget()
        assert live_1280 is not None
        controls_1280 = max(
            widget.panes.docks["rail"].width(), widget.panes.docks["inputs"].width()
        )
        assert live_1280.width() > controls_1280
        assert live_1280.width() > 1280 // 3
        # No horizontal scrollbar on window at 1280 px: window width >= minimumSizeHint
        assert window.width() >= widget.minimumSizeHint().width()

        # 1600 px: Roomy mode
        window.resize(1600, 900)
        for _ in range(5):
            app.processEvents()
        assert widget.layout_mode == LayoutMode.ROOMY
        live_1600 = widget.panes.centralWidget()
        assert live_1600 is not None
        controls_1600 = max(
            widget.panes.docks["rail"].width(), widget.panes.docks["inputs"].width()
        )
        assert live_1600.width() > controls_1600

        # 1920 px: Roomy mode
        window.resize(1920, 1080)
        for _ in range(5):
            app.processEvents()
        assert widget.layout_mode == LayoutMode.ROOMY
        live_1920 = widget.panes.centralWidget()
        assert live_1920 is not None
        controls_1920 = max(
            widget.panes.docks["rail"].width(), widget.panes.docks["inputs"].width()
        )
        assert live_1920.width() > controls_1920
    finally:
        widget.shutdown()
        window.close()


@pytest.mark.ui
def test_all_docks_wrapped_in_scroll_areas(tmp_path: Path) -> None:
    """Every dock's direct widget is a QScrollArea so overflowing content scrolls."""
    _app()
    widget = gui.CaptureRigWidget(settings=_settings(tmp_path))
    try:
        for key, dock in widget.panes.docks.items():
            dock_widget = dock.widget()
            assert isinstance(dock_widget, QScrollArea), (
                f"Dock '{key}' widget must be wrapped in a QScrollArea."
            )
    finally:
        widget.shutdown()
