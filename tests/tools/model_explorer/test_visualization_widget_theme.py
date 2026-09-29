"""Regression test for the light-mode panel inside the dark app (issue #8885).

``VisualizationWidget.info_label`` used to hardcode
``background-color: #f5f5f5; color: #333`` -- a light-grey panel with dark
text docked inside the otherwise-dark launcher. It must use the shared
theme tokens instead.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
import pytest

from src.shared.python.theme.tool_stylesheet import get_tool_colors
from src.tools.model_explorer.visualization_widget import VisualizationWidget

pytestmark = [pytest.mark.unit, pytest.mark.ui]

_LIGHT_PANEL_BUG_LITERALS = ("#f5f5f5", "#333")

_APP: QApplication | None = None


def _ensure_qapp() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app
    return app


@pytest.fixture(autouse=True)
def _qapp() -> None:
    _ensure_qapp()


def test_info_label_does_not_hardcode_the_light_panel_colours() -> None:
    widget = VisualizationWidget()
    sheet = widget.info_label.styleSheet()
    for literal in _LIGHT_PANEL_BUG_LITERALS:
        assert literal not in sheet, f"light-panel literal {literal!r} still present"
    widget.deleteLater()


def test_info_label_uses_the_shared_theme_tokens() -> None:
    widget = VisualizationWidget()
    colors = get_tool_colors()
    sheet = widget.info_label.styleSheet()
    assert colors["surface_secondary"] in sheet
    assert colors["text_secondary"] in sheet
    widget.deleteLater()
