"""Tests for src/shared/python/theme/tool_stylesheet.py (issue #8885).

Covers the new UD-owned home for tool-facing stylesheet tokens: it must
resolve live theme colours when available, fall back safely when it
cannot, and expose one canonical "primary action" (run button) style so
every tool shares the same colour.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytestmark = pytest.mark.unit

# NOTE: ``from src.shared.python.theme import tool_stylesheet`` (submodule via
# fromlist) resolves the canonical module's OWN ``__name__``
# (``shared.python.theme``) when retrying the fromlist import, bypassing the
# seam redirect finder entirely. Importing the dotted module path directly
# goes through the finder correctly, matching the existing
# ``from src.shared.python.theme.layout_metrics import LayoutMetrics``
# convention used elsewhere in this repo (see src/shared/python/_seam_redirect.py).
import src.shared.python.theme.tool_stylesheet as ts


def test_primary_action_color_is_a_valid_hex_literal() -> None:
    assert ts.PRIMARY_ACTION_COLOR.startswith("#")
    assert len(ts.PRIMARY_ACTION_COLOR) == 7


def test_primary_button_style_uses_the_canonical_token() -> None:
    style = ts.primary_button_style()
    assert ts.PRIMARY_ACTION_COLOR in style
    assert ts.PRIMARY_ACTION_HOVER_COLOR in style
    assert "QPushButton" in style


def test_get_tool_colors_falls_back_when_theme_lookup_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Law of Demeter: callers never touch startup/palette themselves, so a
    broken live-theme lookup must degrade to the fallback palette instead of
    raising (see issue #8972 for the underlying gap this guards against).
    """

    def _boom() -> None:
        raise ImportError("theme package unavailable")

    monkeypatch.setattr(
        "src.shared.python.theme.palette.get_current_colors",
        _boom,
    )
    colors = ts.get_tool_colors()
    assert colors == ts._FALLBACK_COLORS


def test_get_tool_colors_uses_live_theme_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = {
        "surface_primary": "#000001",
        "surface_secondary": "#000002",
        "border_default": "#000003",
        "text_primary": "#000004",
        "text_secondary": "#000005",
    }
    monkeypatch.setattr(
        "src.shared.python.theme.palette.get_current_colors",
        lambda: live,
    )
    assert ts.get_tool_colors() == live


def test_apply_tool_theme_sets_a_non_empty_stylesheet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication, QWidget

    app = QApplication.instance() or QApplication([])
    _ = app
    widget = QWidget()
    ts.apply_tool_theme(widget)
    sheet = widget.styleSheet()
    assert sheet
    assert "QWidget" in sheet
    widget.deleteLater()
