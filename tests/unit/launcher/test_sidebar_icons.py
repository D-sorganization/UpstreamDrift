"""Regression tests for issue #5624 — every sidebar button has an icon.

Defect C: ``IconColorizer.get_icon`` raises ``ValueError`` for any
icon name not in its tiny in-memory SVG registry.  The current
sidebar requests ``sports_golf``, ``directions_run``, ``videocam``,
``build``, ``chat``, ``accessibility`` — none of which are
registered — so the ``except (ImportError, ValueError)`` block in
``_build_sidebar_button`` swallows the failure and the button ends
up text-only, matching the missing-icons screenshot in #5624.

These tests pin the contract that every sidebar nav button renders
a non-null icon pixmap at the standard 22x22 size.

Design: TDD/DbC/LOD/DRY.
"""

from __future__ import annotations

import pytest
from PyQt6.QtCore import QSize

pytestmark = pytest.mark.ui


# The ``ui_setup`` fixture is provided by tests/unit/launcher/conftest.py.


_REQUIRED_BUTTON_LABELS = (
    "Home",
    "Engines",
    "Biomechanics",
    "Simulation",
    "Motion Match",
    "MoCap",
    "Tools",
    "Chat",
    "Documentation",
    "Settings",
)


def _find_sidebar_buttons(window) -> dict[str, object]:
    from PyQt6.QtWidgets import QToolButton

    found: dict[str, object] = {}
    for btn in window.findChildren(QToolButton):
        name = btn.accessibleName() or btn.text()
        if name in _REQUIRED_BUTTON_LABELS:
            found[name] = btn
    return found


def _assert_button_has_icon(button) -> None:
    """DRY helper for the icon non-null contract."""
    icon = button.icon()
    assert not icon.isNull(), (
        f"button {button.accessibleName()!r} has a null QIcon — "
        "IconColorizer fallback path was hit"
    )
    pix = icon.pixmap(QSize(22, 22))
    assert (
        not pix.isNull()
    ), f"button {button.accessibleName()!r} renders a null 22x22 pixmap"
    assert (
        pix.width() > 0 and pix.height() > 0
    ), f"button {button.accessibleName()!r} renders a zero-sized pixmap"


class TestSidebarIcons:
    """Every left-sidebar navigation button must show a real icon."""

    @pytest.mark.parametrize("label", _REQUIRED_BUTTON_LABELS)
    def test_button_has_non_null_icon(self, ui_setup, label) -> None:
        buttons = _find_sidebar_buttons(ui_setup)
        assert (
            label in buttons
        ), f"sidebar is missing the {label!r} button — check _setup_global_sidebar()"
        _assert_button_has_icon(buttons[label])

    def test_no_button_falls_back_to_text_without_icon(self, ui_setup) -> None:
        """Visual contract: text fallback is never the *only* signal.

        ``_build_sidebar_button`` sets the text label for accessibility,
        so the button has a non-empty ``text()`` by design.  The
        regression is when the icon is null AND only text shows.  We
        enforce that every nav button has a non-null icon — together
        with the parametrized test above, this guarantees both signals
        are present.
        """
        buttons = _find_sidebar_buttons(ui_setup)
        for label, btn in buttons.items():
            assert (
                not btn.icon().isNull()
            ), f"button {label!r} fell back to text-only — icon is null"
