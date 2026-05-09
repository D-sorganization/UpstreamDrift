"""Tooltip coverage test for Pose Studio.

Mirrors ``tests/ui/test_help_coverage.py``: walks the main window's
interactive widgets under ``QT_QPA_PLATFORM=offscreen`` and asserts
that every widget on the curated whitelist exposes a non-empty
``toolTip()``.
"""

from __future__ import annotations

import os

import pytest

from src.shared.python.engine_core.engine_availability import (
    skip_if_unavailable,
)

pytestmark = [
    skip_if_unavailable("pyqt6"),
    skip_if_unavailable("matplotlib"),
    pytest.mark.unit,
]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# Curated whitelist of attribute paths whose tooltip we guarantee.
# Dotted paths are resolved attribute-by-attribute on the main window.
TOOLTIP_WHITELIST: tuple[str, ...] = (
    "engine_picker.combo",
    "engine_picker.status_pill",
    "units_badge",
    "view_3d",
    "btn_save",
    "btn_load",
    "btn_undo",
    "btn_redo",
)


@pytest.fixture(scope="module")
def qapp():  # noqa: ANN201
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def studio(qapp):  # noqa: ANN001, ANN201
    from src.tools.pose_studio.gui import PoseStudioWindow

    win = PoseStudioWindow()
    yield win
    win.close()


def _resolve(obj: object, dotted: str) -> object | None:
    cur: object | None = obj
    for part in dotted.split("."):
        if cur is None:
            return None
        cur = getattr(cur, part, None)
    return cur


def test_pose_studio_tooltip_whitelist(studio) -> None:  # noqa: ANN001
    """Every whitelisted widget must have a non-empty tooltip."""
    missing: list[str] = []
    for path in TOOLTIP_WHITELIST:
        widget = _resolve(studio, path)
        if widget is None:
            missing.append(f"{path}: attribute not present")
            continue
        tip = widget.toolTip() if hasattr(widget, "toolTip") else ""
        if not tip:
            missing.append(f"{path}: tooltip is empty")
    assert not missing, "Missing tooltips:\n  - " + "\n  - ".join(missing)


def test_joint_panel_widgets_have_tooltips(studio) -> None:  # noqa: ANN001
    """Every spinbox + slider in the joint panel must have a tooltip."""
    widgets = studio.joint_panel.joint_widgets()
    assert widgets, "Joint panel exposed no widgets"
    missing = [name for name, w in widgets.items() if not w.toolTip()]
    assert not missing, "Joint widgets missing tooltips: " + ", ".join(missing)
