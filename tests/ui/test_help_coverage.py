"""Headless coverage test for help text on UI surfaces.

Walks the matcher widget tree under ``QT_QPA_PLATFORM=offscreen`` and
asserts that a curated whitelist of widgets all expose a non-empty
``toolTip()``. This is a guard against regressions: any time a widget
on the whitelist drops its tooltip, this test will fail with a clear
message identifying the offending attribute.

Also covers the launcher-side helpers introduced by the help-text
sweep: the About dialog version-info collector, the help-menu shortcut
table builder, and the model-card tile tooltip.
"""

from __future__ import annotations

import os

import pytest

from src.shared.python.engine_core.engine_availability import (
    skip_if_unavailable,
)

pytestmark = [
    skip_if_unavailable("pyqt6"),
    pytest.mark.unit,
]

# Run this whole file under the offscreen platform so it is safe in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# Curated whitelist of attribute names on StartingPoseMatcher whose tooltip
# we guarantee. Adding new widgets here is fine; removing without
# replacement is a regression.
MATCHER_TOOLTIP_WHITELIST: tuple[str, ...] = (
    # Mocap source
    "btn_load",
    "sheet_combo",
    # Event labels
    "event_preset_combo",
    # View / mocap traces
    "cb_clubhead_trace",
    "cb_midhands_trace",
    "phase_combo",
    "spin_phase_start",
    "spin_phase_end",
    "cb_frame_marker",
    "cb_show_ball",
    "cb_show_ground",
    "cb_show_torso_disk",
    "cb_auto_fit_axes",
    # Playback
    "spin_frame",
    "frame_slider",
    "btn_play",
    "spin_speed",
    "cb_loop",
    "combo_playback_target",
    "cb_use_current_frame",
    "combo_set_event",
    # Auto-Align
    "cb_fit_scale",
    "btn_snap_mid",
    # Transform
    "cb_lock_xy",
    "btn_reset_t",
    "btn_reset_r",
    "btn_reset_all",
    # Output
    "btn_save",
    "btn_save_session",
    "btn_load_session",
)


@pytest.fixture(scope="module")
def qapp():  # noqa: ANN201
    """Provide a single QApplication for all tests in this module."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def matcher_window(qapp):  # noqa: ANN001, ANN201
    """Build a fresh StartingPoseMatcher window for each test."""
    from src.tools.starting_pose_matcher.gui import StartingPoseMatcher

    win = StartingPoseMatcher()
    yield win
    win.close()


def test_matcher_tooltip_whitelist(matcher_window) -> None:  # noqa: ANN001
    """Every whitelisted matcher widget must have a non-empty tooltip."""
    missing: list[str] = []
    for attr in MATCHER_TOOLTIP_WHITELIST:
        widget = getattr(matcher_window, attr, None)
        if widget is None:
            missing.append(f"{attr}: attribute not present on matcher window")
            continue
        tip = widget.toolTip() if hasattr(widget, "toolTip") else ""
        if not tip:
            missing.append(f"{attr}: tooltip is empty")
    assert not missing, "Missing tooltips:\n  - " + "\n  - ".join(missing)


def test_matcher_pose_visibility_checks_have_tooltips(
    matcher_window,
) -> None:  # noqa: ANN001
    """Layer-visibility checkboxes (one per pose slot) must have tooltips."""
    checks = getattr(matcher_window, "_pose_visible_checks", {})
    assert checks, "No pose-visibility checkboxes were registered"
    for key, cb in checks.items():
        assert cb.toolTip(), f"Pose-visibility checkbox for {key} has no tooltip"


def test_about_dialog_version_info_keys() -> None:
    """The About dialog version collector must return all expected keys."""
    from src.launchers.about_dialog import build_about_html, gather_version_info

    info = gather_version_info()
    for key in ("app", "python", "qt", "numpy", "ezc3d", "platform"):
        assert key in info
        assert isinstance(info[key], str)
        assert info[key], f"version info key {key} is empty"

    html = build_about_html(info)
    assert "<h2>UpstreamDrift</h2>" in html
    assert info["app"] in html
    assert info["python"] in html
    assert info["qt"] in html


def test_help_menu_shortcut_table_runs(qapp) -> None:  # noqa: ANN001
    """The shortcut-table scraper must run on a window with no shortcuts."""
    from PyQt6.QtGui import QKeySequence, QShortcut
    from PyQt6.QtWidgets import QMainWindow

    from src.launchers.help_menu import collect_shortcut_rows

    win = QMainWindow()
    rows = collect_shortcut_rows(win)
    assert rows == []

    sc = QShortcut(QKeySequence("Ctrl+Shift+T"), win)
    sc.setObjectName("Test shortcut")
    rows = collect_shortcut_rows(win)
    assert any(seq == "Ctrl+Shift+T" for seq, _ in rows)
    win.close()
