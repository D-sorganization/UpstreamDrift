"""Coverage tests for ``starting_pose_matcher.gui_source_panel``.

Avoids ``pytest.importorskip("PyQt6.QtWidgets", ...)`` which crashes the
Windows / Python 3.14 PyQt6 build at pytest collection time
(``Windows fatal exception 0xc0000139`` on re-import). Instead we
import PyQt6 directly inside a try/except and skip the module if the
import fails for any reason.

Test-only; no production code changes (issue #4673).
"""

from __future__ import annotations

import os

# Headless GUI BEFORE any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

# pytest-qt loads PySide6 at startup on this Windows / Python 3.14 install,
# which prevents subsequent PyQt6 DLL loading. When we detect the conflict
# we skip cleanly. CI Linux uses PyQt6 only and runs every test below.
if "PySide6" in sys.modules:
    pytest.skip(
        "PySide6 already loaded — PyQt6 DLLs unavailable", allow_module_level=True
    )

try:
    from PyQt6.QtWidgets import QApplication  # noqa: F401  (used in fixture)

    _HAVE_QT = True
except Exception:  # noqa: BLE001 — DLL load can raise OSError or worse
    _HAVE_QT = False

if not _HAVE_QT:  # pragma: no cover - environment-dependent
    pytest.skip("PyQt6.QtWidgets unavailable", allow_module_level=True)


from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.starting_pose_matcher.gui_source_panel import (  # noqa: E402
    BODY_FILE_FILTER,
    CLUB_FILE_FILTER,
    DataSourcesPanel,
    _try_clamp_signed_int,
    shared_timegrid_ok,
)
from src.tools.starting_pose_matcher.session_schema import (  # noqa: E402
    DEFAULT_BODY_MARKER_SET,
    AlignOptionsBlock,
    BodySourceBlock,
    ClubSourceBlock,
    DataSourcesBlock,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# --------------------------------------------------------------------------- #
# Pure helpers (no Qt instance needed)                                       #
# --------------------------------------------------------------------------- #


def test_clamp_signed_int_in_bounds():
    assert _try_clamp_signed_int(5, 0, 10) == 5


def test_clamp_signed_int_below_lower():
    assert _try_clamp_signed_int(-3, 0, 10) == 0


def test_clamp_signed_int_above_upper():
    assert _try_clamp_signed_int(99, 0, 10) == 10


def test_shared_timegrid_ok_equal_arrays():
    from types import SimpleNamespace

    a = SimpleNamespace(time=np.array([0.0, 0.1, 0.2]))
    b = SimpleNamespace(time=np.array([0.0, 0.1, 0.2]))
    assert shared_timegrid_ok(a, b) is True


def test_shared_timegrid_ok_unequal_arrays():
    from types import SimpleNamespace

    a = SimpleNamespace(time=np.array([0.0, 0.1, 0.2]))
    b = SimpleNamespace(time=np.array([0.0, 0.1, 0.3]))
    assert shared_timegrid_ok(a, b) is False


def test_shared_timegrid_ok_different_shapes():
    from types import SimpleNamespace

    a = SimpleNamespace(time=np.array([0.0, 0.1]))
    b = SimpleNamespace(time=np.array([0.0, 0.1, 0.2]))
    assert shared_timegrid_ok(a, b) is False


def test_shared_timegrid_ok_none_inputs():
    assert shared_timegrid_ok(None, None) is False
    from types import SimpleNamespace

    a = SimpleNamespace(time=None)
    b = SimpleNamespace(time=np.zeros(3))
    assert shared_timegrid_ok(a, b) is False


def test_filters_are_generic():
    for token in ("Wiffle", "ProV1", "TW_", "GW_"):
        assert token not in CLUB_FILE_FILTER
        assert token not in BODY_FILE_FILTER
    assert ".xlsx" in CLUB_FILE_FILTER
    assert ".c3d" in BODY_FILE_FILTER


# --------------------------------------------------------------------------- #
# Panel construction + state round-trips (Qt-light)                          #
# --------------------------------------------------------------------------- #


def test_panel_builds_with_default_state(qapp):
    panel = DataSourcesPanel()
    assert panel.title() == "Data sources"
    # Default state.
    assert panel.cb_club.isChecked() is False
    assert panel.cb_body.isChecked() is False
    assert panel.rb_club_only.isChecked() is True
    assert panel.rb_align_impact.isChecked() is True
    assert panel.combo_marker_set.currentText() == DEFAULT_BODY_MARKER_SET


def test_panel_widgets_have_object_names(qapp):
    panel = DataSourcesPanel()
    for name in (
        "cb_club",
        "cb_body",
        "btn_club_browse",
        "btn_body_browse",
        "rb_club_only",
        "rb_club_ball",
        "lbl_club_path",
        "lbl_body_path",
        "combo_marker_set",
        "rb_align_impact",
        "rb_align_address",
        "spin_sample_rate",
        "spin_duration",
    ):
        assert panel.findChild(object, name) is not None, f"missing widget: {name}"


def test_align_options_returns_live_values(qapp):
    panel = DataSourcesPanel()
    panel.spin_sample_rate.setValue(1500)
    panel.spin_duration.setValue(0.500)
    panel.rb_align_address.setChecked(True)
    opts = panel.align_options()
    assert opts.sample_rate_hz == pytest.approx(1500.0)
    assert opts.simulation_time_s == pytest.approx(0.5)
    assert opts.time_alignment == "address"


def test_snapshot_round_trip(qapp):
    panel = DataSourcesPanel()
    panel.cb_club.setChecked(True)
    panel.rb_club_ball.setChecked(True)
    panel.cb_body.setChecked(True)
    panel.combo_marker_set.setCurrentText("All markers")
    panel.spin_sample_rate.setValue(800)
    panel.spin_duration.setValue(0.250)
    block = panel.snapshot()
    assert block.club.enabled is True
    assert block.club.include_ball is True
    assert block.body.enabled is True
    assert block.body.marker_set == "All markers"
    assert block.align.sample_rate_hz == pytest.approx(800.0)
    assert block.align.time_alignment == "impact"


def test_restore_with_none_uses_default(qapp):
    panel = DataSourcesPanel()
    panel.cb_club.setChecked(True)
    panel.restore(None)
    assert panel.cb_club.isChecked() is False
    assert panel.combo_marker_set.currentText() == DEFAULT_BODY_MARKER_SET


def test_restore_full_block(qapp):
    panel = DataSourcesPanel()
    block = DataSourcesBlock(
        club=ClubSourceBlock(
            enabled=True, file_path="/tmp/foo.xlsx", include_ball=True
        ),
        body=BodySourceBlock(
            enabled=True, file_path="/tmp/body.c3d", marker_set="Lower body only"
        ),
        align=AlignOptionsBlock(
            sample_rate_hz=2000.0, simulation_time_s=0.45, time_alignment="address"
        ),
    )
    panel.restore(block)
    assert panel.cb_club.isChecked() is True
    assert panel.rb_club_ball.isChecked() is True
    assert panel.cb_body.isChecked() is True
    assert panel.combo_marker_set.currentText() == "Lower body only"
    assert panel.spin_sample_rate.value() == 2000
    assert panel.spin_duration.value() == pytest.approx(0.45)
    assert panel.rb_align_address.isChecked() is True
    # snapshot round-trips back.
    block2 = panel.snapshot()
    assert block2 == block


def test_restore_unknown_marker_set_keeps_combo_intact(qapp):
    panel = DataSourcesPanel()
    block = DataSourcesBlock(body=BodySourceBlock(marker_set="not-a-real-set"))
    panel.restore(block)
    # Combo wasn't changed (findText returned -1).
    assert panel.combo_marker_set.currentText() == DEFAULT_BODY_MARKER_SET


def test_restore_no_file_paths_shows_placeholder(qapp):
    panel = DataSourcesPanel()
    panel.restore(DataSourcesBlock())
    assert panel.lbl_club_path.text() == "(no file)"
    assert panel.lbl_body_path.text() == "(no file)"


def test_emit_targets_with_no_slots_emits_none(qapp):
    panel = DataSourcesPanel()
    received: list = []
    panel.targets_changed.connect(received.append)
    # Toggling a checkbox with nothing loaded -> emits None.
    panel.cb_club.setChecked(True)
    panel.cb_club.setChecked(False)
    assert received[-1] is None


def test_current_targets_initially_none(qapp):
    panel = DataSourcesPanel()
    assert panel.current_targets() is None


def test_force_set_body_target_without_club(qapp):
    panel = DataSourcesPanel()
    received: list = []
    panel.targets_changed.connect(received.append)
    from types import SimpleNamespace

    body = SimpleNamespace(
        time=np.array([0.0, 0.1, 0.2]), marker_xyz=np.zeros((3, 1, 3))
    )
    panel._force_set_body_target(body, "/tmp/body.c3d")
    # body-only is allowed by MultiSourceTarget; latest target should be set.
    # Even if it isn't (e.g. validator rejects), no crash.
    assert panel.lbl_body_path.text() == "body.c3d"
