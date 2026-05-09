"""Headless smoke test for the Pose Studio main window.

Mirrors the offscreen pattern used by ``tests/ui/test_help_coverage.py``:
runs under ``QT_QPA_PLATFORM=offscreen`` and skips cleanly when PyQt6
or matplotlib is unavailable.
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


def test_window_constructs_without_exception(studio) -> None:  # noqa: ANN001
    """The window must instantiate without raising."""
    assert studio is not None
    assert studio.windowTitle() == "Pose Studio"


def test_engine_picker_swap_succeeds(studio) -> None:  # noqa: ANN001
    """Swapping the engine via the picker must not raise."""
    from src.tools.pose_studio.core import SUPPORTED_ENGINES

    if len(SUPPORTED_ENGINES) < 2:
        pytest.skip("need >= 2 engines to exercise the picker")
    target = SUPPORTED_ENGINES[1]
    studio.engine_picker.combo.setCurrentText(target)
    assert studio._engine_controller.engine_name == target  # noqa: SLF001
    assert studio.units_badge.text()


def test_load_reference_pose(studio) -> None:  # noqa: ANN001
    """Loading the reference pose populates the joint panel."""
    studio._on_load_reference()  # noqa: SLF001
    angles = studio._engine_controller.pose.angles_full_dict_deg()  # noqa: SLF001
    # The reference pose has a non-zero forward spine tilt.
    assert angles["SpineStartPositionX"] != 0.0


def test_undo_redo_cycle(studio) -> None:  # noqa: ANN001
    """Undo/redo through the history controller must round-trip."""
    studio._on_load_reference()  # push a pose  # noqa: SLF001
    assert studio._history.can_undo  # noqa: SLF001
    studio._on_undo()  # noqa: SLF001
    assert studio._history.can_redo  # noqa: SLF001
    studio._on_redo()  # noqa: SLF001
    assert not studio._history.can_redo  # noqa: SLF001


def test_view_3d_updates_with_pose(studio) -> None:  # noqa: ANN001
    """The 3D view's update_pose must accept a CanonicalPose."""
    from src.shared.python.pose_interchange.canonical import (
        canonical_from_reference_setup,
    )

    studio.view_3d.update_pose(canonical_from_reference_setup())


def test_save_load_buttons_show_stub_tooltip(studio) -> None:  # noqa: ANN001
    """Save/Load buttons must surface the #4900 stub tooltip."""
    assert "#4900" in studio.btn_save.toolTip()
    assert "#4900" in studio.btn_load.toolTip()
