"""Offscreen user journeys through Pose Studio's real adapters, FK and codecs.

Optional engines may use the application's advertised mock fallback. These
tests qualify GUI wiring and pose interchange, not engine dynamics accuracy.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtTest, QtWidgets
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(f"PyQt6/matplotlib not loadable: {exc}", allow_module_level=True)

from src.tools.pose_studio.core import SUPPORTED_ENGINES
from src.tools.pose_studio.gui import MainWidget, PoseStudioWindow
from src.tools.pose_studio.pose_files import load_pose

pytestmark = pytest.mark.unit


@pytest.fixture
def studio(monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWidget]:
    """Show a real window; never publish IPC or leave dirty-close dialogs open."""
    monkeypatch.delenv("POSE_STUDIO_PUBLISH_REALTIME", raising=False)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = PoseStudioWindow()
    window.resize(1280, 900)
    window.show()
    app.processEvents()
    try:
        yield window.main_widget
    finally:
        window.main_widget._dirty = False
        window.close()
        # Drain pending matplotlib draw_idle callbacks before destroying Qt.
        app.processEvents()
        window.deleteLater()
        app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()


def _click(button: QtWidgets.QPushButton) -> None:
    assert button.isVisible() and button.isEnabled()
    QtTest.QTest.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
    QtWidgets.QApplication.processEvents()


def _edit_joint(studio: MainWidget) -> tuple[str, QtWidgets.QDoubleSpinBox]:
    name, spin = next(iter(studio.joint_panel._spinboxes.items()))
    studio.joint_panel.ensureWidgetVisible(spin)
    spin.setFocus()
    QtTest.QTest.keyClick(spin, QtCore.Qt.Key.Key_Up)
    QtWidgets.QApplication.processEvents()
    assert spin.value() == pytest.approx(0.5)
    assert studio._engine_controller.pose.angles_full_dict_deg()[name] == 0.5
    return name, spin


def test_launch_edit_undo_redo_updates_pose_and_view(studio: MainWidget) -> None:
    assert studio.window().windowTitle() == "Pose Studio"
    assert not studio.btn_undo.isEnabled()
    name, spin = _edit_joint(studio)
    assert studio.is_dirty()
    assert studio.view_3d._service is studio._engine_controller.service
    assert studio.view_3d._service.get_link_transforms()
    _click(studio.btn_undo)
    assert spin.value() == 0.0
    assert studio._engine_controller.pose.angles_full_dict_deg()[name] == 0.0
    _click(studio.btn_redo)
    assert spin.value() == 0.5
    assert not studio.btn_redo.isEnabled()


def test_keyboard_engine_switch_rebinds_view_without_editing_history(
    studio: MainWidget,
) -> None:
    name, spin = _edit_joint(studio)
    combo = studio.engine_picker.combo
    for index, engine in enumerate(SUPPORTED_ENGINES):
        combo.setFocus()
        QtTest.QTest.keyClick(combo, QtCore.Qt.Key.Key_Home)
        for _ in range(index):
            QtTest.QTest.keyClick(combo, QtCore.Qt.Key.Key_Down)
        QtWidgets.QApplication.processEvents()
        controller = studio._engine_controller
        assert combo.currentText() == controller.engine_name == engine
        assert studio.engine_picker.status_pill.text() == controller.status.value
        assert studio.view_3d._service is controller.service
        assert studio.view_3d._landmarks_order  # includes empty-service FK fallback
        assert controller.pose.angles_full_dict_deg()[name] == 0.5
        assert spin.value() == 0.5
    _click(studio.btn_undo)
    assert spin.value() == 0.0  # engine switches must not create history entries
    assert not studio.btn_undo.isEnabled()


def test_save_edit_load_round_trips_native_pose(
    studio: MainWidget, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    name, spin = _edit_joint(studio)
    path = tmp_path / "pose.json"
    # Only the OS file pickers are replaced; Qt actions and codecs are real.
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **kw: (str(path), "")
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", lambda *a, **kw: (str(path), "")
    )
    _click(studio.btn_save)
    assert path.is_file()
    assert not studio.is_dirty()
    saved = load_pose(studio._engine_controller.engine_name, path)
    assert saved.angles_full_dict_deg()[name] == pytest.approx(0.5)
    _click(studio.btn_undo)
    assert spin.value() == 0.0
    _click(studio.btn_load)
    assert spin.value() == pytest.approx(0.5)
    assert not studio.is_dirty()
