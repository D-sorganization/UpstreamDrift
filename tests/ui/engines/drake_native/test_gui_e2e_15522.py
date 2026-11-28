"""Native Drake launch and interaction, with real Qt events and Drake physics."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTEST_QT_API", "pyqt6")


pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")


def _real_pydrake() -> bool:
    """True only for a genuine pydrake install, not a ``sys.modules`` mock."""
    import importlib

    try:
        module = importlib.import_module("pydrake")
    except ImportError:
        return False
    return hasattr(module, "__path__") and type(module).__module__ != "unittest.mock"


# unit-test-gate does not install Drake; a collection-time MagicMock would
# otherwise pass importorskip and fail inside AddMultibodyPlantSceneGraph.
if not _real_pydrake():
    pytest.skip("pydrake is not installed (or is mocked)", allow_module_level=True)

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QWidget
from pytestqt.qtbot import QtBot

from src.engines.physics_engines.drake.python.src import drake_gui_app
from src.engines.physics_engines.drake.python.src.drake_gui_app import DrakeSimApp

pytestmark = pytest.mark.unit
# Qt exposes static namespace functions; some wheel stubs add a spurious self.
_click = cast(Callable[[QWidget, Qt.MouseButton], None], QTest.mouseClick)
_key = cast(Callable[[QWidget, Qt.Key], None], QTest.keyClick)
_modified_key = cast(
    Callable[[QWidget, Qt.Key, Qt.KeyboardModifier], None], QTest.keyClick
)
_type = cast(Callable[[QWidget, str], None], QTest.keyClicks)


@pytest.fixture(params=[(1280, 800), (390, 844)])
def window(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> Iterator[DrakeSimApp]:
    """Launch the real engine window and dispose timers and Qt ownership."""
    monkeypatch.setenv("MESHCAT_HOST", "127.0.0.1")
    # Exercise the built-in golf model without the optional meshed-golfer bundle.
    monkeypatch.setattr(drake_gui_app, "MESHED_GOLFER_SPEC", tmp_path / "absent.yaml")
    widget = DrakeSimApp()
    qtbot.addWidget(widget)
    widget.resize(*request.param)
    widget.show()
    qtbot.waitUntil(widget.isVisible)
    try:
        yield widget
    finally:
        timer = widget.timer
        timer.stop()
        timer.timeout.disconnect()
        timer.deleteLater()
        widget.close()
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert sip.isdeleted(widget)
        assert sip.isdeleted(timer)


def test_launch_record_run_stop_and_reset(window: DrakeSimApp, qtbot: QtBot) -> None:
    """Clicks advance actual physics, record finite states and clear the run."""
    assert window.plant is not None
    assert window.context is not None
    assert window.plant.num_positions() > 0
    assert window.context.get_time() == 0
    _click(window.btn_record, Qt.MouseButton.LeftButton)
    _click(window.btn_run, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.get_recording_frame_count() >= 3, timeout=15_000)
    _click(window.btn_run, Qt.MouseButton.LeftButton)
    assert not window.is_running
    stopped_at = window.context.get_time()
    qtbot.wait(30)
    assert window.context.get_time() == stopped_at
    data = window.recorder.export_to_dict()
    times = data["times"]
    positions = data["joint_positions"]
    assert len(times) >= 3
    assert np.all(np.diff(times) > 0)
    assert positions.shape == (len(times), window.plant.num_positions())
    assert np.isfinite(positions).all()
    assert np.isfinite(data["joint_velocities"]).all()
    _click(window.btn_reset, Qt.MouseButton.LeftButton)
    assert window.context.get_time() == 0
    assert window.get_recording_frame_count() == 0
    assert window.lbl_rec_status.text() == "Frames: 0"
    assert not window.btn_record.isChecked()


def test_kinematic_edit_reaches_drake_and_dynamic_mode(
    window: DrakeSimApp, qtbot: QtBot
) -> None:
    """A visible joint editor updates the real plant while physics is paused."""
    from pydrake.multibody.tree import JointIndex

    combo = window.mode_combo
    combo.setFocus()
    _key(combo, Qt.Key.Key_End)
    assert window.operating_mode == "kinematic"
    assert window.controls_stack.currentIndex() == 1
    assert window.spinboxes
    joint_index, spin = next(iter(window.spinboxes.items()))
    spin.setFocus()
    _modified_key(spin, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    _type(spin, "0.25")
    _key(spin, Qt.Key.Key_Tab)
    assert spin.value() == pytest.approx(0.25)
    assert window.sliders[joint_index].value() == 25
    plant = window.plant
    context = window.context
    assert plant is not None and context is not None
    joint = plant.get_joint(JointIndex(joint_index))
    plant_context = plant.GetMyContextFromRoot(context)
    positions = plant.GetPositions(plant_context)
    assert positions[joint.position_start()] == pytest.approx(0.25)
    qtbot.wait(20)
    assert context.get_time() == 0
    combo.setFocus()
    _key(combo, Qt.Key.Key_Home)
    assert window.operating_mode == "dynamic"
    assert window.controls_stack.currentIndex() == 0
    assert not window.is_running
