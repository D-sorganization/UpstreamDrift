"""Native offscreen QTest flows for the Pinocchio Golf Model window (#15522)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pinocchio")
pytest.importorskip("meshcat")
pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402

from src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui import (  # noqa: E402
    PinocchioGUI,
)

# PyQt6's installed stub incorrectly types the QTest namespace as an instance.
_mouse_click: Any = QTest.mouseClick
_key_click: Any = QTest.keyClick
_key_clicks: Any = QTest.keyClicks


@pytest.fixture
def window(qtbot: Any) -> Iterator[PinocchioGUI]:
    result = PinocchioGUI()
    qtbot.addWidget(result)
    result.show()
    index = next(
        i
        for i, m in enumerate(result.available_models)
        if m["path"] and m["path"].endswith("arm.urdf")
    )
    result.model_combo.setCurrentIndex(index)
    assert result.model is not None
    yield result
    result.timer.stop()


def test_launch_loads_arm_and_builds_kinematic_controls(window: PinocchioGUI) -> None:
    assert window.isVisible()
    assert window.model is not None
    assert window.model.nv >= 1
    assert len(window.joint_sliders) == len(window.joint_spinboxes) > 0
    assert window.timer.isActive()


def test_run_pause_reset_and_record_flow(window: PinocchioGUI, qtbot: Any) -> None:
    assert window.q is not None
    start = np.array(window.q, copy=True)
    _mouse_click(window.btn_record, Qt.MouseButton.LeftButton)
    _mouse_click(window.btn_run, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.recorder.get_num_frames() >= 3, timeout=10000)
    assert window.btn_run.text() == "Pause Simulation"
    assert window.sim_time > 0.0
    assert window.q is not None
    assert not np.allclose(window.q, start)

    _mouse_click(window.btn_run, Qt.MouseButton.LeftButton)
    assert window.btn_run.text() == "Run Simulation"
    frozen = window.sim_time
    qtbot.wait(100)
    assert window.sim_time == frozen

    _mouse_click(window.btn_record, Qt.MouseButton.LeftButton)
    assert window.btn_record.text() == "Record"
    assert window.get_recording_frame_count() >= 3

    _mouse_click(window.btn_reset, Qt.MouseButton.LeftButton)
    assert window.sim_time == 0.0
    assert window.q is not None
    assert np.allclose(window.q, start)
    assert window.lbl_rec_status.text() == "Frames: 0"


def test_kinematic_mode_spinbox_drives_model_pose(window: PinocchioGUI) -> None:
    window.mode_combo.setCurrentIndex(1)
    assert window.operating_mode == "kinematic"
    spin = window.joint_spinboxes[0]
    slider = window.joint_sliders[0]
    spin.setFocus()
    spin.selectAll()
    _key_clicks(spin, "0.5")
    _key_click(spin, Qt.Key.Key_Return)
    assert window.q is not None
    assert window.q[0] == pytest.approx(0.5)
    assert slider.value() == 50

    window.mode_combo.setCurrentIndex(0)
    assert window.operating_mode == "dynamic"


def test_standalone_close_stops_physics_timer(window: PinocchioGUI) -> None:
    _mouse_click(window.btn_run, Qt.MouseButton.LeftButton)
    assert window.timer.isActive()
    window.close()
    assert not window.isVisible()
    assert not window.timer.isActive(), "Closing the window must stop the physics loop"
    assert not window.is_running
