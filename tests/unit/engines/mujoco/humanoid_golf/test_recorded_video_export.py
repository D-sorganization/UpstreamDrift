"""MuJoCo recorded-swing video export wired to the UI (#12888, #12878)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
mujoco = pytest.importorskip("mujoco", reason="needs mujoco")
pytest.importorskip("cv2", reason="MP4 export needs opencv")
pytest.importorskip("PyQt6", reason="needs PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.biomechanics import (  # noqa: E402
    SwingRecorder,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.simulation_controls_mixin import (  # noqa: E402
    _SimulationControlsMixin,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export import (  # noqa: E402
    export_recorded_swing_video,
)
from src.shared.python.biomechanics.biomechanics_data import (  # noqa: E402
    BiomechanicalData,
)

pytestmark = [pytest.mark.unit]

XML = """<mujoco><worldbody><light pos="0 0 3"/>
<body pos="0 0 1"><joint name="hinge" type="hinge" axis="0 1 0"/>
<geom type="capsule" size="0.05 0.3"/></body></worldbody></mujoco>"""
FRAMES = 12


@pytest.fixture
def scene() -> tuple[mujoco.MjModel, mujoco.MjData, SwingRecorder]:
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    recorder = SwingRecorder()
    recorder.start_recording()
    for i in range(FRAMES):
        recorder.record_frame(
            BiomechanicalData(
                time=i * 0.0333,
                joint_positions=np.array([0.1 * i]),
                joint_velocities=np.array([0.1]),
            )
        )
    recorder.stop_recording()
    return model, data, recorder


def _frame_count(path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def test_recorded_swing_exports_a_decodable_mp4(
    scene: tuple[mujoco.MjModel, mujoco.MjData, SwingRecorder], tmp_path: Path
) -> None:
    model, data, recorder = scene

    out = export_recorded_swing_video(
        model, data, recorder, tmp_path / "swing.mp4", size=(96, 64)
    )

    assert out.is_file() and out.stat().st_size > 0
    assert _frame_count(out) == FRAMES


def test_export_without_a_recording_says_what_to_do(
    scene: tuple[mujoco.MjModel, mujoco.MjData, SwingRecorder], tmp_path: Path
) -> None:
    model, data, _ = scene

    with pytest.raises(ValueError, match="Start Recording"):
        export_recorded_swing_video(model, data, SwingRecorder(), tmp_path / "x.mp4")


def test_export_rejects_a_recording_from_another_model(
    scene: tuple[mujoco.MjModel, mujoco.MjData, SwingRecorder], tmp_path: Path
) -> None:
    model, data, recorder = scene
    recorder.frames[0].joint_positions = np.zeros(7)

    with pytest.raises(ValueError, match="joint"):
        export_recorded_swing_video(model, data, recorder, tmp_path / "x.mp4")


class _Host(QtWidgets.QWidget, _SimulationControlsMixin):
    """Minimal ControlsTab stand-in carrying just what the mixin touches."""

    def __init__(self, sim_widget: SimpleNamespace) -> None:
        super().__init__()
        self.sim_widget = sim_widget  # type: ignore[assignment]
        self.main_window = SimpleNamespace(statusBar=lambda: None)  # type: ignore[assignment]
        self.record_btn = QtWidgets.QPushButton()
        self.recording_label = QtWidgets.QLabel("Not recording")


@pytest.fixture
def host(
    scene: tuple[mujoco.MjModel, mujoco.MjData, SwingRecorder],
) -> _Host:
    QtWidgets.QApplication.instance() or QtWidgets.QApplication(["t"])
    model, data, recorder = scene
    sim = SimpleNamespace(model=model, data=data, get_recorder=lambda: recorder)
    return _Host(sim)


def test_export_video_action_writes_the_chosen_file(
    host: _Host, tmp_path: Path
) -> None:
    out = tmp_path / "from_ui.mp4"
    with (
        patch.object(
            QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(out), "")
        ),
        patch.object(QtWidgets.QMessageBox, "information"),
        patch.object(QtWidgets.QMessageBox, "warning") as warn,
        patch.object(QtWidgets.QMessageBox, "critical") as crit,
    ):
        host.on_export_video()

    assert out.is_file(), (warn.call_args, crit.call_args)
    assert _frame_count(out) == FRAMES


def test_export_video_action_warns_when_nothing_is_recorded(
    host: _Host,
) -> None:
    host.sim_widget.get_recorder = lambda: SwingRecorder()  # type: ignore[attr-defined]
    with patch.object(QtWidgets.QMessageBox, "warning") as warn:
        host.on_export_video()

    assert warn.called


def test_recording_label_follows_the_record_button(host: _Host) -> None:
    host.on_record_toggled(True)
    assert "recording" in host.recording_label.text().lower()
    assert host.recording_label.text() != "Not recording"

    host.on_record_toggled(False)
    assert "stopped" in host.recording_label.text().lower()
