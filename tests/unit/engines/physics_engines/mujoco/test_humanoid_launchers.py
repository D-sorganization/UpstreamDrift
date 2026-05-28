"""Tests for MuJoCo Python Humanoid Launchers."""

import sys
from unittest.mock import MagicMock, patch
import numpy as np


from src.engines.physics_engines.mujoco.python.humanoid_launcher import (
    RemoteRecorder,
    HumanoidLauncher,
)
from PyQt6.QtWidgets import QApplication

# Keep a module-level reference so the QApplication lives for the full test
# session. Without this, `QApplication.instance() or QApplication(sys.argv)`
# creates a temporary that is immediately garbage-collected, which can destroy
# the application before any QWidget constructs and raise "Must construct a
# QApplication before a QWidget."
_qapp = QApplication.instance() or QApplication(sys.argv)


def test_remote_recorder():
    """Test RemoteRecorder."""
    recorder = RemoteRecorder()
    assert recorder.data["times"] == []

    # Test process_packet
    packet = {
        "time": 1.0,
        "qpos": [0.1, 0.2],
        "qvel": [0.3, 0.4],
        "qfrc_actuator": [0.5, 0.6],
        "iaa": {"source1": [1.0, 2.0]},
        "cf": {"ztcf": [3.0, 4.0], "zvcf": [5.0, 6.0]},
    }

    recorder.process_packet(packet)

    assert recorder.data["times"] == [1.0]
    assert np.array_equal(recorder.data["joint_positions"][0], np.array([0.1, 0.2]))
    assert np.array_equal(recorder.data["joint_velocities"][0], np.array([0.3, 0.4]))
    assert np.array_equal(recorder.data["joint_torques"][0], np.array([0.5, 0.6]))
    assert np.array_equal(
        recorder.data["induced_accelerations"]["source1"][0], np.array([1.0, 2.0])
    )
    assert np.array_equal(recorder.data["ztcf_accel"][0], np.array([3.0, 4.0]))
    assert np.array_equal(recorder.data["zvcf_accel"][0], np.array([5.0, 6.0]))

    # Test get_time_series
    times, vals = recorder.get_time_series("joint_positions")
    assert np.array_equal(times, np.array([1.0]))
    assert np.array_equal(vals[0], np.array([0.1, 0.2]))

    # Test get_induced_acceleration_series
    times, vals = recorder.get_induced_acceleration_series("source1")
    assert np.array_equal(times, np.array([1.0]))
    assert np.array_equal(vals[0], np.array([1.0, 2.0]))

    # Test export
    data = recorder.export_to_dict()
    assert "times" in data


@patch(
    "src.engines.physics_engines.mujoco.python.humanoid_launcher.ConfigurationManager"
)
def test_humanoid_launcher_init(mock_config_manager):
    """Test HumanoidLauncher initialization."""
    mock_cm_instance = MagicMock()
    mock_config = MagicMock()
    mock_cm_instance.load.return_value = mock_config
    mock_config_manager.return_value = mock_cm_instance

    # Instantiate
    with patch.object(HumanoidLauncher, "setup_ui"):
        launcher = HumanoidLauncher()

        assert launcher.config == mock_config
        assert launcher.recorder is not None


@patch(
    "src.engines.physics_engines.mujoco.python.humanoid_launcher.ConfigurationManager"
)
def test_humanoid_launcher_sim_mixin(mock_config_manager):
    """Test SimulationMixin methods."""
    mock_cm_instance = MagicMock()
    mock_config = MagicMock()
    mock_config.live_view = False
    mock_cm_instance.load.return_value = mock_config
    mock_config_manager.return_value = mock_cm_instance

    with patch.object(HumanoidLauncher, "setup_ui"):
        launcher = HumanoidLauncher()

        # test get_simulation_command
        with (
            patch("pathlib.Path.exists", return_value=False),
            patch("platform.system", return_value="Linux"),
        ):
            cmd, env = launcher.get_simulation_command()
            assert "docker" in cmd
            assert "run" in cmd
            assert "-e" in cmd
            assert "MUJOCO_GL=osmesa" in cmd

        # test start_simulation
        with (
            patch("humanoid_launcher_sim.ProcessWorker") as mock_pw,
            patch.object(launcher, "save_config"),
            patch.object(
                launcher, "get_simulation_command", return_value=(["mock_cmd"], {})
            ),
        ):
            mock_worker = MagicMock()
            mock_pw.return_value = mock_worker

            # Mock the buttons that start_simulation tries to enable/disable
            launcher.btn_run = MagicMock()
            launcher.btn_stop = MagicMock()
            launcher.txt_log = MagicMock()

            launcher.start_simulation()
            assert launcher.simulation_thread == mock_worker
            mock_worker.start.assert_called_once()


@patch(
    "src.engines.physics_engines.mujoco.python.humanoid_launcher.ConfigurationManager"
)
def test_humanoid_launcher_analysis_mixin(mock_config_manager):
    """Test AnalysisMixin methods."""
    mock_cm_instance = MagicMock()
    mock_config = MagicMock()
    mock_cm_instance.load.return_value = mock_config
    mock_config_manager.return_value = mock_cm_instance

    with patch.object(HumanoidLauncher, "setup_ui"):
        launcher = HumanoidLauncher()

        launcher.txt_log = MagicMock()
        launcher.live_plot = MagicMock()

        # test log data stream
        packet_str = 'DATA_JSON:{"time": 1.0, "qpos": [], "qvel": []}'
        launcher.log(packet_str)
        assert launcher.recorder.data["times"][-1] == 1.0
        launcher.live_plot.update_plot.assert_called_once()

        # test clear_log
        launcher.clear_log()
        launcher.txt_log.clear.assert_called_once()

        # test save_config
        launcher.spin_height = MagicMock()
        launcher.spin_height.value.return_value = 1.8
        launcher.slider_weight = MagicMock()
        launcher.slider_weight.value.return_value = 100
        launcher.slider_length = MagicMock()
        launcher.slider_length.value.return_value = 100
        launcher.slider_mass = MagicMock()
        launcher.slider_mass.value.return_value = 100
        launcher.chk_two_hand = MagicMock()
        launcher.chk_two_hand.isChecked.return_value = True
        launcher.chk_face = MagicMock()
        launcher.chk_fingers = MagicMock()
        launcher.txt_save_path = MagicMock()
        launcher.txt_load_path = MagicMock()
        launcher.chk_live = MagicMock()
        launcher.combo_control = MagicMock()
        launcher.combo_control.currentText.return_value = "PD"

        launcher.save_config()
        assert launcher.config.height_m == 1.8
        mock_cm_instance.save.assert_called_once()


def test_get_dockable_ui_returns_qwidget_container(qapp) -> None:
    """Issue #6509: get_dockable_ui must return a QWidget, not a QMainWindow.

    QMainWindow has top-level window flags by default and cannot be embedded
    as a tab in the unified launcher's DraggableTabWidget.
    """
    from PyQt6.QtWidgets import QMainWindow, QWidget

    from src.engines.physics_engines.mujoco.python.humanoid_launcher import (
        get_dockable_ui,
    )

    # Use a real QWidget as the stub (MagicMock fails QHBoxLayout.addWidget type check).
    # A plain QWidget passes Qt's C++ type guard while avoiding HumanoidLauncher init.
    fake_launcher = QWidget()

    with patch(
        "src.engines.physics_engines.mujoco.python.humanoid_launcher.HumanoidLauncher",
        return_value=fake_launcher,
    ):
        widget = get_dockable_ui()

    assert isinstance(widget, QWidget), "get_dockable_ui must return a QWidget"
    assert not isinstance(widget, QMainWindow), "container must not be a QMainWindow"
    # The inner launcher must be embedded inside the container
    assert fake_launcher.parent() is widget
