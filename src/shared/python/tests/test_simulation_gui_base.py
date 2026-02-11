"""Tests for SimulationGUIBase abstract base class.

Verifies that the base class provides the correct UI skeleton and that
subclasses can properly implement the abstract interface.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

# Skip entire module if PyQt6 is not installed
pytest.importorskip("PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from src.shared.python.ui.simulation_gui_base import SimulationGUIBase  # noqa: E402

# ---------------------------------------------------------------------------
# Concrete test implementation
# ---------------------------------------------------------------------------


class ConcreteSimGUI(SimulationGUIBase):
    """Minimal concrete implementation for testing."""

    WINDOW_TITLE = "Test Simulation"
    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 480

    def __init__(self) -> None:
        self._step_count = 0
        self._reset_count = 0
        self._vis_count = 0
        self._loaded_index: int | None = None
        self._recording = False
        self._frame_count = 0
        self._exported_to: str | None = None
        super().__init__()

    def step_simulation(self) -> None:
        self._step_count += 1
        self.sim_time += 0.01

    def reset_simulation(self) -> None:
        self._reset_count += 1
        self.sim_time = 0.0

    def update_visualization(self) -> None:
        self._vis_count += 1

    def load_model(self, index: int) -> None:
        self._loaded_index = index

    def sync_kinematic_controls(self) -> None:
        pass

    def start_recording(self) -> None:
        self._recording = True

    def stop_recording(self) -> None:
        self._recording = False

    def get_recording_frame_count(self) -> int:
        return self._frame_count

    def export_data(self, filename: str) -> None:
        self._exported_to = filename

    def get_joint_names(self) -> list[str]:
        return ["joint_a", "joint_b"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp() -> Any:
    """Ensure a QApplication exists for the whole test module."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    return app


@pytest.fixture()
def gui(qapp):
    """Create a fresh ConcreteSimGUI for each test."""
    window = ConcreteSimGUI()
    yield window
    window.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWindowSetup:
    """Test basic window configuration."""

    def test_window_title(self, gui: ConcreteSimGUI) -> None:
        assert gui.windowTitle() == "Test Simulation"

    def test_window_size(self, gui: ConcreteSimGUI) -> None:
        assert gui.width() == 640
        assert gui.height() == 480

    def test_initial_state(self, gui: ConcreteSimGUI) -> None:
        assert gui.operating_mode == "dynamic"
        assert gui.is_running is False
        assert gui.sim_time == 0.0


class TestCommonWidgets:
    """Test that common UI widgets are created."""

    def test_model_combo_exists(self, gui: ConcreteSimGUI) -> None:
        assert isinstance(gui.model_combo, QtWidgets.QComboBox)

    def test_mode_combo_exists(self, gui: ConcreteSimGUI) -> None:
        assert isinstance(gui.mode_combo, QtWidgets.QComboBox)
        assert gui.mode_combo.count() == 2

    def test_run_button_exists(self, gui: ConcreteSimGUI) -> None:
        assert isinstance(gui.btn_run, QtWidgets.QPushButton)
        assert gui.btn_run.isCheckable()

    def test_reset_button_exists(self, gui: ConcreteSimGUI) -> None:
        assert isinstance(gui.btn_reset, QtWidgets.QPushButton)

    def test_record_button_exists(self, gui: ConcreteSimGUI) -> None:
        assert isinstance(gui.btn_record, QtWidgets.QPushButton)
        assert gui.btn_record.isCheckable()

    def test_visualization_checkboxes_exist(self, gui: ConcreteSimGUI) -> None:
        assert isinstance(gui.chk_show_forces, QtWidgets.QCheckBox)
        assert isinstance(gui.chk_show_torques, QtWidgets.QCheckBox)
        assert isinstance(gui.chk_mobility, QtWidgets.QCheckBox)
        assert isinstance(gui.chk_force_ellip, QtWidgets.QCheckBox)
        assert isinstance(gui.chk_live_analysis, QtWidgets.QCheckBox)

    def test_matrix_labels_exist(self, gui: ConcreteSimGUI) -> None:
        assert gui.lbl_cond.text() == "--"
        assert gui.lbl_rank.text() == "--"

    def test_main_tab_widget(self, gui: ConcreteSimGUI) -> None:
        assert isinstance(gui.main_tab_widget, QtWidgets.QTabWidget)
        assert gui.main_tab_widget.count() >= 1

    def test_controls_stack(self, gui: ConcreteSimGUI) -> None:
        assert isinstance(gui.controls_stack, QtWidgets.QStackedWidget)
        assert gui.controls_stack.count() == 2  # dynamic + kinematic


class TestModeSwitch:
    """Test dynamic/kinematic mode switching."""

    def test_switch_to_kinematic(self, gui: ConcreteSimGUI) -> None:
        gui.mode_combo.setCurrentText("Kinematic (Pose)")
        assert gui.operating_mode == "kinematic"
        assert gui.controls_stack.currentIndex() == 1
        assert gui.is_running is False

    def test_switch_to_dynamic(self, gui: ConcreteSimGUI) -> None:
        gui.mode_combo.setCurrentText("Kinematic (Pose)")
        gui.mode_combo.setCurrentText("Dynamic (Physics)")
        assert gui.operating_mode == "dynamic"
        assert gui.controls_stack.currentIndex() == 0


class TestSimulationControls:
    """Test run/pause/reset controls."""

    def test_toggle_run_starts(self, gui: ConcreteSimGUI) -> None:
        gui._toggle_run(True)
        assert gui.is_running is True
        assert "Pause" in gui.btn_run.text()

    def test_toggle_run_stops(self, gui: ConcreteSimGUI) -> None:
        gui._toggle_run(True)
        gui._toggle_run(False)
        assert gui.is_running is False
        assert "Run" in gui.btn_run.text()

    def test_reset_stops_simulation(self, gui: ConcreteSimGUI) -> None:
        gui._toggle_run(True)
        gui._on_reset_clicked()
        assert gui.is_running is False
        assert gui._reset_count == 1
        assert gui.sim_time == 0.0


class TestRecording:
    """Test recording controls."""

    def test_toggle_recording_on(self, gui: ConcreteSimGUI) -> None:
        gui.btn_record.setChecked(True)
        gui._toggle_recording()
        assert gui._recording is True
        assert "Stop" in gui.btn_record.text()

    def test_toggle_recording_off(self, gui: ConcreteSimGUI) -> None:
        gui.btn_record.setChecked(True)
        gui._toggle_recording()
        gui.btn_record.setChecked(False)
        gui._toggle_recording()
        assert gui._recording is False
        assert gui.btn_record.text() == "Record"

    def test_update_recording_label(self, gui: ConcreteSimGUI) -> None:
        gui._frame_count = 42
        gui.update_recording_label()
        assert "42" in gui.lbl_rec_status.text()


class TestGameLoop:
    """Test the game loop dispatch."""

    def test_game_loop_steps_when_running(self, gui: ConcreteSimGUI) -> None:
        gui.is_running = True
        gui._game_loop()
        assert gui._step_count == 1
        assert gui._vis_count == 1

    def test_game_loop_skips_step_when_paused(self, gui: ConcreteSimGUI) -> None:
        gui.is_running = False
        gui._game_loop()
        assert gui._step_count == 0
        assert gui._vis_count == 1  # vis always called

    def test_game_loop_skips_step_in_kinematic_mode(self, gui: ConcreteSimGUI) -> None:
        gui.operating_mode = "kinematic"
        gui.is_running = True
        gui._game_loop()
        assert gui._step_count == 0


class TestModelLoading:
    """Test model loading dispatch."""

    def test_model_changed_dispatches(self, gui: ConcreteSimGUI) -> None:
        gui.model_combo.addItem("TestModel")
        gui.model_combo.setCurrentIndex(0)
        assert gui._loaded_index == 0


class TestExport:
    """Test export controls."""

    def test_export_warns_no_data(self, gui: ConcreteSimGUI) -> None:
        gui._frame_count = 0
        with patch.object(QtWidgets.QMessageBox, "warning") as mock_warn:
            gui._on_export_clicked()
            mock_warn.assert_called_once()

    def test_export_with_data(self, gui: ConcreteSimGUI) -> None:
        gui._frame_count = 10
        with patch.object(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            return_value=("test_export.csv", ""),
        ):
            gui._on_export_clicked()
            assert gui._exported_to == "test_export.csv"


class TestAbstractEnforcement:
    """Test that abstract methods must be implemented."""

    def test_cannot_instantiate_base_directly(self, qapp) -> None:
        with pytest.raises(TypeError):
            SimulationGUIBase()  # type: ignore[abstract]
