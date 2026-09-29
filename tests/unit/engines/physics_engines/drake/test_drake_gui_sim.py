"""Tests for drake_gui_sim.py."""

from unittest.mock import MagicMock, patch

import pytest

from src.engines.physics_engines.drake.python.src.drake_gui_sim import (
    SimulationMixin,
)


class DummySimGUI(SimulationMixin):
    """Dummy class for testing SimulationMixin."""

    def __init__(self):
        """Initialize mock UI components."""
        self.operating_mode = "dynamic"
        self.controls_stack = MagicMock()
        self.is_running = False
        self.btn_run = MagicMock()
        self.btn_record = MagicMock()
        self.lbl_rec_status = MagicMock()
        self.time_step = 0.001
        self.simulator = MagicMock()
        self.context = MagicMock()
        self.plant = MagicMock()
        self.recorder = MagicMock()
        self.eval_context = MagicMock()
        self.visualizer = MagicMock()
        self.diagram = MagicMock()

        self.sliders = {}
        self.spinboxes = {}

    def _update_status(self, msg: str) -> None:
        """Mock update status."""

    def _sync_kinematic_sliders(self) -> None:
        """Mock sync sliders."""

    def _reset_state(self) -> None:
        """Mock reset state."""

    def _update_visualization(self) -> None:
        """Mock update viz."""

    def _is_analysis_enabled(self) -> bool:
        """Mock analysis enabled."""
        return False


@pytest.fixture
def dummy_sim() -> DummySimGUI:
    """Fixture providing a DummySimGUI instance."""
    return DummySimGUI()


class TestSimulationMixin:
    """Tests for SimulationMixin methods."""

    def test_on_mode_changed_kinematic(self, dummy_sim: DummySimGUI) -> None:
        """Test switching to kinematic mode."""
        dummy_sim._on_mode_changed("Kinematic")

        assert dummy_sim.operating_mode == "kinematic"
        dummy_sim.controls_stack.setCurrentIndex.assert_called_once_with(1)
        assert dummy_sim.is_running is False
        dummy_sim.btn_run.setChecked.assert_called_once_with(False)
        dummy_sim.btn_run.setText.assert_called_once_with("▶ Run Simulation")

    def test_on_mode_changed_dynamic(self, dummy_sim: DummySimGUI) -> None:
        """Test switching to dynamic mode."""
        dummy_sim._on_mode_changed("Dynamic")

        assert dummy_sim.operating_mode == "dynamic"
        dummy_sim.controls_stack.setCurrentIndex.assert_called_once_with(0)

    def test_toggle_run(self, dummy_sim: DummySimGUI) -> None:
        """Test toggling run state."""
        dummy_sim._toggle_run(True)
        assert dummy_sim.is_running is True
        dummy_sim.btn_run.setText.assert_called_once_with("■ Stop Simulation")

        dummy_sim._toggle_run(False)
        assert dummy_sim.is_running is False
        dummy_sim.btn_run.setText.assert_called_with("▶ Run Simulation")

    def test_reset_simulation(self, dummy_sim: DummySimGUI) -> None:
        """Test resetting simulation."""
        dummy_sim.is_running = True
        dummy_sim._reset_simulation()

        assert dummy_sim.is_running is False
        dummy_sim.btn_run.setChecked.assert_called_once_with(False)
        dummy_sim.btn_run.setText.assert_called_once_with("▶ Run Simulation")

    def test_game_loop_paused(self, dummy_sim: DummySimGUI) -> None:
        """Test game loop when paused."""
        dummy_sim.is_running = False
        dummy_sim._advance_physics = MagicMock()

        dummy_sim._game_loop()

        # Should not advance physics if paused
        dummy_sim._advance_physics.assert_not_called()

    def test_game_loop_running(self, dummy_sim: DummySimGUI) -> None:
        """Test game loop when running."""
        dummy_sim.is_running = True
        dummy_sim.operating_mode = "dynamic"
        dummy_sim._advance_physics = MagicMock()

        dummy_sim._game_loop()

        dummy_sim._advance_physics.assert_called_once()

    def test_advance_physics(self, dummy_sim: DummySimGUI) -> None:
        """Test advancing physics."""
        dummy_sim.context.get_time.return_value = 1.0
        dummy_sim.recorder.is_recording = True
        dummy_sim._record_frame = MagicMock()

        dummy_sim._advance_physics(dummy_sim.simulator, dummy_sim.context)

        dummy_sim.simulator.AdvanceTo.assert_called_once_with(1.0 + dummy_sim.time_step)
        dummy_sim._record_frame.assert_called_once_with(dummy_sim.context)

    @patch(
        "src.engines.physics_engines.drake.python.src.drake_gui_sim.QtWidgets.QMessageBox"
    )
    @patch("src.engines.physics_engines.drake.python.src.drake_gui_sim.HAS_QT", True)
    def test_export_data_empty(
        self, mock_qmessagebox: MagicMock, dummy_sim: DummySimGUI
    ) -> None:
        """Test exporting data when empty."""
        dummy_sim.recorder.times = []
        dummy_sim._export_data()

        mock_qmessagebox.warning.assert_called_once()
