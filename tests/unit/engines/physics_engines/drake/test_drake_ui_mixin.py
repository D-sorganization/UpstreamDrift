"""Tests for drake_ui_mixin.py."""

from unittest.mock import MagicMock, patch

import pytest

from src.engines.physics_engines.drake.python.src.drake_ui_mixin import (
    DrakeUIMixin,
    HAS_QT,
)

if not HAS_QT:
    pytest.skip(
        "Skipping PyQt tests since PyQt6 is not available", allow_module_level=True
    )


class DummyUIGUI(DrakeUIMixin):
    """Dummy class to test DrakeUIMixin."""

    def __init__(self):
        """Initialize mock UI components."""
        self.available_models = [{"name": "Default Golf Model", "path": None}]
        self.operating_mode = "dynamic"
        self.is_running = False
        self.time_step = 0.001
        self.plant = MagicMock()
        self.context = MagicMock()
        self.diagram = MagicMock()
        self.visualizer = MagicMock()
        self.recorder = MagicMock()
        self.sliders = {}
        self.spinboxes = {}

        # We need setCentralWidget since it's a QMainWindow method
        self.setCentralWidget = MagicMock()

    def get_joint_names(self):
        """Mock get_joint_names."""
        return ["j1"]

    def _update_status(self, msg: str) -> None:
        """Mock update status."""

    def _update_visualization(self) -> None:
        """Mock update visualization."""

    def _on_model_changed(self, idx):
        pass

    def _show_induced_acceleration_plot(self):
        pass

    def _show_counterfactuals_plot(self):
        pass

    def _show_swing_plane_analysis(self):
        pass

    def _show_advanced_plots(self):
        pass

    def _export_data(self):
        pass

    def _show_overlay_dialog(self):
        pass

    def _on_visualization_changed(self):
        pass


@pytest.fixture
def dummy_ui() -> DummyUIGUI:
    """Fixture providing a DummyUIGUI instance."""
    return DummyUIGUI()


class TestDrakeUIMixin:
    """Tests for DrakeUIMixin methods."""

    @patch("src.engines.physics_engines.drake.python.src.drake_ui_mixin.LivePlotWidget")
    @patch("src.engines.physics_engines.drake.python.src.drake_ui_mixin.QtCore")
    @patch("src.engines.physics_engines.drake.python.src.drake_ui_mixin.QtGui")
    @patch("src.engines.physics_engines.drake.python.src.drake_ui_mixin.QtWidgets")
    def test_setup_ui(
        self,
        mock_qt_widgets: MagicMock,
        mock_qt_gui: MagicMock,
        mock_qt_core: MagicMock,
        mock_live_plot: MagicMock,
        dummy_ui: DummyUIGUI,
    ) -> None:
        """Test _setup_ui runs without crashing."""
        # This function heavily uses Qt, so we just mock everything and make sure it calls through
        dummy_ui._build_kinematic_controls = MagicMock()
        dummy_ui._show_induced_acceleration_plot = MagicMock()
        dummy_ui._show_counterfactuals_plot = MagicMock()
        dummy_ui._show_swing_plane_analysis = MagicMock()
        dummy_ui._show_advanced_plots = MagicMock()
        dummy_ui._export_data = MagicMock()
        dummy_ui._show_overlay_dialog = MagicMock()
        dummy_ui._on_visualization_changed = MagicMock()

        dummy_ui._setup_ui()

        # Verify it created UI elements
        assert hasattr(dummy_ui, "model_combo")
        assert hasattr(dummy_ui, "mode_combo")
        assert hasattr(dummy_ui, "main_tab_widget")
        assert hasattr(dummy_ui, "controls_stack")
        assert hasattr(dummy_ui, "btn_run")

        # Verify it built kinematic controls
        dummy_ui._build_kinematic_controls.assert_called_once()

    def test_on_mode_changed_kinematic(self, dummy_ui: DummyUIGUI) -> None:
        """Test switching to kinematic mode."""
        dummy_ui.controls_stack = MagicMock()
        dummy_ui.btn_run = MagicMock()
        dummy_ui._sync_kinematic_sliders = MagicMock()

        dummy_ui._on_mode_changed("Kinematic")

        assert dummy_ui.operating_mode == "kinematic"
        dummy_ui.controls_stack.setCurrentIndex.assert_called_once_with(1)
        assert dummy_ui.is_running is False
        dummy_ui._sync_kinematic_sliders.assert_called_once()

    def test_on_mode_changed_dynamic(self, dummy_ui: DummyUIGUI) -> None:
        """Test switching to dynamic mode."""
        dummy_ui.controls_stack = MagicMock()
        dummy_ui.btn_run = MagicMock()
        dummy_ui.is_running = True

        dummy_ui._on_mode_changed("Dynamic")

        assert dummy_ui.operating_mode == "dynamic"
        dummy_ui.controls_stack.setCurrentIndex.assert_called_once_with(0)
        dummy_ui.btn_run.setText.assert_called_with("■ Stop Simulation")

    def test_toggle_run(self, dummy_ui: DummyUIGUI) -> None:
        """Test toggling run state."""
        dummy_ui.btn_run = MagicMock()

        dummy_ui._toggle_run(True)
        assert dummy_ui.is_running is True
        dummy_ui.btn_run.setText.assert_called_with("■ Stop Simulation")

        dummy_ui._toggle_run(False)
        assert dummy_ui.is_running is False
        dummy_ui.btn_run.setText.assert_called_with("▶ Run Simulation")

    def test_reset_simulation(self, dummy_ui: DummyUIGUI) -> None:
        """Test resetting simulation."""
        dummy_ui.btn_run = MagicMock()
        dummy_ui._reset_state = MagicMock()

        dummy_ui._reset_simulation()

        assert dummy_ui.is_running is False
        dummy_ui.btn_run.setChecked.assert_called_once_with(False)
        dummy_ui._reset_state.assert_called_once()

    def test_toggle_recording(self, dummy_ui: DummyUIGUI) -> None:
        """Test toggling recording."""
        dummy_ui.btn_record = MagicMock()
        dummy_ui.recorder.times = [1, 2, 3]

        dummy_ui._toggle_recording(True)
        dummy_ui.recorder.start.assert_called_once()
        dummy_ui.btn_record.setText.assert_called_with("Stop Recording")

        dummy_ui._toggle_recording(False)
        dummy_ui.recorder.stop.assert_called_once()
        dummy_ui.btn_record.setText.assert_called_with("Record")
