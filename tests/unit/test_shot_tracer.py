"""Unit tests for shot tracer module."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.engine_availability import PYQT6_AVAILABLE

pytestmark = pytest.mark.skipif(
    not PYQT6_AVAILABLE, reason="PyQt6 GUI libraries not available"
)

# Mock flight_models before importing shot_tracer
sys.modules["flight_models"] = MagicMock()

if PYQT6_AVAILABLE:
    from src.launchers.shot_tracer import (
        MultiModelShotTracerWidget,
        MultiModelShotTracerWindow,
    )


@pytest.fixture
def mock_flight_models():
    with (
        patch("launchers.shot_tracer.FlightModelRegistry") as mock_registry,
        patch("launchers.shot_tracer.UnifiedLaunchConditions") as mock_launch,
        patch("launchers.shot_tracer.compare_models") as mock_compare,
        patch("launchers.shot_tracer.FlightModelType") as mock_type,
    ):
        # Setup FlightModelType enum-like behavior
        mock_type.WATERLOO_PENNER.value = "waterloo_penner"
        mock_type.MACDONALD_HANZELY.value = "macdonald_hanzely"
        mock_type.NATHAN.value = "nathan"
        mock_type.__iter__.return_value = [
            mock_type.WATERLOO_PENNER,
            mock_type.MACDONALD_HANZELY,
            mock_type.NATHAN,
        ]

        # Setup Registry
        mock_model = MagicMock()
        mock_model.name = "Test Model"
        mock_model.description = "Test Description"
        mock_model.reference = "Test Reference"
        mock_registry.get_model.return_value = mock_model

        yield mock_registry, mock_launch, mock_compare, mock_type


@pytest.fixture
def widget(qtbot, mock_flight_models):
    widget = MultiModelShotTracerWidget()
    qtbot.addWidget(widget)
    return widget


def test_initialization(widget):
    """Test that the widget initializes correctly."""
    assert widget.windowTitle() == ""  # Widget doesn't have a title, Window does
    assert widget.speed_spin.value() == 163.0
    assert widget.angle_spin.value() == 11.0
    assert len(widget.model_checkboxes) == 3


def test_presets(widget):
    """Test that presets update the spin boxes."""
    # Apply 7-Iron preset
    widget._apply_preset("7iron")
    assert widget.speed_spin.value() == 118.0
    assert widget.angle_spin.value() == 16.0
    assert widget.spin_spin.value() == 7000.0

    # Apply Driver preset
    widget._apply_preset("driver")
    assert widget.speed_spin.value() == 163.0
    assert widget.angle_spin.value() == 11.0
    assert widget.spin_spin.value() == 2500.0


def test_get_selected_models(widget, mock_flight_models):
    """Test retrieval of selected models."""
    mock_registry, _, _, mock_type = mock_flight_models

    # All checked by default
    selected = widget._get_selected_models()
    assert len(selected) == 3

    # Uncheck one
    widget.model_checkboxes["waterloo_penner"].setChecked(False)
    selected = widget._get_selected_models()
    assert len(selected) == 2
    assert mock_type.WATERLOO_PENNER not in selected


def test_run_comparison_no_selection(widget, qtbot):
    """Test running comparison with no models selected."""
    # Uncheck all
    for checkbox in widget.model_checkboxes.values():
        checkbox.setChecked(False)

    with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warning:
        widget._run_comparison()
        mock_warning.assert_called_once()
        assert "No Models" in mock_warning.call_args[0][2]


def test_run_comparison_success(widget, mock_flight_models):
    """Test successful comparison run."""
    mock_registry, mock_launch, mock_compare, _ = mock_flight_models

    # Setup mock result
    mock_result = MagicMock()
    mock_result.carry_distance = 250.0
    mock_result.max_height = 30.0
    mock_result.flight_time = 6.0
    mock_result.landing_angle = 45.0
    mock_result.to_position_array.return_value = [[0, 0, 0], [100, 10, 10], [250, 0, 0]]

    mock_compare.return_value = {"Test Model": mock_result}

    widget._run_comparison()

    # Verify launch conditions created
    mock_launch.from_imperial.assert_called_once()

    # Verify compare called
    mock_compare.assert_called_once()

    # Verify results table updated
    assert widget.results_table.rowCount() == 1
    item = widget.results_table.item(0, 0)
    assert item.text() == "Test Model"
    item = widget.results_table.item(0, 1)
    # 250m * 1.09361 = 273.4 yd
    assert item.text() == "273.4"


def test_clear_visualization(widget, mock_flight_models):
    """Test clearing the visualization."""
    mock_registry, _, mock_compare, _ = mock_flight_models

    # Populate data first
    mock_result = MagicMock()
    mock_result.to_position_array.return_value = [[0, 0, 0]]
    mock_compare.return_value = {"Test Model": mock_result}
    widget._run_comparison()

    assert len(widget.results) > 0
    assert widget.results_table.rowCount() > 0

    # Clear
    widget.clear_btn.click()

    assert len(widget.results) == 0
    assert widget.results_table.rowCount() == 0


def test_window_initialization(qtbot):
    """Test the main window initialization."""
    window = MultiModelShotTracerWindow()
    qtbot.addWidget(window)
    assert window.windowTitle() == "Golf Shot Tracer - Multi-Model Comparison"
    assert isinstance(window.centralWidget(), MultiModelShotTracerWidget)
