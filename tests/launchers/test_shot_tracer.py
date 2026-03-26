"""Tests for shot_tracer."""

from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

from src.launchers.shot_tracer import (  # noqa: E402
    MultiModelShotTracerWidget,
    MultiModelShotTracerWindow,
)


@pytest.fixture
def mock_flight_models():
    class MockModelType:
        value = "mock"

    class MockModel:
        name = "Mock Model"
        description = "Mock Description"
        reference = "Mock Ref"

    with (
        patch("src.launchers.shot_tracer.FlightModelType", [MockModelType]) as ModelTypeMock,
        patch("src.launchers.shot_tracer.FlightModelRegistry") as RegistryMock,
    ):
        RegistryMock.get_model.return_value = MockModel()
        yield ModelTypeMock, RegistryMock


@pytest.fixture
def tracer_widget(qapp, mock_flight_models):
    with patch("src.launchers.shot_tracer.PYQTGRAPH_AVAILABLE", False):
        from PyQt6.QtWidgets import QWidget

        parent_widget = QWidget()
        widget = MultiModelShotTracerWidget(parent=parent_widget)
        yield widget


def test_widget_init(tracer_widget):
    assert len(tracer_widget.model_checkboxes) == 1
    assert tracer_widget.speed_spin.value() == 163.0


def test_apply_preset(tracer_widget):
    tracer_widget._apply_preset("7iron")
    assert tracer_widget.speed_spin.value() == 118.0
    assert tracer_widget.angle_spin.value() == 16.0
    assert tracer_widget.spin_spin.value() == 7000.0


def test_apply_preset_unknown(tracer_widget):
    tracer_widget.speed_spin.setValue(100.0)
    tracer_widget._apply_preset("unknown_club")
    assert tracer_widget.speed_spin.value() == 100.0


def test_get_selected_models(tracer_widget):
    models = tracer_widget._get_selected_models()
    assert len(models) == 1

    tracer_widget.model_checkboxes["mock"].setChecked(False)
    models = tracer_widget._get_selected_models()
    assert len(models) == 0


@patch("src.launchers.shot_tracer.QMessageBox.warning")
def test_run_comparison_no_models(mock_warning, tracer_widget):
    tracer_widget.model_checkboxes["mock"].setChecked(False)
    tracer_widget._run_comparison()
    mock_warning.assert_called_once()
    assert "Please select at least one model." in mock_warning.call_args[0][2]


@patch("src.launchers.shot_tracer.UnifiedLaunchConditions")
@patch("src.launchers.shot_tracer.compare_models")
def test_run_comparison_success(mock_compare, mock_launch, tracer_widget):
    mock_result = MagicMock()
    mock_result.carry_distance = 100.0
    mock_result.max_height = 50.0
    mock_result.flight_time = 5.0
    mock_result.landing_angle = 45.0
    mock_result.to_position_array.return_value = []

    mock_compare.return_value = {"Mock Model": mock_result}

    tracer_widget._run_comparison()

    mock_compare.assert_called_once()
    assert tracer_widget.results_table.rowCount() == 1

    item = tracer_widget.results_table.item(0, 0)
    assert item.text() == "Mock Model"


@patch("src.launchers.shot_tracer.QMessageBox.warning")
@patch("src.launchers.shot_tracer.UnifiedLaunchConditions")
@patch("src.launchers.shot_tracer.compare_models")
def test_run_comparison_error(mock_compare, mock_launch, mock_warning, tracer_widget):
    mock_compare.side_effect = ValueError("Test error")

    tracer_widget._run_comparison()

    mock_warning.assert_called_once()
    assert "Test error" in mock_warning.call_args[0][2]


def test_clear_visualization(tracer_widget):
    tracer_widget.results = {"test": "result"}
    tracer_widget.results_table.setRowCount(1)

    tracer_widget._clear_visualization()

    assert len(tracer_widget.results) == 0
    assert tracer_widget.results_table.rowCount() == 0


def test_window_init(qapp, mock_flight_models):
    with patch("src.launchers.shot_tracer.PYQTGRAPH_AVAILABLE", False):
        window = MultiModelShotTracerWindow()
        assert window.windowTitle() == "Golf Shot Tracer - Multi-Model Comparison"
        assert isinstance(window.central_widget, MultiModelShotTracerWidget)


@patch("src.launchers.shot_tracer.PYQTGRAPH_AVAILABLE", True)
@patch("src.launchers.shot_tracer.gl")
def test_pyqtgraph_available_visualization(mock_gl, qapp, mock_flight_models):
    from PyQt6.QtWidgets import QWidget

    parent_widget = QWidget()

    class MockGLViewWidget(QWidget):
        def setCameraPosition(self, **kwargs):
            pass

        def addItem(self, item):
            pass

        def removeItem(self, item):
            pass

        def clear(self):
            pass

    mock_gl.GLViewWidget.return_value = MockGLViewWidget()

    mock_plot_item = MagicMock()
    mock_gl.GLLinePlotItem.return_value = mock_plot_item

    widget = MultiModelShotTracerWidget(parent=parent_widget)

    # Test update_visualization
    mock_result = MagicMock()
    mock_result.to_position_array.return_value = [[0, 0, 0]]
    widget.results = {"Mock Model": mock_result}

    # Add an existing item to hit the clear old trajectories logic
    widget.trajectory_plots["old"] = MagicMock()

    widget._update_visualization()

    assert "Mock Model" in widget.trajectory_plots

    # Test clear_visualization with pyqtgraph available
    widget._clear_visualization()
    assert len(widget.results) == 0
    assert len(widget.trajectory_plots) == 0


def test_update_results_table_no_header(tracer_widget):
    # Simulate header being None
    tracer_widget.results_table.horizontalHeader = MagicMock(return_value=None)
    tracer_widget.results = {
        "Mock Model": MagicMock(
            carry_distance=100.0, max_height=50.0, flight_time=5.0, landing_angle=45.0
        )
    }
    tracer_widget._update_results_table()
    assert tracer_widget.results_table.rowCount() == 1


@patch("src.launchers.shot_tracer.QApplication")
@patch("src.launchers.shot_tracer.MultiModelShotTracerWindow")
@patch("src.launchers.shot_tracer.sys.exit")
def test_main(mock_exit, mock_window, mock_app):
    from src.launchers.shot_tracer import main

    mock_app_instance = MagicMock()
    mock_app.return_value = mock_app_instance
    mock_window_instance = MagicMock()
    mock_window.return_value = mock_window_instance

    main()

    mock_app_instance.setStyle.assert_called_with("Fusion")
    mock_window_instance.show.assert_called_once()
    mock_app_instance.exec.assert_called_once()
    mock_exit.assert_called_once()
