"""Tests for shot_tracer.

The ``mock_flight_models``/``tracer_widget`` fixtures used below live in
``tests/launchers/conftest.py`` (moved there in ADR-0047 H2, #9351) so
every test module in this directory can use them without an import.
"""

import runpy
from pathlib import Path
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
from src.launchers.shot_tracer import (  # noqa: E402
    MultiModelShotTracerWidget,
    MultiModelShotTracerWindow,
)

pytestmark = pytest.mark.unit


def test_widget_init(tracer_widget) -> None:
    assert len(tracer_widget.model_checkboxes) == 1
    assert tracer_widget.speed_spin.value() == 163.0


def test_no_dead_animation_timer(tracer_widget) -> None:
    # Regression for #6903: the never-started/connected animation QTimer (and
    # its orphaned animation_index counter) were dead code and have been
    # removed.
    assert not hasattr(tracer_widget, "animation_timer")
    assert not hasattr(tracer_widget, "animation_index")


def test_apply_preset(tracer_widget) -> None:
    tracer_widget._apply_preset("7iron")
    assert tracer_widget.speed_spin.value() == 118.0
    assert tracer_widget.angle_spin.value() == 16.0
    assert tracer_widget.spin_spin.value() == 7000.0


def test_apply_preset_unknown(tracer_widget) -> None:
    tracer_widget.speed_spin.setValue(100.0)
    tracer_widget._apply_preset("unknown_club")
    assert tracer_widget.speed_spin.value() == 100.0


def test_get_selected_models(tracer_widget) -> None:
    models = tracer_widget._get_selected_models()
    assert len(models) == 1

    tracer_widget.model_checkboxes["mock"].setChecked(False)
    models = tracer_widget._get_selected_models()
    assert len(models) == 0


@patch("src.launchers.shot_tracer.QMessageBox.warning")
def test_run_comparison_no_models(mock_warning, tracer_widget) -> None:
    tracer_widget.model_checkboxes["mock"].setChecked(False)
    tracer_widget._run_comparison()
    mock_warning.assert_called_once()
    assert "Please select at least one model." in mock_warning.call_args[0][2]


@patch("src.launchers.shot_tracer.UnifiedLaunchConditions")
@patch("src.launchers.shot_tracer.compare_models")
def test_run_comparison_success(mock_compare, mock_launch, tracer_widget) -> None:
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
def test_run_comparison_error(
    mock_compare, mock_launch, mock_warning, tracer_widget
) -> None:
    mock_compare.side_effect = ValueError("Test error")

    tracer_widget._run_comparison()

    mock_warning.assert_called_once()
    assert "Test error" in mock_warning.call_args[0][2]


def test_clear_visualization(tracer_widget) -> None:
    tracer_widget.results = {"test": "result"}
    tracer_widget.results_table.setRowCount(1)

    tracer_widget._clear_visualization()

    assert len(tracer_widget.results) == 0
    assert tracer_widget.results_table.rowCount() == 0


def test_window_init(qapp, mock_flight_models) -> None:
    with patch("src.launchers.shot_tracer.PYQTGRAPH_AVAILABLE", False):
        window = MultiModelShotTracerWindow()
        assert window.windowTitle() == "Golf Shot Tracer - Multi-Model Comparison"
        assert isinstance(window.central_widget, MultiModelShotTracerWidget)


@patch("src.launchers._shot_tracer_gui.gl")
@patch("src.launchers._shot_tracer_gui.PYQTGRAPH_AVAILABLE", True)
@patch("src.launchers.shot_tracer.gl")
@patch("src.launchers.shot_tracer.PYQTGRAPH_AVAILABLE", True)
def test_pyqtgraph_available_visualization(
    mock_gl, _mock_impl_gl, qapp, mock_flight_models
) -> None:
    # Also patches the _shot_tracer_gui implementation module's own
    # PYQTGRAPH_AVAILABLE/gl (the unused _mock_impl_gl parameter), purely
    # so mock.patch restores THEM to their pre-test value on teardown.
    # _sync_public_overrides() copies the facade's gl/PYQTGRAPH_AVAILABLE
    # into the implementation module on every _load_pyqtgraph() call
    # whenever the facade's own copy is non-None -- which during this
    # test it always is -- so mock_gl (not _mock_impl_gl) is what
    # actually renders below. Without this second patch, only the
    # facade's copy would be restored when the decorators unwind (back
    # to None); the sync guard skips re-syncing a None facade value, so
    # the implementation module's True/mock_gl would leak past this
    # test and corrupt any later test that constructs a widget without
    # itself patching pyqtgraph availability (e.g.
    # tests/unit/test_shot_tracer.py's ``widget`` fixture).
    from PyQt6.QtWidgets import QWidget

    parent_widget = QWidget()

    class MockGLViewWidget(QWidget):
        def setCameraPosition(self, **kwargs) -> None:
            pass

        def addItem(self, item) -> None:
            pass

        def removeItem(self, item) -> None:
            pass

        def clear(self) -> None:
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


def test_update_results_table_no_header(tracer_widget) -> None:
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
def test_shot_tracer_main(mock_exit, mock_window, mock_app) -> None:
    from src.launchers.shot_tracer import main

    mock_app_instance = MagicMock()
    # No pre-existing Qt application: the fresh-construction path must run.
    mock_app.instance.return_value = None
    mock_app.return_value = mock_app_instance
    mock_window_instance = MagicMock()
    mock_window.return_value = mock_window_instance

    main()

    mock_app_instance.setStyle.assert_called_with("Fusion")
    mock_window_instance.show.assert_called_once()
    mock_app_instance.exec.assert_called_once()
    mock_exit.assert_called_once()


@patch("src.launchers.shot_tracer.QApplication")
@patch("src.launchers.shot_tracer.MultiModelShotTracerWindow")
@patch("src.launchers.shot_tracer.sys.exit")
def test_shot_tracer_main_reuses_existing_qapplication(
    mock_exit, mock_window, mock_app
) -> None:
    """An existing QApplication must be reused, never re-instantiated (#9099)."""
    from src.launchers.shot_tracer import main

    existing_app = MagicMock()
    mock_app.instance.return_value = existing_app

    main()

    mock_app.assert_not_called()
    existing_app.setStyle.assert_not_called()
    existing_app.exec.assert_called_once()
    mock_exit.assert_called_once()


def test_shot_tracer_script_entrypoint_invokes_main(monkeypatch) -> None:
    """The launcher facade must execute its GUI entry point as a script."""
    from src.launchers import _shot_tracer_gui
    from src.launchers import shot_tracer

    mock_main = MagicMock()
    monkeypatch.setattr(_shot_tracer_gui, "main", mock_main)

    runpy.run_path(Path(shot_tracer.__file__), run_name="__main__")

    mock_main.assert_called_once_with()
