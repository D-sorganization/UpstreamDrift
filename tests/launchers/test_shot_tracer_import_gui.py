"""GUI-level tests for ADR-0047 H2: Shot Tracer trajectory import (#9351).

Reuses the ``mock_flight_models``/``tracer_widget`` fixtures from
``tests/launchers/conftest.py`` (native-model plumbing mocked, pyqtgraph
disabled) so these tests focus purely on the import wiring: the button,
the imported-trajectories list, curve labeling, and refusal dialogs.
Runs headless via the session ``qapp`` fixture (``QT_QPA_PLATFORM``
forced to ``offscreen`` in that same conftest).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.launchers._shot_tracer_trajectory_import import (
    ImportedTrajectoryCurve,
    TrajectoryImportError,
)

pytestmark = pytest.mark.unit


def _curve(
    label: str = "swing_sim.flight / MacDonald-Hanzely",
) -> ImportedTrajectoryCurve:
    model_family, model_name = (part.strip() for part in label.split("/", 1))
    return ImportedTrajectoryCurve(
        label=label,
        positions=np.array([[0.0, 0.0, 0.0], [50.0, 1.0, 10.0], [90.0, 0.0, 0.0]]),
        source_id="swing_sim.flight:MacDonald-Hanzely",
        model_family=model_family,
        model_name=model_name,
        frame_id="flight_xfwd_yleft_zup",
    )


def test_import_button_exists(tracer_widget) -> None:
    assert tracer_widget.import_btn.text() == "Import Trajectory Record…"


def test_imported_list_starts_empty(tracer_widget) -> None:
    assert tracer_widget.imported_list.count() == 0
    assert tracer_widget.imported_trajectories == {}


@patch("src.launchers._shot_tracer_gui.import_trajectory_record")
def test_load_trajectory_record_success_labels_curve(
    mock_import, tracer_widget
) -> None:
    """A successful import is labeled with provenance and listed."""
    curve = _curve("swing_sim.flight / MacDonald-Hanzely")
    mock_import.return_value = curve

    tracer_widget._load_trajectory_record(Path("does-not-matter.json"))

    assert "swing_sim.flight / MacDonald-Hanzely" in tracer_widget.imported_trajectories
    stored = tracer_widget.imported_trajectories["swing_sim.flight / MacDonald-Hanzely"]
    assert stored.model_family == "swing_sim.flight"
    assert stored.model_name == "MacDonald-Hanzely"

    assert tracer_widget.imported_list.count() == 1
    assert tracer_widget.imported_list.item(0).text() == (
        "swing_sim.flight / MacDonald-Hanzely"
    )


@patch("src.launchers._shot_tracer_gui.import_trajectory_record")
def test_load_trajectory_record_ud_family_labels_curve(
    mock_import, tracer_widget
) -> None:
    """The UD-family label shape is identical: family / name, always set."""
    curve = _curve("ud.flight_models / Waterloo/Penner")
    mock_import.return_value = curve

    tracer_widget._load_trajectory_record(Path("ud_record.json"))

    assert (
        tracer_widget.imported_list.item(0).text()
        == "ud.flight_models / Waterloo/Penner"
    )
    assert "ud.flight_models / Waterloo/Penner" in tracer_widget.imported_trajectories


@patch("src.launchers._shot_tracer_gui.QMessageBox.warning")
@patch("src.launchers._shot_tracer_gui.import_trajectory_record")
def test_load_trajectory_record_refusal_shows_named_reason(
    mock_import, mock_warning, tracer_widget
) -> None:
    """A refusal never drops silently: the dialog carries the exact reason."""
    mock_import.side_effect = TrajectoryImportError(
        "unsupported frame 'app_xtarget_yup_zright': Shot Tracer's plot-frame "
        "conversion is only implemented for ['flight_xfwd_yleft_zup']"
    )

    tracer_widget._load_trajectory_record(Path("bad_frame.json"))

    mock_warning.assert_called_once()
    dialog_message = mock_warning.call_args[0][2]
    assert "unsupported frame" in dialog_message
    assert "app_xtarget_yup_zright" in dialog_message

    # A refused record is never added anywhere.
    assert tracer_widget.imported_trajectories == {}
    assert tracer_widget.imported_list.count() == 0


@patch("src.launchers._shot_tracer_gui.QMessageBox.warning")
@patch("src.launchers._shot_tracer_gui.import_trajectory_record")
def test_load_trajectory_record_unknown_field_refusal(
    mock_import, mock_warning, tracer_widget
) -> None:
    """The reader's own wire-violation reason surfaces verbatim."""
    mock_import.side_effect = TrajectoryImportError(
        "unknown trajectory fields: ['extra_field']"
    )

    tracer_widget._load_trajectory_record(Path("unknown_field.json"))

    mock_warning.assert_called_once()
    assert "unknown trajectory fields" in mock_warning.call_args[0][2]
    assert tracer_widget.imported_trajectories == {}


@patch("src.launchers._shot_tracer_gui.QFileDialog.getOpenFileName")
@patch("src.launchers._shot_tracer_gui.import_trajectory_record")
def test_import_button_click_opens_dialog_and_loads(
    mock_import, mock_dialog, tracer_widget
) -> None:
    """Clicking the button drives the dialog -> load path end to end."""
    curve = _curve()
    mock_import.return_value = curve
    mock_dialog.return_value = ("chosen_record.json", "Trajectory records (*.json)")

    tracer_widget.import_btn.click()

    mock_dialog.assert_called_once()
    mock_import.assert_called_once_with(Path("chosen_record.json"))
    assert curve.label in tracer_widget.imported_trajectories


@patch("src.launchers._shot_tracer_gui.QFileDialog.getOpenFileName")
@patch("src.launchers._shot_tracer_gui.import_trajectory_record")
def test_import_button_cancelled_dialog_imports_nothing(
    mock_import, mock_dialog, tracer_widget
) -> None:
    """Cancelling the file dialog (empty path) must not attempt an import."""
    mock_dialog.return_value = ("", "")

    tracer_widget.import_btn.click()

    mock_import.assert_not_called()
    assert tracer_widget.imported_trajectories == {}


@patch("src.launchers._shot_tracer_gui.import_trajectory_record")
def test_imported_curves_survive_clear_are_reset(mock_import, tracer_widget) -> None:
    """Clear resets imported state too, mirroring native-result clearing."""
    mock_import.return_value = _curve()
    tracer_widget._load_trajectory_record(Path("record.json"))
    assert tracer_widget.imported_list.count() == 1

    tracer_widget._clear_visualization()

    assert tracer_widget.imported_trajectories == {}
    assert tracer_widget.imported_list.count() == 0


@patch("src.launchers._shot_tracer_gui.import_trajectory_record")
def test_imported_curve_plotted_alongside_native_results(
    mock_import, qapp, mock_flight_models
) -> None:
    """With pyqtgraph available, an imported curve gets its own GL line.

    Patches ``PYQTGRAPH_AVAILABLE``/``gl`` on the *implementation* module
    (``_shot_tracer_gui``) directly, never the ``shot_tracer`` facade:
    ``_sync_public_overrides`` only overwrites the implementation's
    globals when the facade's own copy is non-``None``, so a facade-level
    patch here would leave a stale ``True``/mock ``gl`` behind after this
    test's ``@patch`` context exits (the facade reverts to ``None``,
    which the sync guard skips) — corrupting every later test that
    constructs a widget without itself patching pyqtgraph availability.
    Patching the implementation module directly is correctly saved and
    restored by ``mock.patch`` regardless of that sync path.
    """
    from PyQt6.QtWidgets import QWidget

    from src.launchers.shot_tracer import MultiModelShotTracerWidget

    class MockGLViewWidget(QWidget):
        def setCameraPosition(self, **kwargs) -> None:
            pass

        def addItem(self, item) -> None:
            pass

        def removeItem(self, item) -> None:
            pass

        def clear(self) -> None:
            pass

    mock_gl = MagicMock()
    mock_gl.GLViewWidget.return_value = MockGLViewWidget()
    mock_plot_item = MagicMock()
    mock_gl.GLLinePlotItem.return_value = mock_plot_item

    with (
        patch("src.launchers._shot_tracer_gui.PYQTGRAPH_AVAILABLE", True),
        patch("src.launchers._shot_tracer_gui.gl", mock_gl),
    ):
        parent_widget = QWidget()
        widget = MultiModelShotTracerWidget(parent=parent_widget)
        widget.gl_widget.addItem = MagicMock(wraps=widget.gl_widget.addItem)

        curve = _curve()
        mock_import.return_value = curve
        widget._load_trajectory_record(Path("record.json"))

        assert f"imported:{curve.label}" in widget.trajectory_plots
        widget.gl_widget.addItem.assert_any_call(mock_plot_item)
