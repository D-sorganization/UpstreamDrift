"""Native QTest receipt-to-heatmap flows; all study values are synthetic.

The real verifier, load model and renderer run without OpenSim or mocked slots.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytestmark = pytest.mark.unit


@pytest.fixture
def viewer():
    try:
        from PyQt6 import QtCore, QtTest, QtWidgets
        import matplotlib  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6/matplotlib not loadable: {exc}")
    from src.tools.spine_viewer.gui import SpineViewerWidget

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = SpineViewerWidget()
    widget.resize(1280, 800)
    widget.show()
    app.processEvents()
    assert widget.isVisible()
    yield widget, QtCore, QtTest.QTest, QtWidgets
    widget.cleanup()
    widget.close()


def _open_receipt(viewer, monkeypatch, path: Path | None) -> None:
    """Replace only the OS file picker; click the real connected Open button."""
    widget, core, qtest, widgets = viewer
    buttons = widget.findChildren(widgets.QPushButton)
    button = next(b for b in buttons if b.text() == "Open Study Receipt...")
    with monkeypatch.context() as patch:
        patch.setattr(
            widgets.QFileDialog,
            "getOpenFileName",
            lambda *args, **kwargs: (str(path) if path else "", "Receipt (*.json)"),
        )
        qtest.mouseClick(button, core.Qt.MouseButton.LeftButton)


def test_open_switch_load_quantity_and_cancel(
    viewer, monkeypatch, study_factory
) -> None:
    widget, core, qtest, widgets = viewer
    assert "NO STUDY LOADED" in widget.status_text()
    _open_receipt(viewer, monkeypatch, study_factory())
    assert "UNQUALIFIED" in widget.status_text()
    assert "invented-law-a" in widget.distribution_text()
    figure = widget.current_figure()
    assert figure is not None
    np.testing.assert_array_equal(
        figure.axes[0].images[0].get_array()[0], [0, 100, 300, 200]
    )
    combo = next(
        c for c in widget.findChildren(widgets.QComboBox) if c.findText("tension") >= 0
    )
    combo.setFocus()
    qtest.keyClick(combo, core.Qt.Key.Key_Down)
    assert widget.current_quantity() == "tension"
    tension = widget.current_figure()
    assert tension is not None and tension is not figure
    np.testing.assert_array_equal(
        tension.axes[0].images[0].get_array()[0], [0, -100, -300, -200]
    )
    assert "UNQUALIFIED" in tension.axes[0].get_title()
    _open_receipt(viewer, monkeypatch, None)
    assert widget.current_figure() is tension


def test_tampered_loads_are_refused_then_valid_receipt_recovers(
    viewer, monkeypatch, study_factory
) -> None:
    widget, core, qtest, widgets = viewer
    receipt = study_factory()
    _open_receipt(viewer, monkeypatch, receipt)
    loads = receipt.parent / "s1_loads.npz"
    loads.write_bytes(loads.read_bytes() + b"tampered")
    _open_receipt(viewer, monkeypatch, receipt)
    assert "ERROR" in widget.status_text()
    assert widget.current_figure() is None
    assert widget.distribution_text() == "distribution_source: -"
    combo = next(
        c for c in widget.findChildren(widgets.QComboBox) if c.findText("tension") >= 0
    )
    qtest.keyClick(combo, core.Qt.Key.Key_Down)
    assert widget.current_figure() is None
    _open_receipt(viewer, monkeypatch, study_factory())
    assert widget.current_figure() is not None
    assert "UNQUALIFIED" in widget.status_text()


def test_rejected_run_clears_previous_heatmap(
    viewer, monkeypatch, study_factory
) -> None:
    widget, core, qtest, _widgets = viewer
    _open_receipt(viewer, monkeypatch, study_factory())
    assert widget.current_figure() is not None
    previous_canvas = widget.current_figure().canvas
    edit = widget.study_id_edit()
    edit.setFocus()
    qtest.keyClick(edit, core.Qt.Key.Key_A, core.Qt.KeyboardModifier.ControlModifier)
    qtest.keyClick(edit, core.Qt.Key.Key_Backspace)
    qtest.mouseClick(widget.run_study_button(), core.Qt.MouseButton.LeftButton)
    assert "Error:" in widget.run_study_status_text()
    assert widget.run_study_button().isEnabled()
    assert not widget.is_running_study()
    assert widget.current_figure() is None
    assert not previous_canvas.isVisible()
    assert "ERROR" in widget.status_text()
    assert widget.distribution_text() == "distribution_source: -"
    qtest.keyClicks(edit, "recovered_study")
    _open_receipt(viewer, monkeypatch, study_factory())
    assert "UNQUALIFIED" in widget.status_text()
    assert widget.current_figure() is not None
