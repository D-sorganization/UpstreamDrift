"""Verify the Force Plates tab is wired into the main viewer window."""

from __future__ import annotations

import sys

import pytest

from ._viewer_test_helpers import make_synthetic_model

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qt_app():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app


def test_main_viewer_has_force_plates_tab(qt_app) -> None:
    from src.apps.c3d_viewer import C3DViewerMainWindow  # type: ignore

    win = C3DViewerMainWindow()
    assert win.tabs.count() >= 6
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert "Force Plates" in titles


def test_force_plot_tab_handles_empty_analog(qt_app) -> None:
    from src.apps.ui.tabs.force_plot_tab import ForcePlotTab  # type: ignore

    tab = ForcePlotTab()
    model = make_synthetic_model(["m0", "m1"], n_frames=20, include_analog=False)
    tab.update_from_model(model)
    assert "No force-plate data" in tab.status_label.text()


def test_force_plot_tab_no_match_pattern(qt_app) -> None:
    """Analog present but no force-plate-pattern channels -> graceful no-data."""
    from src.apps.ui.tabs.force_plot_tab import ForcePlotTab  # type: ignore

    tab = ForcePlotTab()
    model = make_synthetic_model(["m0"], n_frames=10, include_analog=True)
    # Synthetic analog channel "Fz1" matches the pattern; rename it so it doesn't.
    from src.apps.core.models import AnalogData  # type: ignore

    model.analog = {
        "EMG_biceps": AnalogData(
            name="EMG_biceps", values=model.analog["Fz1"].values, unit="V"
        )
    }
    tab.update_from_model(model)
    assert "No force-plate data" in tab.status_label.text()
