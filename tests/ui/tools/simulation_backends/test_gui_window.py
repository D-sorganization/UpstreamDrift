"""Coverage for the standalone window, menu/About, ``main()``, the export
dialog handler, and the GUI error paths of the Simulation Backends tile.

These exercise the parts of :mod:`src.tools.simulation_backends_launcher.gui`
and ``__main__`` that the core-method tests in ``test_gui.py`` do not reach:
the ``QMainWindow`` wrapper, the menu/About dialog, the Qt event-loop entry
point, the ``QFileDialog``-backed export button, and the backend-unavailable
failure branch. Dialogs and the event loop are monkeypatched so the tests stay
headless and non-blocking.
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

pytest.importorskip("PyQt6")

pytestmark = pytest.mark.unit


def test_window_has_file_and_help_menus(qapp) -> None:  # noqa: ANN001
    from src.tools.simulation_backends_launcher.gui import (
        MainWidget,
        SimulationBackendsWindow,
    )

    window = SimulationBackendsWindow()
    try:
        assert isinstance(window.main_widget, MainWidget)
        menubar = window.menuBar()
        titles = [action.text() for action in menubar.actions()]
        assert any("File" in title for title in titles)
        assert any("Help" in title for title in titles)
    finally:
        window.close()


def test_about_dialog_invoked(qapp, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    from src.tools.simulation_backends_launcher import gui

    seen: dict[str, bool] = {}
    monkeypatch.setattr(
        gui.QtWidgets.QMessageBox,
        "about",
        lambda *args, **kwargs: seen.setdefault("shown", True),
    )
    window = gui.SimulationBackendsWindow()
    try:
        window._show_about()
        assert seen.get("shown") is True
    finally:
        window.close()


def test_main_entry_runs_and_returns_exit_code(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:  # noqa: ANN001
    from src.tools.simulation_backends_launcher import gui

    monkeypatch.setattr(
        gui.QtWidgets.QApplication, "exec", lambda self: 0, raising=False
    )
    assert gui.main([]) == 0


def test_export_button_writes_file_when_path_chosen(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:  # noqa: ANN001
    from src.tools.simulation_backends_launcher import gui

    widget = gui.MainWidget()
    widget.run_rollout()  # default ODE backend populates _last_trace
    out = tmp_path / "trace.h5"
    monkeypatch.setattr(
        gui.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(out), "HDF5 (*.h5)"),
    )
    widget._on_export_clicked()
    assert out.exists()


def test_export_button_cancelled_is_noop(qapp, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    from src.tools.simulation_backends_launcher import gui

    widget = gui.MainWidget()
    widget.run_rollout()
    monkeypatch.setattr(
        gui.QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: ("", ""),
    )
    widget._on_export_clicked()  # cancel branch: no exception, no write


def test_export_button_without_trace_sets_status(qapp) -> None:  # noqa: ANN001
    from src.tools.simulation_backends_launcher import gui

    widget = gui.MainWidget()
    widget._on_export_clicked()
    assert "export" in widget.status_label.text().lower()


def test_selecting_mjwarp_marks_unavailable_and_rollout_fails(qapp) -> None:  # noqa: ANN001
    from src.tools.simulation_backends_launcher import gui

    widget = gui.MainWidget()
    index = widget.backend_combo.findData("mjwarp")
    assert index >= 0
    widget.backend_combo.setCurrentIndex(index)
    # _refresh_capabilities_label reports the GPU backend as unavailable here.
    assert "unavailable" in widget.capabilities_label.text().lower()
    # run_rollout catches BackendNotAvailableError and surfaces it, never raises.
    widget.run_rollout()
    assert "failed" in widget.status_label.text().lower()


def test_dunder_main_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.tools.simulation_backends_launcher.__main__ as entry
    import src.tools.simulation_backends_launcher.gui as gui

    monkeypatch.setattr(gui, "main", lambda argv=None: 0)
    assert entry.main() == 0


def test_dunder_main_import_error_returns_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import src.tools.simulation_backends_launcher.__main__ as entry

    # Force the lazy GUI import to fail (simulating missing PyQt6/matplotlib).
    monkeypatch.setitem(sys.modules, "src.tools.simulation_backends_launcher.gui", None)
    assert entry.main() == 1
    assert "pip install" in capsys.readouterr().err
