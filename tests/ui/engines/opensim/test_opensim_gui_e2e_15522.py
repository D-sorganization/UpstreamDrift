"""Native headless QTest input drives the OpenSim Golf GUI (Refs #15522).

Real mouse events go through the real ``OpenSimGolfGUI`` window, buttons,
matplotlib canvas and status labels. Only the two process-external seams are
replaced: the OpenSim runtime probe (the library is not installable in CI) and
the blocking modal boundaries (``QFileDialog`` / ``QMessageBox``), which are
recorded rather than shown. The model injected for the run path is a
``GolfSwingModel`` stand-in returning the real ``SimulationResult`` dataclass.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
pytest.importorskip("matplotlib")

from PyQt6.QtCore import QTimer, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QDialog, QPushButton, QTextEdit  # noqa: E402

pytestmark = pytest.mark.unit

_GUI = "src.engines.physics_engines.opensim.python.opensim_gui"
_RUNTIME = "src.shared.python.engine_core.opensim_runtime"
_CORE = "src.engines.physics_engines.opensim.python.opensim_golf.core"


class _Dialogs:
    """Records modal-box calls instead of blocking the offscreen run."""

    def __init__(self) -> None:
        self.critical: list[tuple[str, str]] = []
        self.warning: list[tuple[str, str]] = []


class _FakeModel:
    """Stands in for ``GolfSwingModel`` and returns a real result object."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def run_simulation(self) -> Any:
        if self.error is not None:
            raise self.error
        core = importlib.import_module(_CORE)
        t = np.linspace(0.0, 0.5, 11)
        return core.SimulationResult(
            time=t,
            states=np.column_stack([np.sin(t), np.cos(t)]),
            muscle_forces=np.zeros((11, 1)),
            control_signals=np.zeros((11, 1)),
            joint_torques=np.column_stack([t, 2 * t]),
            marker_positions={},
        )


@pytest.fixture
def dialogs(monkeypatch: pytest.MonkeyPatch) -> _Dialogs:
    gui = importlib.import_module(_GUI)
    record = _Dialogs()
    monkeypatch.setattr(
        gui.QMessageBox,
        "critical",
        lambda _p, title, text, *a, **k: record.critical.append((title, text)),
    )
    monkeypatch.setattr(
        gui.QMessageBox,
        "warning",
        lambda _p, title, text, *a, **k: record.warning.append((title, text)),
    )
    return record


@pytest.fixture
def window(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Launch the real window with OpenSim reported as not installed."""
    gui = importlib.import_module(_GUI)
    runtime = importlib.import_module(_RUNTIME)

    def _missing(*_a: Any, **_k: Any) -> Any:
        raise ModuleNotFoundError("No module named 'opensim'")

    monkeypatch.setattr(gui, "probe_opensim", lambda: runtime.probe_opensim(_missing))
    win = gui.OpenSimGolfGUI(model_path=None)
    win.show()
    QTest.qWaitForWindowExposed(win)  # type: ignore[call-overload]  # PyQt6 6.9 stubs
    try:
        yield win
    finally:
        win._main_widget.cleanup()
        win.close()


def _click(button: QPushButton) -> None:
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)  # type: ignore[call-overload]  # PyQt6 6.9 stubs


def test_launch_reports_setup_required_with_run_disabled(window: Any) -> None:
    assert window.windowTitle() == "OpenSim Golf Interface"
    assert window.isVisible()
    assert "OpenSim Not Installed" in window.lbl_status.text()
    assert "conda install -c opensim-org opensim" in window.lbl_details.text()
    assert not window.btn_run.isEnabled()
    assert window.btn_load.isEnabled()
    assert window.btn_help.isEnabled()


def test_clicking_disabled_run_does_nothing(window: Any, dialogs: _Dialogs) -> None:
    _click(window.btn_run)
    assert window.result is None
    assert dialogs.warning == []
    assert dialogs.critical == []


def test_load_click_with_cancelled_dialog_keeps_state(
    window: Any, monkeypatch: pytest.MonkeyPatch, dialogs: _Dialogs
) -> None:
    gui = importlib.import_module(_GUI)
    asked: list[str] = []

    def _cancel(_parent: Any, caption: str, *_a: Any, **_k: Any) -> tuple[str, str]:
        asked.append(caption)
        return "", ""

    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName", _cancel)
    _click(window.btn_load)
    assert asked == ["Select OpenSim Model"]
    assert window.model_path is None
    assert window.model is None
    assert dialogs.critical == []


def test_load_click_without_opensim_reports_not_installed(
    window: Any, monkeypatch: pytest.MonkeyPatch, dialogs: _Dialogs, tmp_path: Any
) -> None:
    gui = importlib.import_module(_GUI)
    chosen = str(tmp_path / "golfer.osim")
    monkeypatch.setattr(
        gui.QFileDialog, "getOpenFileName", lambda *_a, **_k: (chosen, "")
    )
    _click(window.btn_load)
    assert window.model_path == chosen
    assert window.model is None
    assert "Status: ERROR" in window.lbl_status.text()
    assert not window.btn_run.isEnabled()
    assert [title for title, _ in dialogs.critical]
    assert window.initialization_error


def test_run_click_plots_results_and_reports_duration(
    window: Any, dialogs: _Dialogs
) -> None:
    window.model = _FakeModel()
    window.btn_run.setEnabled(True)
    _click(window.btn_run)
    assert window.result is not None
    assert "Simulation Complete. Duration: 0.50s, Steps: 11" in (
        window.lbl_details.text()
    )
    titles = [ax.get_title() for ax in window.fig.axes]
    assert "Joint Angles (rad)" in titles
    assert "Joint Torques (Nm)" in titles
    assert "Marker Trajectories (X-Y)" in titles
    assert window.btn_run.isEnabled()
    assert dialogs.critical == []


def test_run_click_surfaces_solver_failure_and_recovers(
    window: Any, dialogs: _Dialogs
) -> None:
    window.model = _FakeModel(error=RuntimeError("integrator diverged"))
    window.btn_run.setEnabled(True)
    _click(window.btn_run)
    assert window.result is None
    assert dialogs.critical == [("Simulation Error", "integrator diverged")]
    assert "Error occurred" in window.lbl_details.text()
    assert "Running" not in window.lbl_details.text()
    assert window.btn_run.isEnabled()


def test_help_click_opens_modal_guide_and_close_click_dismisses_it(
    window: Any, qapp: QApplication
) -> None:
    seen: dict[str, Any] = {}

    def _drive_dialog() -> None:
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, QDialog)
        seen["title"] = dialog.windowTitle()
        seen["text"] = dialog.findChild(QTextEdit).toPlainText()
        close = next(b for b in dialog.findChildren(QPushButton) if b.text() == "Close")
        _click(close)

    QTimer.singleShot(0, _drive_dialog)
    _click(window.btn_help)
    assert seen["title"] == "OpenSim Golf - Getting Started Guide"
    assert "OpenSim" in seen["text"]
    assert QApplication.activeModalWidget() is None


def test_keyboard_activation_of_help_button_opens_the_guide(
    window: Any, qapp: QApplication
) -> None:
    """Space on the focused button is the keyboard path to the same dialog."""
    opened: list[str] = []

    def _dismiss() -> None:
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, QDialog)
        opened.append(dialog.windowTitle())
        dialog.accept()

    window.btn_help.setFocus()
    QTimer.singleShot(0, _dismiss)
    QTest.keyClick(window.btn_help, Qt.Key.Key_Space)  # type: ignore[call-overload]  # PyQt6 6.9 stubs
    assert opened == ["OpenSim Golf - Getting Started Guide"]
