"""Headless native Cross-Engine Dashboard click-through tests (Refs #15522).

Qt controls, the ``QThreadPool`` comparison worker, the ``pendulum_stub``
physics engine and the matplotlib canvases are all real. Only window-geometry
persistence is replaced so the tests never write user settings. No pytest-qt
is required.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtWidgets, sip
    from PyQt6.QtTest import QTest
except (ImportError, OSError) as exc:
    pytest.skip(f"Native PyQt6 is unavailable: {exc}", allow_module_level=True)

pytest.importorskip("matplotlib")

from src.launchers import cross_engine_dashboard as ced

pytestmark = pytest.mark.unit

_WORKER_TIMEOUT_MS = 30_000


@pytest.fixture
def window(monkeypatch: pytest.MonkeyPatch) -> Iterator[QtWidgets.QMainWindow]:
    """Launch the production dashboard window and always tear it down."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(ced, "persist_window_geometry", lambda *_a, **_k: None)
    widget = ced.CrossEngineDashboardWindow()
    widget.show()
    app.processEvents()
    assert widget.isVisible()
    try:
        yield widget
    finally:
        pool = QtCore.QThreadPool.globalInstance()
        pool.waitForDone(_WORKER_TIMEOUT_MS)
        if not sip.isdeleted(widget):
            widget.close()
            widget.deleteLater()
        app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()


def _click(button: QtWidgets.QAbstractButton) -> None:
    # Checkbox layouts stretch beyond their indicator/label hit region.
    position = (
        QtCore.QPoint(8, button.height() // 2)
        if isinstance(button, QtWidgets.QCheckBox)
        else button.rect().center()
    )
    QTest.mouseClick(button, QtCore.Qt.MouseButton.LeftButton, pos=position)


def _type_value(spin: QtWidgets.QAbstractSpinBox, text: str) -> None:
    """Replace a spin box value with real keystrokes and commit with Tab."""
    spin.setFocus()
    spin.selectAll()
    QTest.keyClicks(spin, text)
    QTest.keyClick(spin, QtCore.Qt.Key.Key_Tab)
    assert spin.value() == pytest.approx(float(text))


def _run_and_wait(window: QtWidgets.QMainWindow) -> None:
    """Click Run and pump the event loop until the worker re-enables it."""
    _click(window._run_btn)
    assert not window._run_btn.isEnabled()
    assert window._status_label.text() == "Running…"
    waited = 0
    while not window._run_btn.isEnabled() and waited < _WORKER_TIMEOUT_MS:
        QTest.qWait(25)
        waited += 25
    assert window._run_btn.isEnabled(), "comparison worker never completed"


def _small_run_config(window: QtWidgets.QMainWindow) -> None:
    _type_value(window._trials_spin, "3")
    _type_value(window._tend_spin, "0.5")


def test_launch_shows_documented_defaults(window) -> None:
    """A fresh launch exposes every engine, only the stub selected, and idle."""
    assert window.windowTitle() == "Cross-Engine Perturbation Comparison Dashboard"
    assert tuple(window._engine_checks) == tuple(ced._ENGINE_NAMES)
    checked = [n for n, cb in window._engine_checks.items() if cb.isChecked()]
    assert checked == ["pendulum_stub"]
    assert window._trials_spin.value() == 10
    assert window._amp_spin.value() == pytest.approx(0.1)
    assert window._tend_spin.value() == pytest.approx(1.5)
    assert window._dt_spin.value() == pytest.approx(0.01)
    assert window._run_btn.text() == "Run Comparison"
    assert window._run_btn.isEnabled()
    assert window._status_label.text() == "Ready"


def test_edit_run_populates_charts_and_overlay(window) -> None:
    """Typed config drives a real worker run that fills all three charts."""
    _small_run_config(window)
    _run_and_wait(window)

    assert window._status_label.text() == "Done"
    assert set(window._traj_handles) == {"pendulum_stub"}
    legend = window._ax_tr.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["pendulum_stub"]
    assert len(window._ax_tr.collections) == 1
    offsets = np.asarray(window._ax_tr.collections[0].get_offsets(), dtype=float)
    assert offsets.ndim == 2 and offsets.shape[0] > 2 and offsets.shape[1] == 2
    assert np.isfinite(offsets).all()
    assert len(window._ax_rs.patches) == 1
    assert 0.0 <= window._ax_rs.patches[0].get_height() <= 1.0
    assert len(window._ax_cv.patches) >= 1
    assert all(p.get_height() >= 0.0 for p in window._ax_cv.patches)


def test_rerun_with_second_engine_updates_overlay(window) -> None:
    """Ticking another engine and re-running replaces the previous results."""
    _small_run_config(window)
    _run_and_wait(window)
    first_handles = dict(window._traj_handles)
    assert set(first_handles) == {"pendulum_stub"}

    other = next(n for n in window._engine_checks if n != "pendulum_stub")
    _click(window._engine_checks[other])
    assert window._engine_checks[other].isChecked()
    _run_and_wait(window)

    assert set(window._traj_handles) == {"pendulum_stub", other}
    assert window._status_label.text().startswith("Done")
    assert len(window._ax_rs.patches) == 2


def test_run_without_engine_is_refused_and_recovers(window) -> None:
    """Unticking every engine blocks the run; re-ticking restores it."""
    _click(window._engine_checks["pendulum_stub"])
    assert not window._engine_checks["pendulum_stub"].isChecked()
    _click(window._run_btn)

    assert window._status_label.text() == "Select at least one engine"
    assert window._run_btn.isEnabled()
    assert not window._traj_handles

    _click(window._engine_checks["pendulum_stub"])
    _small_run_config(window)
    _run_and_wait(window)
    assert window._status_label.text() == "Done"


def test_invalid_config_reports_error_and_keeps_run_enabled(window) -> None:
    """t_end equal to dt violates the config contract and is surfaced in-UI."""
    _type_value(window._tend_spin, "0.1")
    _type_value(window._dt_spin, "0.1")
    _click(window._run_btn)

    assert window._status_label.text().startswith("Config error:")
    assert window._run_btn.isEnabled()
    assert not window._traj_handles

    _type_value(window._dt_spin, "0.01")
    _type_value(window._trials_spin, "2")
    _run_and_wait(window)
    assert window._status_label.text() == "Done"


def test_reopen_after_close_does_not_touch_deleted_window(window) -> None:
    """Closing the dashboard must not leave the theme singleton dangling."""
    app = QtWidgets.QApplication.instance()
    window.close()
    window.deleteLater()
    app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()

    second = ced.CrossEngineDashboardWindow()
    try:
        second.show()
        app.processEvents()
        assert second.isVisible()
        assert second._status_label.text() == "Ready"
    finally:
        second.close()
        second.deleteLater()
        app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()
