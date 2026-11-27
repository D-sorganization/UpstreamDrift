"""Native Qt input drives the standalone launcher and real ODE workers."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pytest

try:
    import matplotlib  # noqa: F401 - optional GUI dependency probe
    from PyQt6.QtCore import Qt, QThread
    from PyQt6.QtTest import QSignalSpy, QTest
    from PyQt6.QtWidgets import QApplication, QAbstractSpinBox
except (ImportError, OSError) as exc:
    pytest.skip(f"PyQt6/matplotlib not loadable: {exc}", allow_module_level=True)

from src.tools.simulation_backends_launcher.gui import SimulationBackendsWindow

pytestmark = pytest.mark.unit


@pytest.fixture
def window(qapp: QApplication) -> Iterator[SimulationBackendsWindow]:
    """Launch the production window; always cancel/join before Qt deletion."""
    launched = SimulationBackendsWindow()
    launched.show()
    launched.activateWindow()
    QTest.qWait(20)
    assert launched.isVisible()
    try:
        yield launched
    finally:
        launched.main_widget.cleanup()
        launched.close()
        launched.deleteLater()
        qapp.processEvents()


def _type_value(spin: QAbstractSpinBox, text: str) -> None:
    spin.setFocus()
    QTest.keyClick(spin, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(spin, text)
    QTest.keyClick(spin, Qt.Key.Key_Tab)


def _wait_until(predicate: Callable[[], bool]) -> None:
    """Pump real Qt events with a bounded wait, never a fixed solver sleep."""
    for _ in range(1500):
        if predicate():
            return
        QTest.qWait(10)
    assert predicate(), "Qt action did not reach its terminal state within 15 s"


def _click_and_finish(window: SimulationBackendsWindow, *, sweep: bool) -> None:
    widget = window.main_widget
    completed = QSignalSpy(widget.action_bar.finished)
    failed = QSignalSpy(widget.action_bar.failed)
    button = widget.sweep_button if sweep else widget.run_button
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert not widget.run_button.isEnabled()
    assert not widget.sweep_button.isEnabled()
    _wait_until(lambda: bool(completed) or bool(failed))
    assert not failed, widget.report_text.toPlainText()
    assert len(completed) == 1
    _wait_until(lambda: not widget.action_bar.is_running)
    assert widget.run_button.isEnabled()
    assert widget.sweep_button.isEnabled()


@pytest.mark.headless_safe
def test_rollout_keyboard_inputs_change_real_physics(
    window: SimulationBackendsWindow,
) -> None:
    widget = window.main_widget
    assert widget.current_backend_name() == "ode"
    assert "device=cpu" in widget.capabilities_label.text()
    _type_value(widget.horizon_spin, "20")
    _type_value(widget.dt_spin, "0.005")
    _type_value(widget.upper_mass_spin, "8.5")
    _type_value(widget.lower_clubhead_mass_spin, "0.3")
    assert widget.current_params().upper.mass_kg == pytest.approx(8.5)
    assert widget.current_params().lower.clubhead_mass_kg == pytest.approx(0.3)
    _click_and_finish(window, sweep=False)
    trace = widget._last_trace
    assert trace is not None
    assert trace.backend == "ode"
    assert trace.num_steps == 21
    assert trace.t[-1] == pytest.approx(0.1)
    assert np.isfinite(trace.q).all()
    assert not np.allclose(trace.q[-1], trace.q[0])
    assert len(widget.figure.axes[0].lines) == 2
    assert "Rollout complete: backend=ode" in widget.status_label.text()

    # With zero initial velocity, disabling gravity must leave the pose at rest.
    QTest.mouseClick(widget.gravity_check, Qt.MouseButton.LeftButton)
    assert not widget.gravity_check.isChecked()
    _click_and_finish(window, sweep=False)
    resting = widget._last_trace
    assert resting is not None and resting is not trace
    np.testing.assert_allclose(resting.v, 0.0, atol=1e-12)
    np.testing.assert_allclose(resting.q, np.tile(resting.q[0], (21, 1)))
    plotted = widget.figure.axes[0].lines[0]
    np.testing.assert_allclose(plotted.get_ydata(), resting.q[:, 0])


@pytest.mark.headless_safe
def test_sweep_button_populates_numeric_report_and_plot(
    window: SimulationBackendsWindow,
) -> None:
    widget = window.main_widget
    _type_value(widget.horizon_spin, "20")
    _type_value(widget.lower_clubhead_mass_spin, "0.4")
    _click_and_finish(window, sweep=True)
    report = widget.report_text.toPlainText()
    assert "Clubhead-mass sweep" in report
    assert "backend: ode" in report
    assert "samples: 24 (0.200..0.600 kg)" in report
    assert "horizon: 20 steps" in report
    line = widget.figure.axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), np.linspace(0.2, 0.6, 24))
    speeds = np.asarray(line.get_ydata())
    assert speeds.shape == (24,)
    assert np.isfinite(speeds).all() and (speeds > 0).all()
    assert np.ptp(speeds) > 0
    assert widget.status_label.text() == "Sweep complete: 24 clubhead masses on ode."


@pytest.mark.headless_safe
def test_close_stops_active_real_worker(
    window: SimulationBackendsWindow,
) -> None:
    widget = window.main_widget
    _type_value(widget.horizon_spin, "5000")
    QTest.mouseClick(widget.sweep_button, Qt.MouseButton.LeftButton)
    assert widget.action_bar.is_running
    threads = widget.action_bar.findChildren(QThread)
    assert any(thread.isRunning() for thread in threads)
    assert window.close()
    assert not window.isVisible()
    assert not widget.action_bar.is_running, "closing must cancel and join the worker"
    assert all(not thread.isRunning() for thread in threads)
