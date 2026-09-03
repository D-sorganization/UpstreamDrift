"""#8880 proof-of-migration: the sweep no longer blocks the GUI thread.

`simulation_backends_launcher` is the worst case the review found: `run_sweep`
loops `_SWEEP_SAMPLES = 24` full backend rollouts, each up to the horizon
maximum of 5000 steps -- up to 120,000 integration steps run inline in a
`clicked` handler, with the button still enabled, no progress, and no cancel.

These tests assert the three things the migration has to deliver: the compute
runs off the GUI thread, it honours a cancel, and the synchronous `run_*`
methods still behave exactly as before (they are what every other test in
this directory drives).
"""

from __future__ import annotations

import time

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QCoreApplication, QThread  # noqa: E402

from src.tools.async_action import WorkerCancelled  # noqa: E402


@pytest.fixture
def widget(qapp):  # noqa: ANN001, ANN201
    """Fresh MainWidget, with any running action cancelled on teardown."""
    from src.tools.simulation_backends_launcher.gui import MainWidget

    main_widget = MainWidget()
    yield main_widget
    main_widget.cleanup()
    main_widget.deleteLater()


def _pump_until(predicate, timeout_s: float = 20.0) -> bool:  # noqa: ANN001
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        QThread.msleep(5)
    QCoreApplication.processEvents()
    return predicate()


def test_sweep_button_runs_off_the_gui_thread(widget) -> None:  # noqa: ANN001
    """The headline defect: the sweep ran inline in the click handler."""
    gui_thread = QThread.currentThread()
    seen: list[object] = []
    original = widget._sweep_speeds

    def _record(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        seen.append(QThread.currentThread())
        return original(*args, **kwargs)

    widget._sweep_speeds = _record  # type: ignore[method-assign]
    widget.horizon_spin.setValue(20)

    widget.sweep_button.click()
    assert _pump_until(lambda: "complete" in widget.status_label.text().lower())

    assert seen, "the sweep never ran"
    assert seen[0] is not gui_thread, "the sweep still runs on the GUI thread"
    assert "Clubhead-mass sweep" in widget.report_text.toPlainText()
    widget.cleanup()


def test_trigger_buttons_are_disabled_while_an_action_runs(widget) -> None:  # noqa: ANN001
    """A second click used to queue a second sweep on top of the first."""
    release: list[bool] = []

    def _blocking(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        while not release:
            QThread.msleep(5)
        return [1.0] * 24

    widget._sweep_speeds = _blocking  # type: ignore[method-assign]

    widget.sweep_button.click()
    assert _pump_until(lambda: not widget.sweep_button.isEnabled())
    assert not widget.run_button.isEnabled()
    assert not widget.crossval_button.isEnabled()
    assert widget.action_bar.cancel_button.isEnabled()

    release.append(True)
    assert _pump_until(lambda: widget.sweep_button.isEnabled())
    widget.cleanup()


def test_cancel_stops_the_sweep(widget) -> None:  # noqa: ANN001
    """There was previously no way to stop a running sweep at all."""
    cancelled: list[int] = []
    widget.action_bar.cancelled.connect(lambda: cancelled.append(1))
    samples: list[int] = []
    original = widget._sweep_masses

    def _many(center: float):  # noqa: ANN202
        # A grid big enough that the cancel lands mid-sweep.
        import numpy as np

        return np.linspace(0.1, 0.5, 400)

    widget._sweep_masses = _many  # type: ignore[method-assign]
    real_speeds = widget._sweep_speeds

    def _counting(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        ctx = kwargs.get("ctx") or (args[4] if len(args) > 4 else None)
        assert ctx is not None, "the async path must pass a WorkerContext"
        for index in range(400):
            ctx.raise_if_cancelled()
            samples.append(index)
            QThread.msleep(2)
        return [1.0] * 400

    widget._sweep_speeds = _counting  # type: ignore[method-assign]
    del real_speeds

    widget.sweep_button.click()
    assert _pump_until(lambda: len(samples) > 3), "the sweep never started"
    widget.action_bar.cancel_button.click()
    assert _pump_until(lambda: bool(cancelled)), "cancel was never honoured"

    assert len(samples) < 400, "the sweep ran to completion despite the cancel"
    assert widget.sweep_button.isEnabled(), "buttons must be re-enabled on cancel"
    widget._sweep_masses = original  # type: ignore[method-assign]
    widget.cleanup()


def test_sweep_speeds_honours_a_cancelled_context(widget) -> None:  # noqa: ANN001
    """The cooperative checkpoint is inside the per-sample loop."""

    class _CancelledContext:
        def raise_if_cancelled(self) -> None:
            raise WorkerCancelled

        def report(self, fraction, message) -> None:  # noqa: ANN001
            return None

    with pytest.raises(WorkerCancelled):
        widget._sweep_speeds(
            "ode", widget._sweep_masses(0.2), 10, 0.005, _CancelledContext()
        )


def test_synchronous_run_methods_still_work(widget) -> None:  # noqa: ANN001
    """The sync core is unchanged; the async wrappers call into it."""
    widget.horizon_spin.setValue(20)
    widget.run_rollout()
    assert widget._last_trace is not None

    widget.run_sweep()
    assert "Clubhead-mass sweep" in widget.report_text.toPlainText()
