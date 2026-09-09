"""Tests for the shared off-GUI-thread action mechanism (#8880).

Before this, every simulation in ``src/tools/`` ran inline in a ``clicked``
handler: no worker, no progress, no cancel, and the triggering button stayed
enabled so a second click queued a second run on top of the first.

These tests pin the mechanism's contract rather than any one tool's use of
it, because the mechanism is what the remaining migrations (tracked in the
follow-up issue) will depend on.
"""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
pytestmark = pytest.mark.unit

from PyQt6.QtCore import QCoreApplication, QThread  # noqa: E402

from src.tools.async_action import (  # noqa: E402
    AsyncActionBar,
    ProgressUpdate,
    WorkerCancelled,
    run_in_worker,
)


def _pump_until(predicate, timeout_s: float = 10.0) -> bool:  # noqa: ANN001
    """Spin the Qt event loop until ``predicate`` holds or the clock runs out."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        QThread.msleep(5)
    QCoreApplication.processEvents()
    return predicate()


# ----------------------------------------------------------------------
# ProgressUpdate contract
# ----------------------------------------------------------------------


def test_progress_update_rejects_a_fraction_outside_the_unit_interval() -> None:
    ProgressUpdate(0.0, "start")
    ProgressUpdate(1.0, "done")
    ProgressUpdate(None, "unknown length")
    with pytest.raises(ValueError, match=r"fraction must be in \[0, 1\]"):
        ProgressUpdate(1.5, "impossible")


# ----------------------------------------------------------------------
# run_in_worker
# ----------------------------------------------------------------------


def test_work_runs_off_the_gui_thread(qapp) -> None:  # noqa: ANN001
    """The whole point: the compute must not happen on the GUI thread."""
    from PyQt6.QtWidgets import QWidget

    parent = QWidget()
    gui_thread = QThread.currentThread()
    observed: list[object] = []
    results: list[object] = []

    def _work(ctx):  # noqa: ANN001, ANN202
        observed.append(QThread.currentThread())
        return 42

    handle = run_in_worker(
        parent,
        _work,
        on_finished=results.append,
        on_failed=lambda message: results.append(RuntimeError(message)),
    )
    assert _pump_until(lambda: bool(results)), "worker never reported back"
    handle.shutdown()

    assert results == [42]
    assert observed and observed[0] is not gui_thread


def test_failure_is_reported_not_raised(qapp) -> None:  # noqa: ANN001
    from PyQt6.QtWidgets import QWidget

    parent = QWidget()
    failures: list[str] = []

    def _work(ctx):  # noqa: ANN001, ANN202
        raise ValueError("bad input")

    handle = run_in_worker(
        parent,
        _work,
        on_finished=lambda _result: None,
        on_failed=failures.append,
    )
    assert _pump_until(lambda: bool(failures))
    handle.shutdown()

    assert failures == ["ValueError: bad input"]


def test_progress_reaches_the_gui_thread(qapp) -> None:  # noqa: ANN001
    from PyQt6.QtWidgets import QWidget

    parent = QWidget()
    updates: list[ProgressUpdate] = []
    done: list[object] = []

    def _work(ctx):  # noqa: ANN001, ANN202
        for index in range(4):
            ctx.report(index / 4, f"step {index}")
        return "ok"

    handle = run_in_worker(
        parent,
        _work,
        on_finished=done.append,
        on_failed=lambda message: done.append(message),
        on_progress=updates.append,
    )
    assert _pump_until(lambda: bool(done))
    handle.shutdown()

    assert done == ["ok"]
    assert [u.message for u in updates] == [f"step {i}" for i in range(4)]


def test_cancel_stops_the_work_at_the_next_checkpoint(qapp) -> None:  # noqa: ANN001
    """Cooperative cancel: the loop must actually stop, not just be ignored."""
    from PyQt6.QtWidgets import QWidget

    parent = QWidget()
    outcome: list[str] = []
    iterations: list[int] = []

    def _work(ctx):  # noqa: ANN001, ANN202
        for index in range(10_000):
            ctx.raise_if_cancelled()
            iterations.append(index)
            QThread.msleep(1)
        return "never reached"

    handle = run_in_worker(
        parent,
        _work,
        on_finished=lambda _r: outcome.append("finished"),
        on_failed=lambda _m: outcome.append("failed"),
        on_cancelled=lambda: outcome.append("cancelled"),
    )
    assert _pump_until(lambda: len(iterations) > 3), "worker never started"
    handle.request_cancel()
    assert _pump_until(lambda: bool(outcome)), "cancel was never honoured"
    handle.shutdown()

    assert outcome == ["cancelled"]
    assert len(iterations) < 10_000


def test_worker_cancelled_is_not_reported_as_a_failure(qapp) -> None:  # noqa: ANN001
    """A cancel is a normal outcome, not an error to show the user."""
    from PyQt6.QtWidgets import QWidget

    parent = QWidget()
    outcome: list[str] = []

    def _work(ctx):  # noqa: ANN001, ANN202
        raise WorkerCancelled

    handle = run_in_worker(
        parent,
        _work,
        on_finished=lambda _r: outcome.append("finished"),
        on_failed=lambda _m: outcome.append("failed"),
        on_cancelled=lambda: outcome.append("cancelled"),
    )
    assert _pump_until(lambda: bool(outcome))
    handle.shutdown()
    assert outcome == ["cancelled"]


def test_run_in_worker_validates_its_arguments(qapp) -> None:  # noqa: ANN001
    from PyQt6.QtWidgets import QWidget

    parent = QWidget()
    with pytest.raises(ValueError, match="parent must be provided"):
        run_in_worker(
            None,  # type: ignore[arg-type]
            lambda ctx: None,
            on_finished=lambda _r: None,
            on_failed=lambda _m: None,
        )
    with pytest.raises(TypeError, match="on_finished must be callable"):
        run_in_worker(
            parent,
            lambda ctx: None,
            on_finished="not callable",  # type: ignore[arg-type]
            on_failed=lambda _m: None,
        )


# ----------------------------------------------------------------------
# AsyncActionBar
# ----------------------------------------------------------------------


@pytest.fixture
def bar(qapp):  # noqa: ANN001, ANN201
    widget = AsyncActionBar()
    yield widget
    widget.shutdown()
    widget.deleteLater()


def test_bar_disables_trigger_buttons_for_the_duration(bar) -> None:  # noqa: ANN001
    """A second click must not be able to queue a second run."""
    from PyQt6.QtWidgets import QPushButton

    trigger = QPushButton("Run")
    bar.set_trigger_buttons(trigger)
    release: list[bool] = []
    done: list[object] = []

    def _work(ctx):  # noqa: ANN001, ANN202
        while not release:
            QThread.msleep(5)
        return "ok"

    assert bar.start("Sweep", _work, on_finished=done.append) is True
    assert _pump_until(lambda: not trigger.isEnabled())
    assert bar.cancel_button.isEnabled()

    # A second start while busy is refused rather than queued.
    assert bar.start("Sweep", _work, on_finished=done.append) is False

    release.append(True)
    assert _pump_until(lambda: bool(done))
    assert _pump_until(lambda: trigger.isEnabled())
    assert bar.cancel_button.isEnabled() is False
    assert done == ["ok"]


def test_bar_cancel_button_stops_the_work(bar) -> None:  # noqa: ANN001
    outcome: list[str] = []
    started: list[int] = []
    bar.cancelled.connect(lambda: outcome.append("cancelled"))

    def _work(ctx):  # noqa: ANN001, ANN202
        while True:
            ctx.raise_if_cancelled()
            started.append(1)
            QThread.msleep(1)

    bar.start("Sweep", _work, on_finished=lambda _r: outcome.append("finished"))
    assert _pump_until(lambda: len(started) > 3)
    bar.cancel_button.click()
    assert _pump_until(lambda: bool(outcome))
    assert outcome == ["cancelled"]
    assert "cancelled" in bar.status_label.text()


def test_bar_start_rejects_an_empty_name(bar) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        bar.start("", lambda ctx: None, on_finished=lambda _r: None)


def test_bar_shutdown_is_safe_when_idle(bar) -> None:  # noqa: ANN001
    assert bar.shutdown() is True
    assert bar.is_running is False
