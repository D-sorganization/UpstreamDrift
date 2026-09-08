"""#9470 migration: launch-monitor analysis handlers leave the GUI thread.

Mirrors ``tests/ui/tools/simulation_backends/test_async_actions.py``: each run
button hands its compute to an :class:`AsyncActionBar` worker thread, every
trigger button is disabled while anything runs, Cancel shortens a looped
compute at its checkpoints, and ``cleanup()`` shuts the worker down. The
synchronous handlers remain present (``present(compute())``) and keep their
existing coverage in ``test_gui.py``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QCoreApplication, QThread  # noqa: E402

from src.tools.async_action import WorkerCancelled  # noqa: E402

FIXTURES = Path(__file__).parents[3] / "fixtures" / "launch_monitor"

#: Every button that starts an analysis; all must be disabled during a run.
_RUN_BUTTONS = (
    "run_treatment_button",
    "run_relationship_button",
    "run_multivariate_button",
    "run_model_button",
    "run_comparison_button",
    "run_dispersion_button",
    "run_trend_button",
)

_RELATIONSHIP_METRICS = frozenset({"club_speed", "ball_speed", "carry_distance"})


class _CancelAfterNChecks:
    """Test double for :class:`WorkerContext` that cancels after N checks."""

    def __init__(self, checks_allowed: int) -> None:
        self._remaining = checks_allowed
        self.reports: list[str] = []

    def raise_if_cancelled(self) -> None:
        if self._remaining <= 0:
            raise WorkerCancelled
        self._remaining -= 1

    def report(self, fraction: float | None, message: str) -> None:
        del fraction
        self.reports.append(message)


@pytest.fixture
def widget(qapp):  # noqa: ANN001, ANN201
    """Fresh MainWidget, with any running action shut down on teardown."""
    from src.tools.launch_monitor_analytics.gui import MainWidget

    main_widget = MainWidget()
    yield main_widget
    main_widget.cleanup()
    main_widget.deleteLater()


def _select_relationship_metrics(widget) -> None:  # noqa: ANN001
    for index in range(widget.relationship_metrics.count()):
        item = widget.relationship_metrics.item(index)
        item.setSelected(item.text() in _RELATIONSHIP_METRICS)


def _pump_until(predicate, timeout_s: float = 10.0) -> bool:  # noqa: ANN001
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        QThread.msleep(5)
    QCoreApplication.processEvents()
    return predicate()


def test_relationship_button_runs_off_the_gui_thread(widget) -> None:  # noqa: ANN001
    """The headline defect: the analysis ran inline in the click handler."""
    widget.import_file(FIXTURES / "trackman.csv")
    _select_relationship_metrics(widget)

    gui_thread = QThread.currentThread()
    seen: list[object] = []
    original = widget._compute_relationship

    def _record(params, ctx=None):  # noqa: ANN001, ANN002, ANN202
        seen.append(QThread.currentThread())
        return original(params, ctx)

    widget._compute_relationship = _record

    widget.run_relationship_button.click()
    assert _pump_until(
        lambda: "complete" in widget.action_bar.status_label.text().lower()
    )

    assert seen, "the relationship analysis never ran"
    assert seen[0] is not gui_thread, "the analysis still runs on the GUI thread"
    assert widget.relationship_table.rowCount() == 3


def test_trigger_buttons_are_disabled_while_an_action_runs(widget) -> None:  # noqa: ANN001
    """A second click used to queue a second analysis on top of the first."""
    widget.import_file(FIXTURES / "trackman.csv")
    _select_relationship_metrics(widget)

    release: list[bool] = []
    original = widget._compute_relationship

    def _blocking(params, ctx=None):  # noqa: ANN001, ANN002, ANN202
        while not release:
            ctx.raise_if_cancelled()
            QThread.msleep(5)
        return original(params, ctx)

    widget._compute_relationship = _blocking

    widget.run_relationship_button.click()
    assert _pump_until(lambda: not widget.run_relationship_button.isEnabled())
    for name in _RUN_BUTTONS:
        assert not getattr(widget, name).isEnabled(), f"{name} stayed enabled"
    assert widget.action_bar.cancel_button.isEnabled()

    release.append(True)
    assert _pump_until(lambda: widget.run_relationship_button.isEnabled())


def test_cancel_stops_a_running_analysis(widget) -> None:  # noqa: ANN001
    """There was previously no way to stop a running analysis at all."""
    cancelled: list[int] = []
    widget.action_bar.cancelled.connect(lambda: cancelled.append(1))
    checks: list[int] = []

    def _counting(frame, params, ctx=None):  # noqa: ANN001, ANN002, ANN202
        from src.tools.launch_monitor_analytics.gui import _DispersionData

        assert ctx is not None, "the async path must pass a WorkerContext"
        for index in range(200):
            ctx.raise_if_cancelled()
            checks.append(index)
            QThread.msleep(3)
        return _DispersionData(rows=[], series=())

    widget._compute_dispersion = _counting

    widget.run_dispersion_button.click()
    assert _pump_until(lambda: len(checks) > 3), "the analysis never started"
    widget.action_bar.cancel_button.click()
    assert _pump_until(lambda: bool(cancelled)), "cancel was never honoured"

    assert len(checks) < 200, "the analysis ran to completion despite the cancel"
    assert widget.run_dispersion_button.isEnabled(), (
        "buttons must be re-enabled on cancel"
    )


def test_dispersion_compute_honours_a_cancelled_context(widget) -> None:  # noqa: ANN001
    """The cooperative checkpoint sits inside the per-group loop."""
    from src.tools.launch_monitor_analytics.gui import _DispersionParams

    count = 9
    widget.analysis_frame = pd.DataFrame(
        {
            "session_id": [f"session-{index % 3}" for index in range(count)],
            "carry_distance": [100.0 + index for index in range(count)],
            "lateral_carry": [1.0 - index for index in range(count)],
        }
    )
    params = _DispersionParams(
        forward="carry_distance",
        lateral="lateral_carry",
        group_column="session_id",
    )
    context = _CancelAfterNChecks(checks_allowed=1)

    with pytest.raises(WorkerCancelled):
        widget._compute_dispersion(widget.analysis_frame, params, context)

    assert context.reports, "the compute never reported progress before cancelling"


def test_synchronous_run_methods_still_work(widget) -> None:  # noqa: ANN001
    """The sync core is unchanged; the async wrappers call into it."""
    widget.import_file(FIXTURES / "trackman.csv")
    _select_relationship_metrics(widget)
    result = widget.run_relationship_analysis()
    assert result.coefficients.shape == (3, 3)
    assert widget.relationship_table.rowCount() == 3


def test_cleanup_shuts_down_a_running_action(widget) -> None:  # noqa: ANN001
    """Closing the tab must cancel and join the worker, not leak the thread."""
    widget.import_file(FIXTURES / "trackman.csv")
    _select_relationship_metrics(widget)

    def _blocking(params, ctx=None):  # noqa: ANN001, ANN002, ANN202
        while True:
            ctx.raise_if_cancelled()
            QThread.msleep(5)

    widget._compute_relationship = _blocking

    widget.run_relationship_button.click()
    assert _pump_until(lambda: widget.action_bar.is_running)
    widget.cleanup()
    assert not widget.action_bar.is_running
