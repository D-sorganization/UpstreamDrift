"""Headless QTest end-to-end flows for the Training Controller dashboard (#15522).

Real ``MainWindow``/``SubmitDialog`` widgets are driven with QTest mouse and
keyboard input against the real controller, scheduler, in-process driver and
job registry.  The only stand-in is the framework runner (no ML framework is
installed in unit lanes); it blocks until released or cancelled, so lifecycle
transitions are observed on a genuinely running job.  Only the modal
``QMessageBox`` boundary is intercepted.

Scope: dashboard interaction, validation feedback and lifecycle wiring.  This
does not certify training numerics or native rendering.  The window enforces a
1024x720 minimum, so the 390x844 case exercises the same layout clamped to its
minimum size.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTEST_QT_API", "pyqt6")
pytest.importorskip("pytestqt")
try:
    import PyQt6.QtCore as qtcore
    import PyQt6.QtTest  # noqa: F401
    import PyQt6.QtWidgets  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(f"PyQt6 not loadable: {exc}", allow_module_level=True)
if not isinstance(qtcore.PYQT_VERSION_STR, str):
    pytest.skip("Real PyQt6 required", allow_module_level=True)

from PyQt6.QtCore import QPoint, Qt, QTimer  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QMessageBox, QPushButton, QTableWidget  # noqa: E402
from pytestqt.qtbot import QtBot  # noqa: E402

from src.shared.python.training import (  # noqa: E402
    CompatibilityChecker,
    Dataset,
    DatasetRegistry,
    JobId,
    JobRegistry,
    RunResult,
    Scheduler,
    TrainingConfig,
    TrainingFramework,
    TrainingStatus,
    new_run_id,
)
from src.shared.python.training.contracts import (  # noqa: E402
    CancelToken,
    ProgressSink,
)
from src.shared.python.training.metrics import TrainingMetric  # noqa: E402
from src.shared.python.training.runtime import (  # noqa: E402
    InProcessDriver,
    RunnerRegistry,
)
from src.tools.training_controller import TrainingDashboardController  # noqa: E402
from src.tools.training_controller.gui import (  # noqa: E402
    MainWindow,
    SubmitDialog,
)

pytestmark = pytest.mark.unit

_STATUS_COLUMN = 2


class _HoldingRunner:
    """Real runner: stays RUNNING until released, then COMPLETED (or CANCELLED)."""

    framework = TrainingFramework.PYTORCH

    def __init__(self) -> None:
        self.release = threading.Event()

    def can_run(self, config: TrainingConfig) -> bool:
        return config.framework is self.framework

    def prepare(self, config: TrainingConfig) -> None:
        del config

    def run(
        self, config: TrainingConfig, *, progress: ProgressSink, cancel: CancelToken
    ) -> RunResult:
        del config, progress
        while not (self.release.is_set() or cancel.is_cancelled):
            self.release.wait(0.01)
        status = (
            TrainingStatus.CANCELLED
            if cancel.is_cancelled
            else TrainingStatus.COMPLETED
        )
        return RunResult(run_id=new_run_id(), status=status, duration_s=0.0)


class _Dashboard:
    """Bundle of the live pieces a test drives."""

    def __init__(self, window: MainWindow, scheduler: Scheduler, runner: Any) -> None:
        self.window = window
        self.widget = window.main_widget
        self.scheduler = scheduler
        self.runner = runner
        self.controller = window.controller


def _click(widget: Any, pos: QPoint | None = None) -> None:
    """Static Qt call, cast around incomplete PyQt6 stub signatures."""
    click = cast(Callable[..., None], QTest.mouseClick)
    if pos is None:
        click(widget, Qt.MouseButton.LeftButton)
    else:
        click(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)


def _type(edit: Any, text: str) -> None:
    """Real key events into a line edit (stub signature is incomplete)."""
    cast(Callable[[Any, str], None], QTest.keyClicks)(edit, text)


def _fill_and_submit(output: str) -> Callable[[SubmitDialog], None]:
    def _action(dialog: SubmitDialog) -> None:
        dialog.output_edit.setText(output)
        _click(dialog.submit_button)

    return _action


@pytest.fixture(params=[(1280, 900), (390, 844)], ids=["desktop", "phone"])
def dash(
    qtbot: QtBot, request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[_Dashboard]:
    """Show the real window over a real scheduler; always tear everything down."""
    runner = _HoldingRunner()
    runners = RunnerRegistry()
    runners.register(runner)
    scheduler = Scheduler(
        registry=JobRegistry(),
        runners=runners,
        driver=InProcessDriver(runners, max_workers=1),
    )
    datasets = DatasetRegistry(
        initial=(
            Dataset(
                dataset_id="dataset-1",
                name="Dataset 1",
                path=tmp_path / "dataset-1",
                format="custom",
            ),
        )
    )
    controller = TrainingDashboardController(
        scheduler, datasets, CompatibilityChecker()
    )
    window = MainWindow(controller)
    qtbot.addWidget(window)
    window.resize(*request.param)
    window.show()
    qtbot.waitUntil(window.isVisible)
    try:
        yield _Dashboard(window, scheduler, runner)
    finally:
        runner.release.set()
        window.close()
        controller.close()
        scheduler.shutdown()


def _drive_dialog(
    dash: _Dashboard,
    action: Callable[[SubmitDialog], None],
) -> dict[str, Any]:
    """Open the modal dialog via a real Submit click and run ``action`` inside it."""
    seen: dict[str, Any] = {}

    def _inside() -> None:
        # Closed dialogs stay parented to the window; pick the live one.
        live = [
            d
            for d in dash.window.findChildren(SubmitDialog, "training-submit-dialog")
            if d.isVisible()
        ]
        dialog = live[0] if live else None
        try:
            assert dialog is not None
            seen["dialog"] = dialog
            action(dialog)
            label = getattr(dialog, "error_label", None)
            seen["error"] = "" if label is None else label.text()
            seen["error_visible"] = label is not None and not label.isHidden()
            seen["still_open"] = dialog.isVisible()
        except Exception as exc:  # noqa: BLE001 - re-raised after exec() returns
            seen["exc"] = exc
        finally:
            if dialog is not None and dialog.isVisible():
                seen["entry_at_exit"] = dialog.entry_edit.text()
                dialog.reject()

    QTimer.singleShot(0, _inside)
    _click(dash.widget.submit_button)
    if "exc" in seen:
        raise seen["exc"]
    return seen


def _output(tmp: Path) -> str:
    return str(tmp / "out")


def _only_job(dash: _Dashboard) -> JobId:
    jobs = dash.controller.current_model().jobs
    assert len(jobs) == 1
    return JobId(jobs[0].job_id)


def _row_status(table: QTableWidget, row: int = 0) -> str:
    item = table.item(row, _STATUS_COLUMN)
    return item.text() if item is not None else ""


def _select_row_by_click(table: QTableWidget, row: int = 0) -> None:
    item = table.item(row, 0)
    assert item is not None
    _click(table.viewport(), table.visualItemRect(item).center())


def _press(button: QPushButton, qtbot: QtBot, status: str, table: QTableWidget) -> None:
    _click(button)
    qtbot.waitUntil(lambda: _row_status(table) == status, timeout=3000)


def test_submit_start_inspect_and_complete(
    dash: _Dashboard, qtbot: QtBot, tmp_path: Path
) -> None:
    """Dialog submit -> queued row -> real start -> inspect -> metrics -> done."""
    table = dash.widget.job_table
    seen = _drive_dialog(dash, _fill_and_submit(_output(tmp_path)))
    assert seen["still_open"] is False
    assert table.rowCount() == 1
    assert _row_status(table) == "queued"
    assert dash.widget.dataset_list.count() == 1

    job_id = _only_job(dash)
    dash.scheduler.start(job_id)
    qtbot.waitUntil(lambda: _row_status(table) == "running", timeout=3000)

    _select_row_by_click(table)
    assert dash.controller.selected_job_id == job_id
    summary = dash.widget.summary_label.text()
    assert job_id.value in summary and "RUNNING" in summary and "PYTORCH" in summary
    assert dash.widget.plot_stack.currentIndex() == 0

    for step in range(3):
        dash.controller.ingest_metric(
            job_id,
            TrainingMetric(
                name="loss", value=1.0 / (step + 1), step=step, timestamp=1.0
            ),
        )
    qtbot.waitUntil(lambda: dash.widget.plot_stack.currentIndex() == 1, timeout=3000)
    (axes,) = dash.widget.figure.axes
    assert len(axes.lines) >= 1 and len(axes.lines[0].get_xdata()) == 3

    dash.runner.release.set()
    dash.scheduler.collect(job_id, timeout=5)
    qtbot.waitUntil(lambda: _row_status(table) == "completed", timeout=3000)
    assert "COMPLETED" in dash.widget.summary_label.text()


def test_pause_resume_cancel_flow_on_running_job(
    dash: _Dashboard, qtbot: QtBot, tmp_path: Path
) -> None:
    """Real Pause/Resume/Cancel clicks; declining the cancel prompt changes nothing."""
    table, widget = dash.widget.job_table, dash.widget
    job = dash.controller.submit_job(
        TrainingConfig(
            framework=TrainingFramework.PYTORCH,
            entry_point="module:train",
            output_dir=tmp_path / "out",
            dataset_id="dataset-1",
        )
    )
    dash.scheduler.start(job.job_id)
    qtbot.waitUntil(lambda: _row_status(table) == "running", timeout=3000)
    assert not widget.pause_button.isEnabled()  # nothing selected yet
    _select_row_by_click(table)
    assert widget.pause_button.isEnabled() and widget.cancel_button.isEnabled()

    _press(widget.pause_button, qtbot, "paused", table)
    assert (
        widget.action_status_label.text()
        == f"Requested pause for job {job.job_id.value}."
    )
    _press(widget.resume_button, qtbot, "running", table)

    with patch.object(
        QMessageBox, "question", return_value=QMessageBox.StandardButton.Cancel
    ) as ask:
        _click(widget.cancel_button)
    ask.assert_called_once()
    assert _row_status(table) == "running"

    with patch.object(
        QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
    ):
        _press(widget.cancel_button, qtbot, "cancelled", table)
    assert "CANCELLED" in widget.summary_label.text()


def test_lifecycle_failure_on_terminal_job_is_reported_not_swallowed(
    dash: _Dashboard, qtbot: QtBot, tmp_path: Path
) -> None:
    """Pausing a finished job surfaces the scheduler refusal to the user."""
    table, widget = dash.widget.job_table, dash.widget
    job = dash.controller.submit_job(
        TrainingConfig(
            framework=TrainingFramework.PYTORCH,
            entry_point="module:train",
            output_dir=tmp_path / "out",
        )
    )
    dash.scheduler.start(job.job_id)
    dash.runner.release.set()
    dash.scheduler.collect(job.job_id, timeout=5)
    qtbot.waitUntil(lambda: _row_status(table) == "completed", timeout=3000)
    _select_row_by_click(table)
    with patch.object(QMessageBox, "warning") as warn:
        _click(widget.pause_button)
    warn.assert_called_once()
    assert "can only pause RUNNING" in widget.action_status_label.text()
    assert widget.action_status_label.property("statusRole") == "error"
    assert _row_status(table) == "completed"


def test_empty_entry_point_is_rejected_visibly_then_recovers(
    dash: _Dashboard, tmp_path: Path
) -> None:
    """Blank input must tell the user why nothing happened, and stay fixable."""

    def _blank_entry(dialog: SubmitDialog) -> None:
        dialog.entry_edit.clear()
        _fill_and_submit(_output(tmp_path))(dialog)

    seen = _drive_dialog(dash, _blank_entry)
    assert seen["still_open"] is True
    assert seen["error_visible"] and "entry point" in seen["error"].lower()
    assert dash.widget.job_table.rowCount() == 0

    def _blank_then_type_entry(dialog: SubmitDialog) -> None:
        _blank_entry(dialog)
        _type(dialog.entry_edit, "module:train")
        _click(dialog.submit_button)

    seen = _drive_dialog(dash, _blank_then_type_entry)
    assert seen["still_open"] is False
    assert dash.widget.job_table.rowCount() == 1


def test_framework_without_runner_is_reported_in_dialog(
    dash: _Dashboard, tmp_path: Path
) -> None:
    """The scheduler's NoRunnerAvailableError must reach the user, not only the log."""

    def _pick_gymnasium_then_submit(dialog: SubmitDialog) -> None:
        index = dialog.framework_combo.findData(TrainingFramework.GYMNASIUM)
        dialog.framework_combo.setCurrentIndex(index)
        dialog.output_edit.setText(_output(tmp_path))
        _click(dialog.submit_button)

    seen = _drive_dialog(dash, _pick_gymnasium_then_submit)
    assert seen["still_open"] is True
    assert seen["error_visible"] and "gymnasium" in seen["error"].lower()
    assert dash.widget.job_table.rowCount() == 0

    def _switch_back_and_submit(dialog: SubmitDialog) -> None:
        _pick_gymnasium_then_submit(dialog)
        dialog.framework_combo.setCurrentIndex(
            dialog.framework_combo.findData(TrainingFramework.PYTORCH)
        )
        _click(dialog.submit_button)

    seen = _drive_dialog(dash, _switch_back_and_submit)
    assert seen["still_open"] is False
    assert dash.widget.job_table.rowCount() == 1
    framework_item = dash.widget.job_table.item(0, 1)
    assert framework_item is not None and framework_item.text() == "pytorch"


def test_dialog_cancel_creates_no_job(dash: _Dashboard) -> None:
    """Cancel in the dialog closes it without queueing anything."""

    def _cancel(dialog: SubmitDialog) -> None:
        _click(dialog.cancel_btn)

    seen = _drive_dialog(dash, _cancel)
    assert seen["still_open"] is False
    assert dash.widget.job_table.rowCount() == 0


def test_closed_window_stops_following_the_controller(
    dash: _Dashboard, tmp_path: Path
) -> None:
    """After close, later scheduler events must not update or crash the widget."""
    dash.window.close()
    dash.controller.submit_job(
        TrainingConfig(
            framework=TrainingFramework.PYTORCH,
            entry_point="module:train",
            output_dir=tmp_path / "out",
        )
    )
    assert dash.controller.current_model().jobs  # controller saw the job ...
    assert dash.widget.job_table.rowCount() == 0  # ... the closed widget did not
