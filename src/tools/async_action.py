"""Run a long computation off the GUI thread, with progress and cancel.

Issue #8880. Across all of ``src/tools/`` exactly one file used ``QThread``
(``starting_pose_matcher/widgets/run_fit_button.py``) and exactly two ever
set a wait cursor. Everything else did heavy compute inline in a ``clicked``
handler: the window stopped repainting, the OS marked it "Not Responding",
the triggering button stayed enabled so a second click queued a second run,
and there was no way to stop it.

This module generalises the one good pattern that already existed -- worker
``QObject`` moved onto a ``QThread``, with ``progress``/``finished``/
``failed``/``cancelled`` signals and a cooperative cancel flag -- into
something any tool can reuse:

* :func:`run_in_worker` -- the mechanism. Hand it a callable taking a
  :class:`WorkerContext`; get back a :class:`WorkerHandle` you can cancel.
* :class:`AsyncActionBar` -- the UI half: a progress bar, a Cancel button,
  and a status line, which knows how to disable the buttons that trigger
  the work while it runs.

**Cancellation is cooperative.** There is no safe way to kill a running
Python callable, so the work must call ``ctx.raise_if_cancelled()`` (or
check ``ctx.is_cancelled``) at a loop boundary. Work that cannot check --
a single opaque C call -- can still be run here to keep the GUI painting,
but Cancel will only take effect when it returns; say so in the tool's UI
rather than offering a Cancel that appears to do nothing.

**Threading contract.** The work callable runs on a worker thread and must
not touch Qt widgets. Present its result in the ``on_finished`` callback,
which is invoked on the GUI thread via the signal connection.

Placement note: the natural home is ``src/shared/python/ui/``, but that tree
is mid-retirement (seam epic #9406), so this lives under ``src/tools/``
alongside its consumers until that settles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "DEFAULT_JOIN_TIMEOUT_MS",
    "AsyncActionBar",
    "ProgressUpdate",
    "WorkerCancelled",
    "WorkerContext",
    "WorkerHandle",
    "run_in_worker",
]

#: How long a cancel waits for the worker thread to unwind before giving up
#: and letting it finish in the background. Bounded so the GUI thread is
#: never blocked indefinitely by a cancel (the mistake #8895 fixed in the
#: Docker close guard).
DEFAULT_JOIN_TIMEOUT_MS: Final[int] = 3000


class WorkerCancelled(Exception):
    """Raised inside worker code when the user has requested cancellation."""


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    """One progress report from a worker.

    Attributes:
        fraction: Completion in ``[0.0, 1.0]``, or ``None`` when the total
            amount of work is unknown (the bar goes busy/indeterminate).
        message: Short human-readable status, e.g. ``"sample 7/24"``.
    """

    fraction: float | None
    message: str

    def __post_init__(self) -> None:
        if self.fraction is not None and not 0.0 <= self.fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {self.fraction}")


class WorkerContext:
    """Handed to the work callable: cancellation checks and progress reports.

    Both methods are safe to call from the worker thread; ``report`` emits a
    Qt signal, which is queued to the GUI thread by the connection type.
    """

    __slots__ = ("_cancelled", "_worker")

    def __init__(self, worker: _CallableWorker) -> None:
        self._worker = worker
        self._cancelled = False

    def request_cancel(self) -> None:
        """Set the cooperative cancel flag (called from the GUI thread)."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        """Raise :class:`WorkerCancelled` if cancellation was requested.

        Call this at every loop boundary in the work callable. This is the
        only thing that makes Cancel actually cancel.
        """
        if self._cancelled:
            raise WorkerCancelled

    def report(self, fraction: float | None, message: str) -> None:
        """Emit a progress update to the GUI thread."""
        self._worker.progress.emit(ProgressUpdate(fraction, message))


class _CallableWorker(QObject):
    """Runs one callable on a worker thread and reports the outcome."""

    progress = pyqtSignal(object)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, work: Callable[[WorkerContext], Any]) -> None:
        super().__init__()
        if not callable(work):
            raise TypeError("work must be callable")
        self._work = work
        self.context = WorkerContext(self)

    def request_cancel(self) -> None:
        """Set the worker's cooperative cancel flag.

        A delegating method rather than letting callers reach through to
        ``worker.context.request_cancel()`` (Law of Demeter).
        """
        self.context.request_cancel()

    def run(self) -> None:
        """Invoke the work callable, translating its outcome into signals."""
        if self.context.is_cancelled:
            self.cancelled.emit()
            return
        try:
            result = self._work(self.context)
        except WorkerCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001 - the worker must never crash the app
            logger.exception("Background action failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        if self.context.is_cancelled:
            self.cancelled.emit()
            return
        self.finished.emit(result)


class WorkerHandle:
    """Owns a running worker + thread pair and can cancel or join them."""

    __slots__ = ("_thread", "_worker")

    def __init__(self, worker: _CallableWorker, thread: QThread) -> None:
        self._worker: _CallableWorker | None = worker
        self._thread: QThread | None = thread

    @property
    def is_running(self) -> bool:
        """Whether the worker thread is still alive."""
        thread = self._thread
        return thread is not None and thread.isRunning()

    def request_cancel(self) -> None:
        """Ask the work to stop at its next cancellation check."""
        if self._worker is not None:
            self._worker.request_cancel()
        if self._thread is not None:
            self._thread.requestInterruption()

    def wait(self, timeout_ms: int = DEFAULT_JOIN_TIMEOUT_MS) -> bool:
        """Join the thread, bounded by ``timeout_ms``."""
        thread = self._thread
        if thread is None:
            return True
        thread.quit()
        return bool(thread.wait(timeout_ms))

    def shutdown(self, timeout_ms: int = DEFAULT_JOIN_TIMEOUT_MS) -> bool:
        """Cancel and join. Returns False if the thread outlived the join."""
        self.request_cancel()
        stopped = self.wait(timeout_ms)
        self._worker = None
        self._thread = None
        return stopped


def run_in_worker(
    parent: QObject,
    work: Callable[[WorkerContext], Any],
    *,
    on_finished: Callable[[Any], None],
    on_failed: Callable[[str], None],
    on_cancelled: Callable[[], None] | None = None,
    on_progress: Callable[[ProgressUpdate], None] | None = None,
) -> WorkerHandle:
    """Run ``work`` on a background thread and report back on the GUI thread.

    Args:
        parent: Qt parent for the thread, so it dies with its owner.
        work: Callable taking a :class:`WorkerContext`. Runs off the GUI
            thread and must not touch widgets.
        on_finished: Receives the callable's return value.
        on_failed: Receives a ``"TypeName: message"`` string.
        on_cancelled: Called when the work honoured a cancel request.
        on_progress: Receives each :class:`ProgressUpdate`.

    Returns:
        A :class:`WorkerHandle`. Keep a reference: dropping it lets the
        thread be garbage-collected mid-run.
    """
    if parent is None:
        raise ValueError("parent must be provided")
    if not callable(on_finished):
        raise TypeError("on_finished must be callable")
    if not callable(on_failed):
        raise TypeError("on_failed must be callable")

    worker = _CallableWorker(work)
    thread = QThread(parent)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(on_finished)
    worker.failed.connect(on_failed)
    if on_cancelled is not None:
        worker.cancelled.connect(on_cancelled)
    if on_progress is not None:
        worker.progress.connect(on_progress)
    for signal in (worker.finished, worker.failed, worker.cancelled):
        signal.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.start()
    return WorkerHandle(worker, thread)


class AsyncActionBar(QWidget):
    """Progress bar + Cancel button + status line for one background action.

    One bar serves a whole tool: it disables the buttons that trigger work
    while any of them is running, so a second click cannot queue a second
    run, and re-enables them on every terminal outcome.
    """

    started = pyqtSignal(str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._handle: WorkerHandle | None = None
        self._trigger_buttons: tuple[QPushButton, ...] = ()
        self._action_name = ""

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.setToolTip("Stop the running action at its next checkpoint.")
        self.cancel_button.clicked.connect(self.cancel)
        self.status_label = QLabel("Idle.")
        self.status_label.setWordWrap(True)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.progress_bar, stretch=1)
        row.addWidget(self.cancel_button)
        row.addWidget(self.status_label, stretch=2)

    # ---- configuration -------------------------------------------------

    def set_trigger_buttons(self, *buttons: QPushButton) -> None:
        """Buttons to disable for the duration of any run."""
        self._trigger_buttons = tuple(buttons)

    @property
    def is_running(self) -> bool:
        """Whether an action is currently running."""
        return self._handle is not None and self._handle.is_running

    # ---- driving ---------------------------------------------------------

    def start(
        self,
        name: str,
        work: Callable[[WorkerContext], Any],
        *,
        on_finished: Callable[[Any], None],
        on_failed: Callable[[str], None] | None = None,
    ) -> bool:
        """Start ``work``, refusing if an action is already running.

        Returns True when the action was started.
        """
        if not name:
            raise ValueError("name must be a non-empty string")
        if self.is_running:
            self.status_label.setText(
                f"{self._action_name} is still running; cancel it first."
            )
            return False

        self._action_name = name
        self._set_busy(True)
        self.status_label.setText(f"{name}: starting...")
        self.started.emit(name)

        def _finished(result: Any) -> None:
            self._set_busy(False)
            self.status_label.setText(f"{name}: complete.")
            on_finished(result)
            self.finished.emit(result)

        def _failed(message: str) -> None:
            self._set_busy(False)
            self.status_label.setText(f"{name} failed: {message}")
            if on_failed is not None:
                on_failed(message)
            self.failed.emit(message)

        def _cancelled() -> None:
            self._set_busy(False)
            self.status_label.setText(f"{name}: cancelled.")
            self.cancelled.emit()

        self._handle = run_in_worker(
            self,
            work,
            on_finished=_finished,
            on_failed=_failed,
            on_cancelled=_cancelled,
            on_progress=self._on_progress,
        )
        return True

    def cancel(self) -> None:
        """Request cancellation of the running action."""
        if self._handle is None:
            return
        self.status_label.setText(f"{self._action_name}: cancelling...")
        self.cancel_button.setEnabled(False)
        self._handle.request_cancel()

    def shutdown(self) -> bool:
        """Cancel and join any running action. Call from ``cleanup``."""
        if self._handle is None:
            return True
        stopped = self._handle.shutdown()
        self._handle = None
        self._set_busy(False)
        return stopped

    # ---- internals -------------------------------------------------------

    def _on_progress(self, update: ProgressUpdate) -> None:
        if update.fraction is None:
            self.progress_bar.setRange(0, 0)  # busy indicator
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(round(update.fraction * 100)))
        self.status_label.setText(f"{self._action_name}: {update.message}")

    def _set_busy(self, busy: bool) -> None:
        self.progress_bar.setVisible(busy)
        if busy:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
        self.cancel_button.setEnabled(busy)
        for button in self._trigger_buttons:
            button.setEnabled(not busy)
