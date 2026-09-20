"""Bounded splash <-> worker <-> shell handshake for the launcher (issue #8360).

:class:`StartupSession` owns the lifetime of the splash screen:

* the async worker's progress/finished/error signals are routed to the
  splash and the main shell only while their C++ objects are alive and only
  for the *current* worker generation (a retried startup ignores late
  callbacks from the previous worker);
* a watchdog bounded by ``STARTUP_TIMEOUT_SEC`` guarantees the splash
  transitions to an actionable state even if the worker never reports;
* required-phase failures and watchdog timeouts open
  :class:`StartupFailureDialog` (Retry / Continue without provider / Copy
  diagnostics / Close) instead of quitting the application;
* optional-provider degradation never blocks the shell: the shell opens and
  a warning toast names the degraded phase.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeGuard, TypeVar, cast

from PyQt6 import sip
from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtWidgets import QApplication

from src.launchers.launcher_constants import STARTUP_TIMEOUT_SEC
from src.launchers.startup import AsyncStartupWorker, SplashScreen, StartupResults
from src.launchers.startup_failure_dialog import (
    StartupFailureAction,
    StartupFailureDialog,
)

logger = logging.getLogger(__name__)

__all__ = ["StartupSession", "is_alive"]

# How long a retired worker thread may take to wind down before we stop
# waiting on it. Phases are individually bounded, so this only covers the
# hand-off between the last phase settling and ``run()`` returning.
_WORKER_WAIT_MS = 1000

DialogFactory = Callable[..., StartupFailureDialog]
T = TypeVar("T")


def is_alive(obj: T | None) -> TypeGuard[T]:
    """True when ``obj`` is a live object whose C++ side (if any) still exists."""
    if obj is None:
        return False
    try:
        return not sip.isdeleted(cast("Any", obj))
    except TypeError:
        # Not a sip wrapper (e.g. a test double): nothing can have deleted it.
        return True


class StartupSession(QObject):
    """Drive one launcher startup with a bounded splash lifetime.

    Invariants:
        * At most one worker generation is current; callbacks from older
          generations are ignored.
        * Within ``timeout_s`` of :meth:`start` the session has settled:
          the shell opened (possibly degraded) or the failure dialog ran.
        * No callback touches a splash or shell whose C++ object is gone.
    """

    def __init__(
        self,
        *,
        splash: SplashScreen | None,
        shell: Any,
        worker_factory: Callable[[], AsyncStartupWorker],
        timeout_s: float = STARTUP_TIMEOUT_SEC,
        dialog_factory: DialogFactory = StartupFailureDialog,
        quit_callback: Callable[[], None] | None = None,
        parent: QObject | None = None,
    ) -> None:
        if shell is None:
            raise ValueError("shell must be provided")
        if not callable(worker_factory):
            raise TypeError("worker_factory must be callable")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        super().__init__(parent)
        self._splash = splash
        self._shell = shell
        self._worker_factory = worker_factory
        self._timeout_s = float(timeout_s)
        self._dialog_factory = dialog_factory
        self._quit = quit_callback or QApplication.quit
        self._generation = 0
        self._settled = False
        self._worker: AsyncStartupWorker | None = None
        self._retired: list[AsyncStartupWorker] = []
        self._last_phase = "not started"
        self._watchdog = QTimer(self)
        self._watchdog.setSingleShot(True)
        self._watchdog.timeout.connect(self._on_watchdog)

    # -- public API -------------------------------------------------------------

    @property
    def settled(self) -> bool:
        """True once the current generation opened the shell or ran the dialog."""
        return self._settled

    @property
    def generation(self) -> int:
        """Number of worker generations started so far."""
        return self._generation

    @property
    def worker(self) -> AsyncStartupWorker | None:
        """The current worker (``None`` before :meth:`start`)."""
        return self._worker

    def start(self) -> None:
        """Start (or, on retry, restart) a worker generation and arm the watchdog."""
        self._retire_current_worker()
        self._generation += 1
        self._settled = False
        generation = self._generation
        worker = self._worker_factory()
        self._worker = worker
        worker.progress_signal.connect(
            lambda msg, pct: self._on_progress(generation, msg, pct)
        )
        worker.finished_signal.connect(
            lambda results: self._on_finished(generation, results)
        )
        worker.error_signal.connect(lambda msg: self._on_error(generation, msg))
        if is_alive(self._splash) and not self._splash.isVisible():
            self._splash.show()
        self._watchdog.start(int(self._timeout_s * 1000))
        worker.start()

    def shutdown(self) -> None:
        """Stop the watchdog and wait briefly for worker threads to wind down."""
        self._watchdog.stop()
        self._retire_current_worker()
        for worker in list(self._retired):
            worker.wait(_WORKER_WAIT_MS)
        self._retired = [w for w in self._retired if w.isRunning()]

    # -- worker callbacks (GUI thread) ----------------------------------------------

    def _is_current(self, generation: int) -> bool:
        return generation == self._generation and not self._settled

    def _on_progress(self, generation: int, message: str, percent: int) -> None:
        if not self._is_current(generation):
            return
        self._last_phase = message
        logger.info("Startup progress: %s%% - %s", percent, message)
        if is_alive(self._splash):
            self._splash.show_message(message, percent)

    def _on_finished(self, generation: int, results: StartupResults) -> None:
        if not self._is_current(generation):
            return
        self._settle()
        self._open_shell(results)
        if results.is_degraded:
            summary = results.degraded_summary()
            logger.warning("%s\n%s", summary, results.format_diagnostics())
            self._toast(summary, "warning")

    def _on_error(self, generation: int, message: str) -> None:
        if not self._is_current(generation):
            return
        self._settle()
        logger.error("Startup failed: %s", message)
        self._present_failure(message)

    def _on_watchdog(self) -> None:
        if self._settled:
            return
        self._settle()
        message = (
            f"Startup did not complete within {self._timeout_s:g}s; "
            f"last reported phase: {self._last_phase}"
        )
        logger.error("%s\n%s", message, self._diagnostics())
        self._present_failure(message)

    # -- transitions ----------------------------------------------------------------

    def _settle(self) -> None:
        self._settled = True
        self._watchdog.stop()

    def _diagnostics(self) -> str:
        worker = self._worker
        return worker.diagnostics_text() if worker is not None else ""

    def _shell_alive(self) -> bool:
        # The shell is typed ``Any`` (launcher or test double); a bool
        # accessor keeps mypy from narrowing it to ``Never``.
        return is_alive(self._shell)

    def _open_shell(self, results: StartupResults) -> None:
        """Apply results to the shell and dismiss the splash, degrading on error."""
        if self._shell_alive():
            try:
                self._shell.update_startup_results(results)
            except (RuntimeError, ValueError, TypeError, OSError):
                logger.exception(
                    "Applying startup results failed; opening shell degraded"
                )
                self._shell.loading = False
                self._toast(
                    "Startup results could not be applied; UpstreamDrift is "
                    "running without models. See the log for diagnostics.",
                    "error",
                )
        self._finish_splash()

    def _finish_splash(self) -> None:
        if not is_alive(self._splash):
            return
        if self._shell_alive():
            self._splash.finish(self._shell)
        else:
            self._splash.close()

    def _toast(self, message: str, kind: str) -> None:
        if self._shell_alive() and hasattr(self._shell, "show_toast"):
            self._shell.show_toast(message, kind)

    def _present_failure(self, message: str) -> None:
        """Show the actionable failure state and apply the user's decision."""
        if is_alive(self._splash):
            self._splash.show_degraded(message)
            self._splash.hide()
        diagnostics = self._diagnostics()
        parent = self._shell if self._shell_alive() else None
        dialog = self._dialog_factory(message, diagnostics, parent=parent)
        action = dialog.ask()
        logger.info("Startup failure action: %s", action.value)
        if action == StartupFailureAction.RETRY:
            self.start()
        elif action == StartupFailureAction.CONTINUE:
            self._continue_degraded()
        else:
            self.shutdown()
            self._quit()

    def _continue_degraded(self) -> None:
        """Open the shell with whatever the worker managed to produce."""
        worker = self._worker
        results = worker.results if worker is not None else StartupResults()
        if not results.phases and worker is not None:
            results.phases = worker.timeline.records
            results.startup_time_ms = worker.timeline.elapsed_ms()
        self._open_shell(results)
        self._toast(
            "UpstreamDrift opened in degraded mode after a startup problem.",
            "warning",
        )

    def _retire_current_worker(self) -> None:
        worker = self._worker
        if worker is None:
            return
        self._worker = None
        worker.requestInterruption()
        # Keep a reference until the thread really stops: destroying a
        # running QThread aborts the process with a Qt teardown message.
        if worker.isRunning() and not worker.wait(_WORKER_WAIT_MS):
            self._retired.append(worker)
            worker.finished.connect(lambda: self._forget_retired(worker))

    def _forget_retired(self, worker: AsyncStartupWorker) -> None:
        if worker in self._retired:
            self._retired.remove(worker)
