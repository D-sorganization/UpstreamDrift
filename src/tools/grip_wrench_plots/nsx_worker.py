"""QThreadPool worker for the NSX job (NSX-43, #12575).

The analysis never runs on the GUI thread.  :class:`NSXWorker` is a
``QRunnable`` whose results come back through queued signals; ``cancel()`` is
honoured between facade stages (``AnalysisCancelledError``).
"""

from __future__ import annotations

import threading

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from src.shared.python.loop_nullspace.facade import AnalysisCancelledError
from src.tools.grip_wrench_plots.nsx_job import NSXJob

__all__ = ["NSXWorker", "NSXWorkerSignals"]


class NSXWorkerSignals(QObject):
    """Signals of one run: ``progress(stage, percent)``, then exactly one of
    ``finished(model)``, ``failed(message)`` or ``cancelled()``."""

    progress = pyqtSignal(str, float)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()


class NSXWorker(QRunnable):
    """Runs an :data:`NSXJob` on a pool thread."""

    def __init__(self, job: NSXJob) -> None:
        super().__init__()
        if not callable(job):
            raise TypeError("job must be callable")
        self.setAutoDelete(False)
        self._job = job
        self._cancel = threading.Event()
        self.signals = NSXWorkerSignals()

    def cancel(self) -> None:
        """Ask the job to stop at its next stage boundary (idempotent)."""
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self) -> None:
        try:
            model = self._job(self._on_stage, self._cancel.is_set)
        except AnalysisCancelledError:
            self.signals.cancelled.emit()
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            if self._cancel.is_set():
                self.signals.cancelled.emit()
            else:
                self.signals.finished.emit(model)

    def _on_stage(self, stage: str, percent: float, _partial: object) -> None:
        self.signals.progress.emit(stage, float(percent))
