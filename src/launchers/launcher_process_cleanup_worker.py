"""Background process-cleanup worker for the UpstreamDrift launcher."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal


class ProcessCleanupWorkerSignals(QObject):
    """Signals emitted by ProcessCleanupWorker."""

    finished = pyqtSignal(list)


class ProcessCleanupWorker(QRunnable):
    """Poll running processes in a worker thread to avoid blocking the UI."""

    def __init__(self, running_processes: dict[str, Any], process_lock: Any) -> None:
        super().__init__()
        self.signals = ProcessCleanupWorkerSignals()
        self.running_processes = running_processes
        self.process_lock = process_lock

    def run(self) -> None:
        """Emit keys for processes that have finished."""
        finished_keys = []
        with self.process_lock:
            for key, proc in list(self.running_processes.items()):
                if proc.poll() is not None:
                    finished_keys.append(key)
        self.signals.finished.emit(finished_keys)
