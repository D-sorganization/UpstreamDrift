"""Runs one ``rig`` CLI command at a time as a child process (QProcess)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PyQt6.QtCore import QProcess, QProcessEnvironment, pyqtSignal
from PyQt6.QtWidgets import QWidget

from src.shared.python.core.contracts import require

from . import commands


class RigProcessRunner(QWidget):
    """Streams the child's merged output; ``finished`` carries the exit code."""

    output = pyqtSignal(str)
    finished = pyqtSignal(int)
    busy_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._pending = False
        self._process.setWorkingDirectory(str(commands.repo_root()))
        env = QProcessEnvironment()
        for key, value in commands.child_environment().items():
            env.insert(key, value)
        self._process.setProcessEnvironment(env)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._drain)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)
        self._process.stateChanged.connect(
            lambda state: self.busy_changed.emit(
                state != QProcess.ProcessState.NotRunning
            )
        )

    @property
    def busy(self) -> bool:
        return self._process.state() != QProcess.ProcessState.NotRunning

    def run(self, argv: Sequence[str]) -> None:
        """Start ``argv``. Precondition: nothing is running."""
        require(not self.busy and not self._pending, "a rig command is already running")
        require(len(argv) >= 1, "argv must name a program")
        self.output.emit("$ " + " ".join(argv) + "\n")
        self._pending = True
        self._process.start(argv[0], list(argv[1:]))

    def stop(self) -> None:
        if self.busy:
            self._process.kill()

    def _drain(self) -> None:
        data = bytes(self._process.readAllStandardOutput().data())
        self.output.emit(data.decode("utf-8", errors="replace"))

    def _on_finished(self, code: int, _status: Any) -> None:
        if not self._pending:
            return
        self._pending = False
        self._drain()
        self.output.emit(f"[exit {code}]\n")
        self.finished.emit(int(code))

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart and self._pending:
            self.output.emit(
                f"Could not start {self._process.program()}: "
                f"{self._process.errorString()}. Check the executable and environment, then retry.\n"
            )
            # Qt does not emit finished for FailedToStart. Restore the same
            # command lifecycle the GUI uses for every other terminal result.
            self._on_finished(-1, None)
