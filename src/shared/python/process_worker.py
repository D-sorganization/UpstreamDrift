"""Process Worker for running subprocesses in a separate thread.

This module provides a reusable QThread-based worker for executing shell commands,
capturing output in real-time, and handling process termination.
"""

import subprocess
import threading
from typing import Any

from src.shared.python.engine_availability import PYQT6_AVAILABLE

if PYQT6_AVAILABLE:
    from PyQt6.QtCore import QThread, pyqtSignal
else:
    # Fallback for headless environments or non-PyQt usage
    class QThread:  # type: ignore[no-redef]
        def __init__(self, parent: Any = None) -> None:
            pass

        def start(self) -> None:
            self.run()

        def run(self) -> None:
            pass

        def wait(self) -> None:
            pass

    class pyqtSignal:  # type: ignore[no-redef]
        def __init__(self, *args: Any) -> None:
            pass

        def emit(self, *args: Any) -> None:
            pass


class ProcessWorker(QThread):
    """Worker thread for running subprocesses without freezing the GUI."""

    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int, str)

    def __init__(
        self, cmd: list[str], cwd: str | None = None, env: dict[str, str] | None = None
    ) -> None:
        """Initialize worker.

        Args:
            cmd: Command list to execute.
            cwd: Working directory.
            env: Environment variables (optional).
        """
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd
        self.env = env
        self.process: subprocess.Popen | None = None
        self._is_running = True
        self._stop_event = threading.Event()

    def run(self) -> None:
        """Execute the command and stream output."""
        try:
            self.log_signal.emit(f"Running command: {' '.join(self.cmd)}")

            # Use current environment if explicit env is not provided, but merge if it is
            run_env = None
            if self.env:
                import os

                run_env = os.environ.copy()
                run_env.update(self.env)

            self.process = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Read stdout in real-time
            if self.process.stdout:
                for line in iter(self.process.stdout.readline, ""):
                    if self._stop_event.is_set():
                        break
                    if line:
                        self.log_signal.emit(line.strip())

            # Wait for completion
            stdout, stderr = self.process.communicate()

            # Process remaining output
            if stdout:
                for line in stdout.splitlines():
                    self.log_signal.emit(line.strip())

            if stderr:
                for line in stderr.splitlines():
                    self.log_signal.emit(f"STDERR: {line}")

            return_code = self.process.returncode
            self.finished_signal.emit(return_code, stderr if stderr else "")

        except Exception as e:
            self.log_signal.emit(f"Error starting process: {e}")
            self.finished_signal.emit(-1, str(e))
        finally:
            if self.process and self.process.poll() is None:
                try:
                    self.process.terminate()
                except Exception as e:
                    self.log_signal.emit(f"Error terminating process: {e}")

    def stop(self) -> None:
        """Stop the process."""
        self._is_running = False
        self._stop_event.set()
        if self.process:
            self.process.terminate()
