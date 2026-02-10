"""Process Worker for running subprocesses in a separate thread.

This module provides a reusable QThread-based worker for executing shell commands,
capturing output in real-time, and handling process termination.
"""

import logging
import subprocess
import threading
from typing import Any

from src.shared.python.engine_availability import PYQT6_AVAILABLE

logger = logging.getLogger(__name__)

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
            stderr_output = ""
            if self.process.stdout:
                for line in iter(self.process.stdout.readline, ""):
                    if self._stop_event.is_set():
                        break
                    if line:
                        self.log_signal.emit(line.strip())

            # Read any remaining stderr (don't use communicate() after manual stdout read)
            if self.process.stderr:
                stderr_output = self.process.stderr.read()
                if stderr_output:
                    for line in stderr_output.splitlines():
                        self.log_signal.emit(f"STDERR: {line}")

            # Wait for process to complete
            self.process.wait()

            return_code = self.process.returncode
            self.finished_signal.emit(return_code, stderr_output)

        except (OSError, subprocess.SubprocessError, ValueError) as e:
            self.log_signal.emit(f"Error starting process: {e}")
            self.finished_signal.emit(-1, str(e))
        finally:
            if self.process:
                # Ensure process is terminated if still running
                if self.process.poll() is None:
                    try:
                        self.process.terminate()
                        self.process.wait(timeout=5)
                    except (RuntimeError, ValueError, OSError) as e:
                        self.log_signal.emit(f"Error terminating process: {e}")
                        try:
                            self.process.kill()
                        except (RuntimeError, ValueError, OSError) as kill_err:
                            logger.debug("Failed to kill process: %s", kill_err)
                # Close file handles to prevent resource leaks
                try:
                    if self.process.stdout:
                        self.process.stdout.close()
                    if self.process.stderr:
                        self.process.stderr.close()
                except (RuntimeError, ValueError, OSError) as close_err:
                    logger.debug("Failed to close process handles: %s", close_err)

    def stop(self) -> None:
        """Stop the process."""
        self._is_running = False
        self._stop_event.set()
        if self.process:
            self.process.terminate()
