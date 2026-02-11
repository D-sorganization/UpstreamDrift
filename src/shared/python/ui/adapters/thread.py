"""Thread Adapter for Framework-Agnostic Background Work.

This module provides an abstraction layer over threading mechanisms,
allowing code to work with or without Qt installed.

Classes:
    BackgroundWorker: Abstract base for background work
    QtWorker: PyQt6/PySide6 QThread implementation
    ThreadWorker: Standard threading implementation

Usage:
    worker = get_worker_adapter(target_function, args=(arg1, arg2))
    worker.on_complete(callback)
    worker.start()
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class BackgroundWorker(ABC):
    """Abstract base class for background workers.

    Provides a common interface for running tasks in the background
    that can be implemented by Qt threads or standard threading.
    """

    def __init__(
        self,
        target: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize worker with target function.

        Args:
            target: Function to run in background
            args: Positional arguments for target
            kwargs: Keyword arguments for target
        """
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self._result: Any = None
        self._error: Exception | None = None
        self._on_complete: Callable[[Any], None] | None = None
        self._on_error: Callable[[Exception], None] | None = None
        self._on_progress: Callable[[int, str], None] | None = None

    @abstractmethod
    def start(self) -> None:
        """Start the background task."""

    @abstractmethod
    def is_running(self) -> bool:
        """Check if task is still running."""

    @abstractmethod
    def wait(self, timeout: float | None = None) -> bool:
        """Wait for task to complete.

        Args:
            timeout: Maximum time to wait in seconds (None = infinite)

        Returns:
            True if task completed, False if timeout
        """

    @abstractmethod
    def cancel(self) -> None:
        """Request cancellation of the task."""

    def on_complete(self, callback: Callable[[Any], None]) -> BackgroundWorker:
        """Set callback for successful completion.

        Args:
            callback: Function called with result on success

        Returns:
            Self for chaining
        """
        self._on_complete = callback
        return self

    def on_error(self, callback: Callable[[Exception], None]) -> BackgroundWorker:
        """Set callback for errors.

        Args:
            callback: Function called with exception on error

        Returns:
            Self for chaining
        """
        self._on_error = callback
        return self

    def on_progress(self, callback: Callable[[int, str], None]) -> BackgroundWorker:
        """Set callback for progress updates.

        Args:
            callback: Function called with (percent, message)

        Returns:
            Self for chaining
        """
        self._on_progress = callback
        return self

    @property
    def result(self) -> Any:
        """Get the result of the task (after completion)."""
        return self._result

    @property
    def error(self) -> Exception | None:
        """Get any error that occurred."""
        return self._error

    def _invoke_complete(self) -> None:
        """Invoke completion callback if set."""
        if self._on_complete and self._error is None:
            try:
                self._on_complete(self._result)
            except (RuntimeError, ValueError, OSError) as e:
                logger.error(f"Error in completion callback: {e}")

    def _invoke_error(self) -> None:
        """Invoke error callback if set."""
        if self._on_error and self._error is not None:
            try:
                self._on_error(self._error)
            except (RuntimeError, ValueError, OSError) as e:
                logger.error(f"Error in error callback: {e}")


class ThreadWorker(BackgroundWorker):
    """Standard threading implementation of BackgroundWorker.

    Uses Python's threading module for background execution.
    Works in any environment without Qt dependency.
    """

    def __init__(
        self,
        target: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize thread worker."""
        super().__init__(target, args, kwargs)
        self._thread: threading.Thread | None = None
        self._cancelled = threading.Event()

    def start(self) -> None:
        """Start the background thread."""
        self._cancelled.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Execute the target function."""
        try:
            self._result = self.target(*self.args, **self.kwargs)
            self._invoke_complete()
        except (RuntimeError, ValueError, OSError) as e:
            self._error = e
            logger.error(f"Background task failed: {e}")
            self._invoke_error()

    def is_running(self) -> bool:
        """Check if thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for thread to complete."""
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def cancel(self) -> None:
        """Request cancellation (task must check _cancelled)."""
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancelled.is_set()


class QtWorker(BackgroundWorker):
    """PyQt6/PySide6 QThread implementation of BackgroundWorker.

    Uses Qt's threading for integration with Qt event loop.
    Signals are emitted on the main thread for safe UI updates.
    """

    def __init__(
        self,
        target: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Qt worker.

        Raises:
            RuntimeError: If Qt is not available
        """
        super().__init__(target, args, kwargs)

        try:
            from PyQt6.QtCore import QThread, pyqtSignal

            class WorkerThread(QThread):
                """Inner QThread implementation."""

                finished_signal = pyqtSignal(object)
                error_signal = pyqtSignal(Exception)
                progress_signal = pyqtSignal(int, str)

                def __init__(
                    self,
                    target: Callable[..., Any],
                    args: tuple[Any, ...],
                    kwargs: dict[str, Any],
                ) -> None:
                    super().__init__()
                    self._target = target
                    self._args = args
                    self._kwargs = kwargs
                    self._cancelled = False

                def run(self) -> None:
                    try:
                        # Check for cancellation before starting
                        if self._cancelled or self.isInterruptionRequested():
                            return
                        result = self._target(*self._args, **self._kwargs)
                        # Check for cancellation before emitting result
                        if not self._cancelled and not self.isInterruptionRequested():
                            self.finished_signal.emit(result)
                    except (RuntimeError, ValueError, OSError) as e:
                        if not self._cancelled and not self.isInterruptionRequested():
                            self.error_signal.emit(e)

                def cancel(self) -> None:
                    self._cancelled = True
                    self.requestInterruption()

                @property
                def is_cancelled(self) -> bool:
                    """Check if cancellation was requested."""
                    return self._cancelled or self.isInterruptionRequested()

            self._qthread = WorkerThread(target, args, kwargs or {})
            self._qthread.finished_signal.connect(self._handle_complete)
            self._qthread.error_signal.connect(self._handle_error)

        except ImportError as e:
            raise RuntimeError("Qt not available. Use ThreadWorker instead.") from e

    def _handle_complete(self, result: Any) -> None:
        """Handle completion signal from Qt thread."""
        self._result = result
        self._invoke_complete()

    def _handle_error(self, error: Exception) -> None:
        """Handle error signal from Qt thread."""
        self._error = error
        self._invoke_error()

    def start(self) -> None:
        """Start the Qt thread."""
        self._qthread.start()

    def is_running(self) -> bool:
        """Check if thread is running."""
        return self._qthread.isRunning()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for thread to complete."""
        if timeout is not None:
            return self._qthread.wait(int(timeout * 1000))
        return self._qthread.wait()

    def cancel(self) -> None:
        """Request thread cancellation."""
        self._qthread.cancel()


def is_qt_available() -> bool:
    """Check if Qt threading is available."""
    try:
        from PyQt6.QtCore import QThread  # noqa: F401

        return True
    except ImportError:
        return False


def get_worker_adapter(
    target: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    force_threading: bool = False,
) -> BackgroundWorker:
    """Get appropriate worker adapter for the current environment.

    Args:
        target: Function to run in background
        args: Positional arguments
        kwargs: Keyword arguments
        force_threading: Force standard threading

    Returns:
        Appropriate BackgroundWorker implementation
    """
    if force_threading or not is_qt_available():
        return ThreadWorker(target, args, kwargs)
    else:
        try:
            return QtWorker(target, args, kwargs)
        except RuntimeError:
            return ThreadWorker(target, args, kwargs)


__all__ = [
    "BackgroundWorker",
    "ThreadWorker",
    "QtWorker",
    "get_worker_adapter",
    "is_qt_available",
]
