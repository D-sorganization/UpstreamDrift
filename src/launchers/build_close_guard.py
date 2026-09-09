"""Shared close-event handling for active launcher builds.

Issue #8895. Two invariants this module exists to hold:

1. **Every close affordance goes through one guard.** A dialog with a
   "Close" button wired to ``QDialog.accept`` and a window-manager X wired
   to ``closeEvent`` has two closes with opposite semantics -- one asks
   before killing a running build, the other orphans it silently. Callers
   must wire their button to ``self.close()``, not ``self.accept()``.
2. **The GUI thread never blocks unbounded.** ``cancel()`` can take
   seconds (two sequential grace waits in ``docker_manager``), and the
   join after it used to be an untimed ``wait()``. Both now run under a
   wait cursor and the join is bounded; if the thread outlives the bound
   the user is told the build is still shutting down in the background
   rather than watching a frozen window.
"""

from __future__ import annotations

from typing import Any, Final, Protocol

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "BUILD_CANCEL_JOIN_TIMEOUT_MS",
    "CancellableBuildThread",
    "confirm_cancel_running_build_for_close",
]

#: Upper bound on how long the GUI thread will wait for a cancelled build
#: thread to finish. ``docker_manager`` spends up to two sequential
#: ``DOCKER_BUILD_CANCEL_GRACE_SECONDS`` (2.0 s) windows inside ``cancel()``,
#: so 5 s leaves headroom for the thread to notice and unwind without
#: letting the window sit unresponsive indefinitely.
BUILD_CANCEL_JOIN_TIMEOUT_MS: Final[int] = 5000


class CancellableBuildThread(Protocol):
    """Build thread contract required when closing launcher dialogs."""

    def isRunning(self) -> bool: ...

    def cancel(self) -> None: ...

    def wait(self, msecs: int = ...) -> bool: ...


def _cancel_and_join(build_thread: CancellableBuildThread) -> bool:
    """Cancel ``build_thread`` and join it under a bounded wait.

    Returns ``True`` when the thread finished within the bound. The wait
    cursor is set and restored in ``try``/``finally`` so an exception in
    ``cancel()`` cannot leave the application stuck showing an hourglass.
    """
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        build_thread.cancel()
        try:
            return bool(build_thread.wait(BUILD_CANCEL_JOIN_TIMEOUT_MS))
        except TypeError:
            # A test double (or an older thread type) whose ``wait`` takes
            # no timeout. Fall back rather than crash the close.
            return bool(build_thread.wait())
    finally:
        QApplication.restoreOverrideCursor()


def confirm_cancel_running_build_for_close(
    parent: QWidget,
    event: Any,
    build_thread: CancellableBuildThread | None,
    *,
    log_message: str,
) -> bool:
    """Cancel an active build before closing, or ignore the close event.

    Returns:
        True when the caller should continue closing; False when the close
        event was ignored because the user chose to keep the build running.

    The close is allowed to proceed even when the join times out: the
    cancel has been requested, and holding the window open would not make
    the thread stop any sooner. The user is told so explicitly instead of
    the build vanishing silently.
    """
    if parent is None:
        raise ValueError("parent must be provided")
    if event is None:
        raise ValueError("event must be provided")
    if not log_message:
        raise ValueError("log_message must be provided")

    if build_thread is None or not build_thread.isRunning():
        return True

    reply = QMessageBox.question(
        parent,
        "Build in progress",
        "Cancel the build and close?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if reply != QMessageBox.StandardButton.Yes:
        event.ignore()
        return False

    logger.info(log_message)
    if not _cancel_and_join(build_thread):
        logger.warning(
            "Build thread did not stop within %d ms of cancel; "
            "closing anyway and letting it unwind in the background",
            BUILD_CANCEL_JOIN_TIMEOUT_MS,
        )
        QMessageBox.information(
            parent,
            "Build still shutting down",
            "The build was cancelled but is still shutting down in the "
            "background. It will stop on its own shortly.",
        )
    return True
