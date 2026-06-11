"""Shared close-event handling for active launcher builds."""

from __future__ import annotations

from typing import Any, Protocol

from PyQt6.QtWidgets import QMessageBox, QWidget

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class CancellableBuildThread(Protocol):
    """Build thread contract required when closing launcher dialogs."""

    def isRunning(self) -> bool: ...

    def cancel(self) -> None: ...

    def wait(self) -> bool: ...


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
    build_thread.cancel()
    build_thread.wait()
    return True
