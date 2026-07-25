"""Contained failure reporting for launcher tile launches.

Functional QA (#8062) found the launcher terminating, blanking, or silently
succeeding whenever a tile could not start.  Every launch surface now routes
its failures through this mixin so the outcome is always the same: the
launcher stays alive and the user gets a message that names what is missing
and how to install it.

Covers:
    * exceptions escaping a tile launch (#8066, #8070, #8072, #8084)
    * child processes that exit immediately after a "Launched" log
      (#8065, #8069)
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QMessageBox

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.theme.style_constants import Styles

logger = get_logger(__name__)

__all__ = ["LaunchFailureReportingMixin"]


class LaunchFailureReportingMixin:
    """Mixin providing contained, actionable launch-failure reporting.

    Expects the host launcher to provide ``show_toast``, ``lbl_status``,
    ``launcher`` and (optionally) ``_append_console_line``.
    """

    #: Grace period (ms) before a freshly spawned tile is checked for liveness.
    CHILD_LIVENESS_GRACE_MS = 4000

    def _watch_child_process(self, display_name: str, process: Any) -> None:
        """Report tiles whose child process dies immediately after launch.

        ``Popen`` returning a handle only proves the interpreter started, not
        that the tool came up. Tiles used to report "Launched <name> (PID: n)"
        and then vanish with no window and no error (#8065, #8069). This
        schedules a non-blocking liveness check and surfaces a failure toast if
        the child is already gone.

        Args:
            display_name: Human-readable tile name.
            process: The spawned ``subprocess.Popen`` handle, or None.

        Raises:
            ValueError: If ``display_name`` is empty.
        """
        if not display_name or not display_name.strip():
            raise ValueError("display_name must be a non-empty string")
        if process is None:
            return

        def _check() -> None:
            returncode = process.poll()
            if returncode is None:
                return
            logger.error(
                "%s exited immediately after launch (returncode=%s)",
                display_name,
                returncode,
            )
            message = (
                f"{display_name} started but closed straight away "
                f"(exit code {returncode}).\n\n"
                "This normally means one of its optional dependencies is "
                "missing. Install the desktop extras with:\n"
                "    pip install upstream-drift[gui-tools]\n\n"
                "Open the console dock (View -> Console) for the child's output."
            )
            if hasattr(self, "_append_console_line"):
                self._append_console_line("Launcher", message)
            self.show_toast(f"{display_name} exited immediately", "error")
            self.lbl_status.setText("! Launch Error")
            self.lbl_status.setStyleSheet(Styles.STATUS_ERROR)

        QTimer.singleShot(self.CHILD_LIVENESS_GRACE_MS, _check)

    def _report_contained_launch_failure(
        self, model_name: str, exc: BaseException
    ) -> None:
        """Surface a failed tile launch without terminating the launcher.

        Any exception raised while starting a tile — a missing optional
        dependency, a broken native library, an absent MATLAB install — is
        contained here and turned into an actionable in-product message
        (issues #8066, #8070, #8072, #8084).

        Args:
            model_name: Human-readable tile name.
            exc: The exception raised by the launch attempt.

        Postcondition:
            The launcher process remains alive and returns to the Ready state.
        """
        from src.launchers.launcher_failure_messages import describe_launch_failure

        logger.exception("Launch failed for %s", model_name, exc_info=exc)

        message = describe_launch_failure(exc, model_name)
        if hasattr(self, "_append_console_line"):
            import traceback as _traceback

            tb_str = "".join(
                _traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            self._append_console_line("Launcher", f"{message}\n\n{tb_str}")

        self.show_toast(f"{model_name} could not be started", "error")
        self._open_launch_failure_dialog(message)

        self.lbl_status.setText("> Ready")
        self.lbl_status.setStyleSheet(Styles.STATUS_INACTIVE)

    def _open_launch_failure_dialog(self, message: str) -> None:
        """Show the failure message without blocking the launcher's event loop.

        ``QMessageBox.warning`` is modal and blocks until dismissed, which is
        exactly the behaviour that made a failed tile feel like a frozen or
        dying application. ``open()`` shows the same dialog window-modally and
        returns immediately, so the launcher stays interactive.

        Args:
            message: The actionable, traceback-free text to display.

        Raises:
            ValueError: If ``message`` is empty.
        """
        if not message or not message.strip():
            raise ValueError("message must be a non-empty string")

        box = QMessageBox(getattr(self, "launcher", None))
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Launch Failed")
        box.setText(message)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        # Keep a reference so the box is not garbage collected before it shows.
        self._last_launch_failure_dialog = box
        box.open()
