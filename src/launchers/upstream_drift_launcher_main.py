"""GUI startup entrypoint for the UpstreamDrift launcher."""

from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from src.launchers.launcher_constants import REPOS_ROOT, logger
from src.launchers.startup_session import StartupSession
from src.launchers.ui_components import (
    ASSETS_DIR,
    AsyncStartupWorker,
    SplashScreen,
)
from src.launchers.app_identity import (
    APP_USER_MODEL_ID,
    CANONICAL_ICON_CANDIDATES,
)
from src.shared.python.ui import resolve_icon_path, set_app_user_model_id

_APP_USER_MODEL_ID = APP_USER_MODEL_ID


def _install_global_ui_zoom(app: QApplication) -> None:
    from src.launchers.app_zoom import install_global_ui_zoom

    install_global_ui_zoom(app)


_event_loop_running = False


def mark_event_loop_running(running: bool = True) -> None:
    """Record that the Qt event loop is live, so errors become recoverable."""
    global _event_loop_running
    _event_loop_running = running


def _crash_log_dir() -> Path:
    from src.launchers.launcher_constants import CONFIG_DIR

    return CONFIG_DIR / "logs"


def _write_crash_traceback(err_msg: str, log_dir: Path | None = None) -> Path | None:
    """Write the traceback under the per-user log dir (never the CWD, #12886)."""
    try:
        target = (log_dir if log_dir is not None else _crash_log_dir()) / (
            "crash_traceback.txt"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(err_msg, encoding="utf-8")
        return target
    except OSError:
        return None


def _show_error_dialog(title: str, message: str, detail_text: str) -> None:
    from src.launchers.launcher_dialogs import CriticalErrorDialog

    CriticalErrorDialog(title=title, message=message, detail_text=detail_text).exec()


def handle_uncaught_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb: TracebackType | None,
    *,
    show_dialog: Callable[[str, str, str], None] = _show_error_dialog,
    log_dir: Path | None = None,
) -> bool:
    """Report an uncaught exception; return True when the app must quit.

    Startup failures (before the event loop runs) are fatal. Once the loop is
    running, an error in one slot is shown in a dialog and the launcher plus
    its other docked tools stay alive (issue #12886).
    """
    err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.error("Uncaught exception:\n%s", err_msg)
    saved = _write_crash_traceback(err_msg, log_dir)
    if exc_type is SystemExit:
        return True
    fatal = not _event_loop_running
    where = f"\n\nTraceback saved to {saved}" if saved else ""
    try:
        if fatal:
            show_dialog(
                "Application Crash",
                "UpstreamDrift has encountered an unexpected error and must close.",
                err_msg,
            )
        else:
            show_dialog(
                "Unexpected Error",
                "An action failed. The launcher is still running; other tools "
                f"are unaffected.{where}",
                err_msg,
            )
    except Exception:  # noqa: BLE001 - the hook itself must never raise
        logger.exception("Could not show error dialog")
    return fatal


def _install_exception_hook() -> None:
    def excepthook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType | None,
    ) -> None:
        if handle_uncaught_exception(exc_type, exc_value, exc_tb):
            QApplication.quit()

    sys.excepthook = excepthook


def _set_windows_app_user_model_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            _APP_USER_MODEL_ID
        )
    except (AttributeError, OSError, NameError, ImportError):
        logger.debug("ctypes not available; skipping Windows AppUserModelID assignment")


def _apply_app_icon(app: QApplication) -> None:
    set_app_user_model_id(_APP_USER_MODEL_ID)
    candidates = [ASSETS_DIR / name for name in CANONICAL_ICON_CANDIDATES]
    app_icon = resolve_icon_path(candidates)
    if app_icon is not None:
        app.setWindowIcon(QIcon(str(app_icon)))


def _apply_stylesheet(app: QApplication) -> None:
    qss_path = ASSETS_DIR / "theme" / "dark_modern.qss"
    if not qss_path.exists():
        return
    try:
        with open(qss_path) as handle:
            app.setStyleSheet(handle.read())
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Could not load QSS: %s", exc)


def _apply_optional_ui_themes(app: QApplication) -> None:
    try:
        from src.shared.python.plot_theme import apply_plot_theme

        apply_plot_theme(settings_app="UpstreamDrift")
    except ImportError:
        logger.debug("Plot theme module not available")

    try:
        from src.shared.python.theme.zoom import install_application_zoom

        install_application_zoom(app)
    except ImportError as exc:
        logger.debug("Zoom support not available: %s", exc)


def _build_app() -> QApplication:
    os.environ.setdefault("GOLF_SUITE_MODE", "local")
    _install_exception_hook()
    _set_windows_app_user_model_id()

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
    _install_global_ui_zoom(app)
    _apply_app_icon(app)
    _apply_stylesheet(app)
    _apply_optional_ui_themes(app)
    return app


def main() -> None:
    """Application entry point."""
    from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher

    app = _build_app()
    splash = SplashScreen()
    splash.show()

    main_window = UpstreamDriftLauncher(loading=True, splash=splash)
    main_window.show()

    # Issue #8360: the session bounds the splash lifetime, routes worker
    # callbacks only to live widgets, degrades on optional-provider failure
    # and offers Retry / Continue / Copy diagnostics / Close on hard failure.
    session = StartupSession(
        splash=splash,
        shell=main_window,
        worker_factory=lambda: AsyncStartupWorker(REPOS_ROOT),
        parent=main_window,
    )
    app.aboutToQuit.connect(session.shutdown)
    session.start()
    QTimer.singleShot(0, mark_event_loop_running)

    sys.exit(app.exec())
