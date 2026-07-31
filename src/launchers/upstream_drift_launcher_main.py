"""GUI startup entrypoint for the UpstreamDrift launcher."""

from __future__ import annotations

import os
import sys
import traceback
from types import TracebackType

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMessageBox

from src.launchers.launcher_constants import REPOS_ROOT, logger
from src.launchers.ui_components import (
    ASSETS_DIR,
    AsyncStartupWorker,
    SplashScreen,
    StartupResults,
)
from src.shared.python.ui import resolve_icon_path, set_app_user_model_id

_APP_USER_MODEL_ID = "D-sorganization.UpstreamDrift"


def _install_global_ui_zoom(app: QApplication) -> None:
    from src.launchers.app_zoom import install_global_ui_zoom

    install_global_ui_zoom(app)


def _write_crash_traceback(err_msg: str) -> None:
    try:
        with open("crash_traceback.txt", "w", encoding="utf-8") as handle:
            handle.write(err_msg)
    except OSError:
        pass


def _install_exception_hook() -> None:
    def excepthook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType | None,
    ) -> None:
        from src.launchers.launcher_dialogs import CriticalErrorDialog

        err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        _write_crash_traceback(err_msg)

        if exc_type is not SystemExit:
            dialog = CriticalErrorDialog(
                title="Application Crash",
                message="UpstreamDrift has encountered an unexpected error and must close.",
                detail_text=err_msg,
            )
            dialog.exec()

        QApplication.quit()

    sys.excepthook = excepthook


def _set_windows_app_user_model_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "UpstreamDrift.Launcher.1"
        )
    except ImportError:
        logger.debug("ctypes not available; skipping Windows AppUserModelID assignment")


def _apply_app_icon(app: QApplication) -> None:
    set_app_user_model_id(_APP_USER_MODEL_ID)
    app_icon = resolve_icon_path(
        [ASSETS_DIR / "golf_logo.ico", ASSETS_DIR / "golf_logo.png"]
    )
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
    worker = AsyncStartupWorker(REPOS_ROOT)

    main_window = UpstreamDriftLauncher(loading=True, splash=splash)
    main_window.show()

    def on_startup_finished(results: StartupResults) -> None:
        """Create and display the main window after startup completes."""
        nonlocal main_window
        try:
            main_window.update_startup_results(results)
            splash.finish(main_window)
        except (RuntimeError, ValueError, TypeError) as exc:
            traceback.print_exc()
            logger.error("Failed to update UpstreamDriftLauncher: %s", exc)
            QApplication.quit()
        worker.wait(1000)

    def on_startup_progress(msg: str, percent: int) -> None:
        """Forward startup progress."""
        logger.info("Startup progress: %s%% - %s", percent, msg)
        splash.show_message(msg, percent)

    def on_startup_error(error_msg: str) -> None:
        """Handle startup failure."""
        logger.error("Startup failed: %s", error_msg)
        splash.close()

        QMessageBox.critical(
            None, "Startup Error", f"Failed to initialize UpstreamDrift:\n\n{error_msg}"
        )
        QApplication.quit()

    worker.progress_signal.connect(on_startup_progress)
    worker.finished_signal.connect(on_startup_finished)
    worker.error_signal.connect(on_startup_error)

    worker.start()

    sys.exit(app.exec())
