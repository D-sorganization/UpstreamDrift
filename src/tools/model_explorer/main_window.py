"""Main window for the Interactive URDF Generator (Model Explorer).

This module is the standalone shell. The actual UI now lives in
:class:`src.tools.model_explorer.gui.MainWidget`; the window here owns
the menu bar, window icon, and unsaved-changes prompt, and delegates
everything else to the embedded widget.

The split is part of Subtask 5 / #4998 of EPIC #4993 — see
``src/tools/model_explorer/_embed_adapter.py`` for the launcher contract.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger

from .gui import MainWidget

logger = get_logger(__name__)


class URDFGeneratorWindow(QMainWindow):
    """Standalone :class:`QMainWindow` shell around :class:`MainWidget`.

    Provides the menu bar, window icon, and ``closeEvent`` prompt. The
    actual UI (segments dock, viewport, properties) is owned by the
    embedded :class:`MainWidget`, so the same widget can be hosted by
    the launcher embed adapter without spinning up a top-level window.
    """

    # Declared as class-level ``pyqtSignal`` attributes so they remain part
    # of ``URDFGeneratorWindow``'s Qt meta-object. Consumers that introspect
    # the window's declared signal surface (class-level/introspection-based
    # wiring) continue to see them; emissions are forwarded from the
    # embedded :class:`MainWidget` via signal-to-signal connections in
    # ``__init__``.
    urdf_generated = pyqtSignal(str)
    segment_added = pyqtSignal(dict)
    segment_removed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Model Explorer - UpstreamDrift")
        self.setMinimumSize(1200, 800)

        self._main_widget = MainWidget(self)
        self.setCentralWidget(self._main_widget)

        self._setup_window_icon()

        # Forward emissions from the embedded widget through the window's
        # own declared signals so existing consumers that subscribed to
        # the window keep working without losing the class-level signal
        # declarations.
        self._main_widget.urdf_generated.connect(self.urdf_generated)
        self._main_widget.segment_added.connect(self.segment_added)
        self._main_widget.segment_removed.connect(self.segment_removed)

        logger.info("URDF Generator window initialized")
        self._main_widget.load_default_model()

    # ---- accessors mirrored from the embedded widget -------------------

    @property
    def urdf_builder(self) -> Any:
        return self._main_widget.urdf_builder

    @property
    def current_file_path(self) -> Path | None:
        return self._main_widget.current_file_path

    @property
    def segment_panel(self) -> Any:
        return self._main_widget.segment_panel

    @property
    def visualization_widget(self) -> Any:
        return self._main_widget.visualization_widget

    @property
    def status_bar(self) -> Any:
        return self._main_widget.status_bar

    @property
    def segment_dock(self) -> Any:
        return self._main_widget.segment_dock

    @property
    def visualization_dock(self) -> Any:
        return self._main_widget.visualization_dock

    @property
    def properties_dock(self) -> Any:
        return self._main_widget.properties_dock

    # ---- menu construction --------------------------------------------

    def _setup_window_icon(self) -> None:
        from PyQt6.QtGui import QIcon

        icon_path = Path(__file__).parent / "assets" / "robot_arm_icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        else:
            logger.warning(f"Icon file not found: {icon_path}")

    # ---- delegate methods for compatibility ---------------------------

    def new_urdf(self) -> None:
        self._main_widget.new_urdf()

    def open_urdf(self) -> None:
        self._main_widget.open_urdf()

    def load_from_library(self) -> None:
        self._main_widget.load_from_library()

    def save_urdf(self) -> None:
        self._main_widget.save_urdf()

    def save_urdf_as(self) -> None:
        self._main_widget.save_urdf_as()

    def export_for_mujoco(self) -> None:
        self._main_widget.export_for_engine("MuJoCo", "golf_robot_mujoco.urdf")

    def export_for_drake(self) -> None:
        self._main_widget.export_for_engine("Drake", "golf_robot_drake.urdf")

    def export_for_pinocchio(self) -> None:
        self._main_widget.export_for_engine("Pinocchio", "golf_robot_pinocchio.urdf")

    def show_about(self) -> None:
        self._main_widget.show_about()

    # ---- close handling ------------------------------------------------

    def closeEvent(self, event: Any) -> None:
        if self._main_widget.has_unsaved_changes():
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                self.save_urdf()
                if self._main_widget.current_file_path is None:
                    event.ignore()
                    return
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

        self._main_widget.cleanup()
        event.accept()
        logger.info("URDF Generator window closed")


def get_dockable_ui() -> URDFGeneratorWindow:
    """Return the main window instance for docking in the unified launcher."""
    return URDFGeneratorWindow()


def main() -> None:
    """Standalone entry point for the URDF Generator."""
    from src.shared.python.logging_pkg.logging_config import configure_gui_logging

    app = QApplication(sys.argv)
    app.setApplicationName("URDF Generator")
    app.setApplicationVersion("1.0.0")

    configure_gui_logging()

    window = URDFGeneratorWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
