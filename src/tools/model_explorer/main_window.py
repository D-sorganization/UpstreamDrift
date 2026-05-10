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
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger

from .gui import MainWidget

logger = get_logger(__name__)

try:
    from upstream_drift_tools.ui.widgets.notepad_widget import NotepadWidget

    HAS_NOTEPAD = True
except ImportError:
    HAS_NOTEPAD = False
    logger.warning("upstream_drift_tools not found, Notepad disabled")


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

        self._notepad_window: Any | None = None

        self._setup_menu_bar()
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

    def _setup_menu_bar(self) -> None:
        menubar = self.menuBar()
        if menubar is None:
            return

        file_menu = menubar.addMenu("&File")
        if file_menu is not None:
            self._setup_file_menu(file_menu)

        edit_menu = menubar.addMenu("&Edit")
        if edit_menu is not None:
            self._setup_edit_menu(edit_menu)

        view_menu = menubar.addMenu("&View")
        if view_menu is not None:
            self._setup_view_menu(view_menu)

        tools_menu = menubar.addMenu("&Tools")
        if tools_menu is not None:
            self._setup_tools_menu(tools_menu)

        help_menu = menubar.addMenu("&Help")
        if help_menu is not None:
            about_action = QAction("&About", self)
            about_action.triggered.connect(self.show_about)
            help_menu.addAction(about_action)

    def _setup_file_menu(self, file_menu: Any) -> None:
        new_action = QAction("&New", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_urdf)
        file_menu.addAction(new_action)

        open_action = QAction("&Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_urdf)
        file_menu.addAction(open_action)

        load_library_action = QAction("Load from &Library...", self)
        load_library_action.setShortcut("Ctrl+L")
        load_library_action.triggered.connect(self.load_from_library)
        file_menu.addAction(load_library_action)

        file_menu.addSeparator()

        save_action = QAction("&Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_urdf)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_urdf_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        export_menu = file_menu.addMenu("&Export")
        if export_menu is not None:
            self._setup_export_menu(export_menu)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _setup_export_menu(self, export_menu: Any) -> None:
        export_mujoco_action = QAction("Export for MuJoCo", self)
        export_mujoco_action.triggered.connect(self.export_for_mujoco)
        export_menu.addAction(export_mujoco_action)

        export_drake_action = QAction("Export for Drake", self)
        export_drake_action.triggered.connect(self.export_for_drake)
        export_menu.addAction(export_drake_action)

        export_pinocchio_action = QAction("Export for Pinocchio", self)
        export_pinocchio_action.triggered.connect(self.export_for_pinocchio)
        export_menu.addAction(export_pinocchio_action)

    def _setup_edit_menu(self, edit_menu: Any) -> None:
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.setEnabled(False)
        edit_menu.addAction(undo_action)

        redo_action = QAction("&Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.setEnabled(False)
        edit_menu.addAction(redo_action)

    def _setup_view_menu(self, view_menu: Any) -> None:
        reset_view_action = QAction("&Reset View", self)
        reset_view_action.setShortcut("Ctrl+R")
        reset_view_action.triggered.connect(self._main_widget.reset_view)
        view_menu.addAction(reset_view_action)

    def _setup_tools_menu(self, tools_menu: Any) -> None:
        advanced_editor_action = QAction("Advanced URDF &Editor...", self)
        advanced_editor_action.setShortcut("Ctrl+E")
        advanced_editor_action.triggered.connect(self._main_widget.open_advanced_editor)
        tools_menu.addAction(advanced_editor_action)

        tools_menu.addSeparator()

        frankenstein_action = QAction("&Frankenstein Mode...", self)
        frankenstein_action.setToolTip("Combine components from multiple URDFs")
        frankenstein_action.triggered.connect(self._main_widget.open_frankenstein_mode)
        tools_menu.addAction(frankenstein_action)

        code_editor_action = QAction("&Code Editor...", self)
        code_editor_action.setToolTip("Edit URDF XML directly with syntax highlighting")
        code_editor_action.triggered.connect(self._main_widget.open_code_editor)
        tools_menu.addAction(code_editor_action)

        if HAS_NOTEPAD:
            tools_menu.addSeparator()
            notepad_action = QAction("&Notepad...", self)
            notepad_action.setShortcut("Ctrl+Shift+N")
            notepad_action.triggered.connect(self._open_notepad)
            tools_menu.addAction(notepad_action)

    def _setup_window_icon(self) -> None:
        from PyQt6.QtGui import QIcon

        icon_path = Path(__file__).parent / "assets" / "robot_arm_icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        else:
            logger.warning(f"Icon file not found: {icon_path}")

    # ---- menu action delegates ----------------------------------------

    def new_urdf(self) -> None:
        self._main_widget.new_urdf()
        self.setWindowTitle("Interactive URDF Generator - Golf Modeling Suite")

    def open_urdf(self) -> None:
        self._main_widget.open_urdf()

    def load_from_library(self) -> None:
        self._main_widget.load_from_library()
        if self._main_widget.current_file_path is not None:
            self.setWindowTitle(
                f"Interactive URDF Generator - "
                f"{self._main_widget.current_file_path.name}"
            )

    def save_urdf(self) -> None:
        self._main_widget.save_urdf()
        if self._main_widget.current_file_path is not None:
            self.setWindowTitle(
                f"Interactive URDF Generator - "
                f"{self._main_widget.current_file_path.name}"
            )

    def save_urdf_as(self) -> None:
        self._main_widget.save_urdf_as()
        if self._main_widget.current_file_path is not None:
            self.setWindowTitle(
                f"Interactive URDF Generator - "
                f"{self._main_widget.current_file_path.name}"
            )

    def export_for_mujoco(self) -> None:
        self._main_widget.export_for_engine("MuJoCo", "golf_robot_mujoco.urdf")

    def export_for_drake(self) -> None:
        self._main_widget.export_for_engine("Drake", "golf_robot_drake.urdf")

    def export_for_pinocchio(self) -> None:
        self._main_widget.export_for_engine("Pinocchio", "golf_robot_pinocchio.urdf")

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "About URDF Generator",
            "Interactive URDF Generator v2.0\n"
            "Part of UpstreamDrift\n\n"
            "Create and edit URDF files with support for\n"
            "parallel kinematic configurations.\n\n"
            "New features in v2.0:\n"
            "- Component library with read-only protection\n"
            "- Frankenstein mode for combining URDFs\n"
            "- Chain manipulation tools\n"
            "- End effector swap system\n"
            "- Joint auto-loader\n"
            "- Mesh/STL browser\n\n"
            "Compatible with MuJoCo, Drake, and Pinocchio.",
        )

    # ---- notepad (window-only feature) --------------------------------

    def _open_notepad(self) -> None:
        if not HAS_NOTEPAD:
            return
        try:
            if not hasattr(self, "_notepad_window") or self._notepad_window is None:
                try:
                    from src.shared.python.paths import get_user_data_dir

                    storage_dir = get_user_data_dir() / "notepad"
                except ImportError:
                    storage_dir = Path.home() / ".upstream_drift" / "notepad"

                self._notepad_window = NotepadWidget(
                    storage_dir=storage_dir, app_name="Upstream Drift"
                )

                main_geo = self.geometry()
                self._notepad_window.move(main_geo.x() + 100, main_geo.y() + 100)

            self._notepad_window.show()
            self._notepad_window.raise_()
            self._notepad_window.activateWindow()
        except (ImportError, OSError, RuntimeError) as e:
            logger.error(f"Error opening notepad: {e}")
            QMessageBox.warning(self, "Error", f"Failed to open notepad: {e}")

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
