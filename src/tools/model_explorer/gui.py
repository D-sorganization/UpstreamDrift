"""Embeddable widget shell for the Model Explorer.

Subtask 5 / #4998 of EPIC #4993 refactors the Model Explorer so it can
launch as either a standalone :class:`QMainWindow` *or* an embedded
``QWidget`` inside the launcher host (tab / dock).

The historical entry point — :class:`URDFGeneratorWindow` in
:mod:`main_window` — still exists for back-compat, but its body now
delegates the actual UI to :class:`MainWidget` defined here. Embeddable
hosts construct :class:`MainWidget` directly and never see the
``QMainWindow`` shell.

The refactor preserves the dock-based layout (segments / viewport /
properties) by hosting an internal :class:`QMainWindow` inside
:class:`MainWidget` with ``Qt.WindowType.Widget`` flags. This is the
idiomatic way to reuse :class:`QMainWindow`'s docking machinery in a
non-top-level context (see Qt docs for ``QMainWindow`` "Creating Main
Window Components").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

try:
    from sidekick.ui.widgets.notepad_widget import NotepadWidget

    HAS_NOTEPAD = True
except ImportError:
    HAS_NOTEPAD = False
    logger.warning("sidekick not found, Notepad disabled")

from .segment_panel import SegmentPanel
from .urdf_builder import URDFBuilder
from .visualization_widget import VisualizationWidget

__all__ = ["MainWidget"]


class MainWidget(QWidget):
    """Embeddable Model Explorer widget.

    Owns the segments panel, 3D visualization, and properties dock that
    used to live directly on :class:`URDFGeneratorWindow`. The widget
    can be parented to any :class:`QWidget`; when the launcher host
    embeds it as a tab or dock, no top-level window is created.

    The widget exposes the same signals as the historical window so
    existing wiring keeps working::

        urdf_generated, segment_added, segment_removed
    """

    urdf_generated = pyqtSignal(str)
    segment_added = pyqtSignal(dict)
    segment_removed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.urdf_builder = URDFBuilder()
        self.current_file_path: Path | None = None
        self._notepad_window: Any | None = None

        # Internal QMainWindow gives us QDockWidget without forcing this
        # object to *be* a top-level window. ``Qt.WindowType.Widget``
        # tells Qt to treat it as a regular child widget.
        self._inner = QMainWindow(self)
        self._inner.setWindowFlags(Qt.WindowType.Widget)
        self._inner.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.GroupedDragging
        )

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._inner)

        self._setup_docks()
        self._setup_status_bar()
        self._setup_menu_bar()
        self._connect_signals()

        self.destroyed.connect(lambda: self.cleanup())

        logger.info("Model Explorer MainWidget initialized")

    # ---- construction --------------------------------------------------

    def _setup_docks(self) -> None:
        """Build the segments / viewport / properties dock layout."""
        self._inner.setCentralWidget(None)

        # Segments dock (left).
        self.segment_panel = SegmentPanel()
        self.segment_dock = QDockWidget("Model Segments", self._inner)
        self.segment_dock.setWidget(self.segment_panel)
        self.segment_dock.setObjectName("SegmentDock")
        self.segment_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self._inner.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self.segment_dock
        )

        # Viewport dock (right).
        self.visualization_widget = VisualizationWidget()
        self.visualization_dock = QDockWidget("3D Viewport", self._inner)
        self.visualization_dock.setWidget(self.visualization_widget)
        self.visualization_dock.setObjectName("ViewportDock")
        self.visualization_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self._inner.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.visualization_dock
        )

        # Properties dock (right, below segments).
        self.properties_dock = QDockWidget("Properties", self._inner)
        self.properties_dock.setObjectName("PropertiesDock")
        self.properties_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        properties_widget = QWidget()
        self.properties_dock.setWidget(properties_widget)
        self._inner.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.properties_dock
        )
        self._inner.splitDockWidget(
            self.segment_dock, self.properties_dock, Qt.Orientation.Vertical
        )

    def _setup_status_bar(self) -> None:
        self.status_bar = QStatusBar()
        self._inner.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _connect_signals(self) -> None:
        self.segment_panel.segment_added.connect(self._on_segment_added)
        self.segment_panel.segment_removed.connect(self._on_segment_removed)
        self.segment_panel.segment_modified.connect(self._on_segment_modified)

    # ---- accessors used by the main_window shell -----------------------

    @property
    def inner_main_window(self) -> QMainWindow:
        """Return the internal QMainWindow (used by the standalone shell).

        The standalone :class:`URDFGeneratorWindow` shells the menu/icon
        onto its own ``self``; ``inner_main_window`` is exposed so the
        shell can mirror status messages and dock visibility actions if
        desired. Embedded hosts must not rely on this.
        """
        return self._inner

    # ---- segment signal handlers --------------------------------------

    def _on_segment_added(self, segment_data: dict) -> None:
        try:
            self.urdf_builder.add_segment(segment_data)
            self.visualization_widget.update_visualization(self.urdf_builder.get_urdf())
            self.segment_added.emit(segment_data)
            self.status_bar.showMessage(f"Added segment: {segment_data['name']}")
            logger.info(f"Segment added: {segment_data['name']}")
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Error adding segment: {e}")
            QMessageBox.warning(self, "Error", f"Failed to add segment: {e}")

    def _on_segment_removed(self, segment_name: str) -> None:
        try:
            self.urdf_builder.remove_segment(segment_name)
            self.visualization_widget.update_visualization(self.urdf_builder.get_urdf())
            self.segment_removed.emit(segment_name)
            self.status_bar.showMessage(f"Removed segment: {segment_name}")
            logger.info(f"Segment removed: {segment_name}")
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Error removing segment: {e}")
            QMessageBox.warning(self, "Error", f"Failed to remove segment: {e}")

    def _on_segment_modified(self, segment_data: dict) -> None:
        try:
            self.urdf_builder.modify_segment(segment_data)
            self.visualization_widget.update_visualization(self.urdf_builder.get_urdf())
            self.status_bar.showMessage(f"Modified segment: {segment_data['name']}")
            logger.info(f"Segment modified: {segment_data['name']}")
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Error modifying segment: {e}")
            QMessageBox.warning(self, "Error", f"Failed to modify segment: {e}")

    # ---- file / URDF actions (delegated from the standalone shell) -----

    def new_urdf(self) -> None:
        """Reset the URDF builder and viewport."""
        self.urdf_builder.clear()
        self.segment_panel.clear()
        self.visualization_widget.clear()
        self.current_file_path = None
        self.status_bar.showMessage("New URDF created")
        logger.info("New URDF created")
        self._update_window_title()

    def open_urdf(self) -> None:
        """Open an existing URDF file via a file dialog."""
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open URDF File",
            "",
            "URDF Files (*.urdf);;XML Files (*.xml);;All Files (*)",
        )

        if file_path:
            try:
                self._load_urdf_file(Path(file_path))
                self.status_bar.showMessage(f"Opened: {file_path}")
                logger.info(f"URDF opened from: {file_path}")
            except (
                FileNotFoundError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as e:
                logger.error(f"Error opening URDF: {e}")
                QMessageBox.critical(self, "Error", f"Failed to open URDF: {e}")

    def load_from_library(self) -> None:  # noqa: C901
        """Load a URDF model from the bundled library."""
        try:
            from .model_library import ModelLibrary
            from .model_loader_dialog import ModelLoaderDialog

            dialog = ModelLoaderDialog(self)
            dialog.model_selected.connect(self._on_library_model_selected)

            if dialog.exec():
                selection = dialog.get_selected_model()
                if not selection:
                    return
                category, model_key = selection
                library = ModelLibrary()

                if category == "golf_clubs":
                    urdf_path = library.generate_golf_club_urdf(model_key)
                    if urdf_path:
                        self._load_urdf_file(urdf_path)
                        self.status_bar.showMessage(f"Loaded golf club: {model_key}")
                elif category == "human":
                    urdf_path = library.get_human_model(model_key)
                    if urdf_path and urdf_path.exists():
                        self._load_urdf_file(urdf_path)
                        self.status_bar.showMessage(f"Loaded human model: {model_key}")
                    else:
                        QMessageBox.information(
                            self,
                            "Model Not Available",
                            "This model is not bundled or downloaded.\n"
                            "Check bundled_assets/ for available models.",
                        )
                elif category in ["pendulum", "robotic", "component", "discovered"]:
                    model_info = library.get_model_info(category, model_key)
                    if model_info and "path" in model_info:
                        raw_path = model_info["path"]
                        path = Path(raw_path)
                        if not path.is_absolute():
                            from src.tools.model_explorer.model_library import (
                                _project_root,
                            )

                            path = _project_root / raw_path
                        if path.exists():
                            self._load_urdf_file(path)
                            self.status_bar.showMessage(
                                f"Loaded {category} model: {model_info['name']}"
                            )
                        else:
                            QMessageBox.warning(
                                self, "Error", f"File not found: {path}"
                            )
                    else:
                        QMessageBox.warning(
                            self,
                            "Error",
                            f"Invalid model configuration for {category}",
                        )
                elif category == "embedded":
                    model_info = library.get_model_info(category, model_key)
                    if model_info:
                        content = model_info["content"]
                        self.visualization_widget.update_visualization(content, None)
                        self.current_file_path = None
                        self.status_bar.showMessage(
                            f"Loaded embedded model: {model_info['name']}"
                        )
                        logger.info(f"Loaded embedded model: {model_info['name']}")
        except ImportError as e:
            logger.error(f"Error loading from library: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load from library: {e}")

    def _on_library_model_selected(self, category: str, model_key: str) -> None:
        logger.info(f"Model selected from library: {category}/{model_key}")

    def _load_urdf_file(self, file_path: Path) -> None:
        """Load URDF file content and refresh the viewport."""
        try:
            urdf_content = file_path.read_text(encoding="utf-8")
            self.visualization_widget.update_visualization(urdf_content, str(file_path))
            self.current_file_path = file_path
            self._update_window_title()
            logger.info(f"URDF loaded: {file_path}")
        except (RuntimeError, TypeError, ValueError) as e:
            logger.error(f"Error loading URDF file: {e}")
            raise

    def save_urdf(self) -> None:
        if self.current_file_path:
            self._save_to_file(self.current_file_path)
        else:
            self.save_urdf_as()

    def save_urdf_as(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save URDF File",
            "golf_robot.urdf",
            "URDF Files (*.urdf);;XML Files (*.xml);;All Files (*)",
        )
        if file_path:
            self._save_to_file(Path(file_path))

    def _save_to_file(self, file_path: Path) -> None:
        try:
            urdf_content = self.urdf_builder.get_urdf()
            file_path.write_text(urdf_content, encoding="utf-8")
            self.current_file_path = file_path
            self._update_window_title()
            self.status_bar.showMessage(f"Saved: {file_path}")
            logger.info(f"URDF saved to: {file_path}")
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Error saving URDF: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save URDF: {e}")

    def export_for_engine(self, engine: str, default_filename: str) -> None:
        """Export the current URDF, optimised for ``engine``.

        Currently exports a generic URDF (Drake / Pinocchio / MuJoCo all
        accept it). The hook exists so engine-specific tag emission can
        be added in one place later.
        """
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export for {engine}",
            default_filename,
            "URDF Files (*.urdf);;XML Files (*.xml)",
        )
        if not file_path:
            return
        try:
            urdf_content = self.urdf_builder.get_urdf()
            Path(file_path).write_text(urdf_content, encoding="utf-8")
            self.status_bar.showMessage(f"Exported for {engine}: {file_path}")
            logger.info(f"{engine} export saved to: {file_path}")
        except (FileNotFoundError, OSError) as e:
            logger.error(f"Error exporting for {engine}: {e}")
            QMessageBox.critical(self, "Error", f"Failed to export for {engine}: {e}")

    # ---- Tools menu actions -------------------------------------------

    def open_advanced_editor(self) -> None:
        try:
            from .urdf_editor_window import URDFEditorWindow

            self._editor_window = URDFEditorWindow()
            if self.current_file_path and self.current_file_path.exists():
                self._editor_window.load_file(self.current_file_path)
            self._editor_window.show()
            self.status_bar.showMessage("Opened Advanced URDF Editor")
        except ImportError as e:
            logger.error(f"Failed to open advanced editor: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open editor: {e}")

    def open_frankenstein_mode(self) -> None:
        try:
            from PyQt6.QtWidgets import QDialog, QVBoxLayout

            from .frankenstein_editor import FrankensteinEditor

            dialog = QDialog(self)
            dialog.setWindowTitle("Frankenstein Mode - Combine URDFs")
            dialog.setMinimumSize(1200, 700)
            layout = QVBoxLayout(dialog)
            frankenstein = FrankensteinEditor()
            layout.addWidget(frankenstein)
            if self.current_file_path and self.current_file_path.exists():
                frankenstein.load_source(self.current_file_path)
            dialog.exec()
            self.status_bar.showMessage("Frankenstein mode closed")
        except ImportError as e:
            logger.error(f"Failed to open Frankenstein mode: {e}")
            QMessageBox.critical(
                self, "Error", f"Failed to open Frankenstein mode: {e}"
            )

    def open_code_editor(self) -> None:
        try:
            from PyQt6.QtWidgets import QDialog, QVBoxLayout

            from .urdf_code_editor import URDFCodeEditorWidget

            dialog = QDialog(self)
            dialog.setWindowTitle("URDF Code Editor")
            dialog.setMinimumSize(800, 600)
            layout = QVBoxLayout(dialog)
            code_editor = URDFCodeEditorWidget()
            layout.addWidget(code_editor)
            if self.current_file_path and self.current_file_path.exists():
                content = self.current_file_path.read_text(encoding="utf-8")
                code_editor.set_content(content, str(self.current_file_path))
            dialog.exec()
            self.status_bar.showMessage("Code editor closed")
        except ImportError as e:
            logger.error(f"Failed to open code editor: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open code editor: {e}")

    # ---- lifecycle -----------------------------------------------------

    def has_unsaved_changes(self) -> bool:
        """Return ``True`` if the builder has segments and no file path."""
        return (
            self.urdf_builder.get_segment_count() > 0 and self.current_file_path is None
        )

    def reset_view(self) -> None:
        self.visualization_widget.reset_view()

    def cleanup(self) -> None:
        """Release any resources held by the widget. Idempotent."""
        # No long-lived background resources today; the visualization
        # widget owns its own GL context which Qt tears down with the
        # widget itself. The hook exists so the embed adapter can call
        # it without special-casing.

    def load_default_model(self) -> None:
        """Load the default URDF configured in QSettings (if any)."""
        try:
            from PyQt6.QtCore import QSettings

            from .model_library import ModelLibrary

            settings = QSettings("UpstreamDrift", "URDFGenerator")
            default_model = settings.value("default_human_model")
            if not default_model:
                default_model = "mujoco_humanoid"
            if default_model:
                logger.info(f"Loading default model: {default_model}")
                library = ModelLibrary()
                urdf_path = library.get_human_model(str(default_model))
                if urdf_path and urdf_path.exists():
                    self._load_urdf_file(urdf_path)
                    self.status_bar.showMessage(
                        f"Loaded default model: {default_model}"
                    )
                else:
                    logger.warning(f"Default model {default_model} not found")
        except ImportError as e:
            logger.error(f"Failed to load default model: {e}")

    # ---- Menu construction and helper actions ---------------------------

    def _setup_menu_bar(self) -> None:
        menubar = self._inner.menuBar()
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
        exit_action.triggered.connect(self._on_exit)
        file_menu.addAction(exit_action)

    def _setup_export_menu(self, export_menu: Any) -> None:
        export_mujoco_action = QAction("Export for MuJoCo", self)
        export_mujoco_action.triggered.connect(
            lambda: self.export_for_engine("MuJoCo", "golf_robot_mujoco.urdf")
        )
        export_menu.addAction(export_mujoco_action)

        export_drake_action = QAction("Export for Drake", self)
        export_drake_action.triggered.connect(
            lambda: self.export_for_engine("Drake", "golf_robot_drake.urdf")
        )
        export_menu.addAction(export_drake_action)

        export_pinocchio_action = QAction("Export for Pinocchio", self)
        export_pinocchio_action.triggered.connect(
            lambda: self.export_for_engine("Pinocchio", "golf_robot_pinocchio.urdf")
        )
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
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)

    def _setup_tools_menu(self, tools_menu: Any) -> None:
        advanced_editor_action = QAction("Advanced URDF &Editor...", self)
        advanced_editor_action.setShortcut("Ctrl+E")
        advanced_editor_action.triggered.connect(self.open_advanced_editor)
        tools_menu.addAction(advanced_editor_action)

        tools_menu.addSeparator()

        frankenstein_action = QAction("&Frankenstein Mode...", self)
        frankenstein_action.setToolTip("Combine components from multiple URDFs")
        frankenstein_action.triggered.connect(self.open_frankenstein_mode)
        tools_menu.addAction(frankenstein_action)

        code_editor_action = QAction("&Code Editor...", self)
        code_editor_action.setToolTip("Edit URDF XML directly with syntax highlighting")
        code_editor_action.triggered.connect(self.open_code_editor)
        tools_menu.addAction(code_editor_action)

        if HAS_NOTEPAD:
            tools_menu.addSeparator()
            notepad_action = QAction("&Notepad...", self)
            notepad_action.setShortcut("Ctrl+Shift+N")
            notepad_action.triggered.connect(self._open_notepad)
            tools_menu.addAction(notepad_action)

    def _on_exit(self) -> None:
        win = self.window()
        if win:
            win.close()

    def _update_window_title(self) -> None:
        win = self.window()
        if win and win.__class__.__name__ == "URDFGeneratorWindow":
            if self.current_file_path:
                win.setWindowTitle(f"Model Explorer - {self.current_file_path.name}")
            else:
                win.setWindowTitle("Model Explorer - UpstreamDrift")

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

                win = self.window()
                main_geo = win.geometry() if win else self.geometry()
                self._notepad_window.move(main_geo.x() + 100, main_geo.y() + 100)

            self._notepad_window.show()
            self._notepad_window.raise_()
            self._notepad_window.activateWindow()
        except (ImportError, OSError, RuntimeError) as e:
            logger.error(f"Error opening notepad: {e}")
            QMessageBox.warning(self, "Error", f"Failed to open notepad: {e}")
