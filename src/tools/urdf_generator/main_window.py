"""Main window for the Interactive URDF Generator."""

from src.shared.python.logging_config import get_logger
import sys
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QWidget,
)

from .segment_panel import SegmentPanel
from .urdf_builder import URDFBuilder
from .visualization_widget import VisualizationWidget

logger = get_logger(__name__)


class URDFGeneratorWindow(QMainWindow):
    """Main window for the Interactive URDF Generator."""

    # Signals
    urdf_generated = pyqtSignal(str)  # Emitted when URDF is generated
    segment_added = pyqtSignal(dict)  # Emitted when a segment is added
    segment_removed = pyqtSignal(str)  # Emitted when a segment is removed

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the main window.

        Args:
            parent: Parent widget, if any.
        """
        super().__init__(parent)
        self.urdf_builder = URDFBuilder()
        self.current_file_path: Path | None = None

        self._setup_ui()
        self._setup_menu_bar()
        self._setup_status_bar()
        self._setup_window_icon()
        self._connect_signals()

        logger.info("URDF Generator window initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Interactive URDF Generator - Golf Modeling Suite")
        self.setMinimumSize(1200, 800)

        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel for segment management
        self.segment_panel = SegmentPanel()
        splitter.addWidget(self.segment_panel)

        # Right panel for 3D visualization
        self.visualization_widget = VisualizationWidget()
        splitter.addWidget(self.visualization_widget)

        # Set splitter proportions (30% left, 70% right)
        splitter.setSizes([360, 840])

        # Properties dock widget
        self._setup_properties_dock()

    def _setup_properties_dock(self) -> None:
        """Set up the properties dock widget."""
        dock = QDockWidget("Properties", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        # Properties widget will be implemented later
        properties_widget = QWidget()
        dock.setWidget(properties_widget)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _setup_menu_bar(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()
        if menubar is None:
            return

        # File menu
        file_menu = menubar.addMenu("&File")
        if file_menu is None:
            return

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
        if export_menu is None:
            return

        export_mujoco_action = QAction("Export for MuJoCo", self)
        export_mujoco_action.triggered.connect(self.export_for_mujoco)
        export_menu.addAction(export_mujoco_action)

        export_drake_action = QAction("Export for Drake", self)
        export_drake_action.triggered.connect(self.export_for_drake)
        export_menu.addAction(export_drake_action)

        export_pinocchio_action = QAction("Export for Pinocchio", self)
        export_pinocchio_action.triggered.connect(self.export_for_pinocchio)
        export_menu.addAction(export_pinocchio_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        if edit_menu is None:
            return

        undo_action = QAction("&Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.setEnabled(
            False
        )  # Undo/redo functionality planned for future release
        edit_menu.addAction(undo_action)

        redo_action = QAction("&Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.setEnabled(
            False
        )  # Undo/redo functionality planned for future release
        edit_menu.addAction(redo_action)

        # View menu
        view_menu = menubar.addMenu("&View")
        if view_menu is None:
            return

        reset_view_action = QAction("&Reset View", self)
        reset_view_action.setShortcut("Ctrl+R")
        reset_view_action.triggered.connect(self.visualization_widget.reset_view)
        view_menu.addAction(reset_view_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        if help_menu is None:
            return

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _setup_status_bar(self) -> None:
        """Set up the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _setup_window_icon(self) -> None:
        """Set up the window icon."""
        from PyQt6.QtGui import QIcon

        icon_path = Path(__file__).parent / "assets" / "robot_arm_icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        else:
            logger.warning(f"Icon file not found: {icon_path}")

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self.segment_panel.segment_added.connect(self._on_segment_added)
        self.segment_panel.segment_removed.connect(self._on_segment_removed)
        self.segment_panel.segment_modified.connect(self._on_segment_modified)

    def _on_segment_added(self, segment_data: dict) -> None:
        """Handle segment addition.

        Args:
            segment_data: Dictionary containing segment information.
        """
        try:
            self.urdf_builder.add_segment(segment_data)
            self.visualization_widget.update_visualization(self.urdf_builder.get_urdf())
            self.segment_added.emit(segment_data)
            self.status_bar.showMessage(f"Added segment: {segment_data['name']}")
            logger.info(f"Segment added: {segment_data['name']}")
        except Exception as e:
            logger.error(f"Error adding segment: {e}")
            QMessageBox.warning(self, "Error", f"Failed to add segment: {e}")

    def _on_segment_removed(self, segment_name: str) -> None:
        """Handle segment removal.

        Args:
            segment_name: Name of the segment to remove.
        """
        try:
            self.urdf_builder.remove_segment(segment_name)
            self.visualization_widget.update_visualization(self.urdf_builder.get_urdf())
            self.segment_removed.emit(segment_name)
            self.status_bar.showMessage(f"Removed segment: {segment_name}")
            logger.info(f"Segment removed: {segment_name}")
        except Exception as e:
            logger.error(f"Error removing segment: {e}")
            QMessageBox.warning(self, "Error", f"Failed to remove segment: {e}")

    def _on_segment_modified(self, segment_data: dict) -> None:
        """Handle segment modification.

        Args:
            segment_data: Dictionary containing updated segment information.
        """
        try:
            self.urdf_builder.modify_segment(segment_data)
            self.visualization_widget.update_visualization(self.urdf_builder.get_urdf())
            self.status_bar.showMessage(f"Modified segment: {segment_data['name']}")
            logger.info(f"Segment modified: {segment_data['name']}")
        except Exception as e:
            logger.error(f"Error modifying segment: {e}")
            QMessageBox.warning(self, "Error", f"Failed to modify segment: {e}")

    def new_urdf(self) -> None:
        """Create a new URDF."""
        # Check for unsaved changes before creating new URDF
        self.urdf_builder.clear()
        self.segment_panel.clear()
        self.visualization_widget.clear()
        self.current_file_path = None
        self.setWindowTitle("Interactive URDF Generator - Golf Modeling Suite")
        self.status_bar.showMessage("New URDF created")
        logger.info("New URDF created")

    def open_urdf(self) -> None:
        """Open an existing URDF file."""
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open URDF File",
            "",
            "URDF Files (*.urdf);;XML Files (*.xml);;All Files (*)",
        )

        if file_path:
            try:
                _ = Path(file_path).read_text(encoding="utf-8")
                # Parse URDF and populate segments (future enhancement)
                self.status_bar.showMessage(f"Opened: {file_path}")
                logger.info(f"URDF opened from: {file_path}")
            except Exception as e:
                logger.error(f"Error opening URDF: {e}")
                QMessageBox.critical(self, "Error", f"Failed to open URDF: {e}")

    def load_from_library(self) -> None:
        """Load a URDF model from the library."""
        try:
            from .model_library import ModelLibrary
            from .model_loader_dialog import ModelLoaderDialog

            dialog = ModelLoaderDialog(self)
            dialog.model_selected.connect(self._on_library_model_selected)

            if dialog.exec():
                selection = dialog.get_selected_model()
                if selection:
                    category, model_key = selection
                    library = ModelLibrary()

                    if category == "golf_clubs":
                        # Generate golf club URDF
                        urdf_path = library.generate_golf_club_urdf(model_key)
                        if urdf_path:
                            self._load_urdf_file(urdf_path)
                            self.status_bar.showMessage(
                                f"Loaded golf club: {model_key}"
                            )
                    elif category == "human":
                        # Load human model URDF
                        model_dir = library.human_models_path / model_key
                        urdf_path = model_dir / "model.urdf"

                        if urdf_path.exists():
                            self._load_urdf_file(urdf_path)
                            self.status_bar.showMessage(
                                f"Loaded human model: {model_key}"
                            )
                        else:
                            QMessageBox.information(
                                self,
                                "Model Not Downloaded",
                                "This model has not been downloaded yet.\n"
                                "Please use the 'Download from human-gazebo' button "
                                "in the model loader dialog.",
                            )

        except Exception as e:
            logger.error(f"Error loading from library: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load from library: {e}")

    def _on_library_model_selected(self, category: str, model_key: str) -> None:
        """Handle model selection from library.

        Args:
            category: Model category ('human' or 'golf_clubs')
            model_key: Model identifier
        """
        logger.info(f"Model selected from library: {category}/{model_key}")

    def _load_urdf_file(self, file_path: Path) -> None:
        """Load URDF file content and update visualization.

        Args:
            file_path: Path to URDF file to load
        """
        try:
            urdf_content = file_path.read_text(encoding="utf-8")
            # URDF parsing for segment panel population is a future enhancement
            # Currently loading for visualization only
            self.visualization_widget.update_visualization(urdf_content)
            self.current_file_path = file_path
            self.setWindowTitle(f"Interactive URDF Generator - {file_path.name}")
            logger.info(f"URDF loaded: {file_path}")
        except Exception as e:
            logger.error(f"Error loading URDF file: {e}")
            raise

    def save_urdf(self) -> None:
        """Save the current URDF."""
        if self.current_file_path:
            self._save_to_file(self.current_file_path)
        else:
            self.save_urdf_as()

    def save_urdf_as(self) -> None:
        """Save the current URDF with a new filename."""
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
        """Save URDF to the specified file.

        Args:
            file_path: Path to save the file to.
        """
        try:
            urdf_content = self.urdf_builder.get_urdf()
            file_path.write_text(urdf_content, encoding="utf-8")
            self.current_file_path = file_path
            self.setWindowTitle(f"Interactive URDF Generator - {file_path.name}")
            self.status_bar.showMessage(f"Saved: {file_path}")
            logger.info(f"URDF saved to: {file_path}")
        except Exception as e:
            logger.error(f"Error saving URDF: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save URDF: {e}")

    def export_for_mujoco(self) -> None:
        """Export URDF optimized for MuJoCo."""
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export for MuJoCo",
            "golf_robot_mujoco.urdf",
            "URDF Files (*.urdf);;XML Files (*.xml)",
        )

        if file_path:
            try:
                # Get MuJoCo-optimized URDF
                urdf_content = self.urdf_builder.get_urdf()
                # Note: Currently exporting generic URDF which MuJoCo can read directly.
                # Specific <mujoco> tags can be added here if needed in future.

                Path(file_path).write_text(urdf_content, encoding="utf-8")
                self.status_bar.showMessage(f"Exported for MuJoCo: {file_path}")
                logger.info(f"MuJoCo export saved to: {file_path}")
            except Exception as e:
                logger.error(f"Error exporting for MuJoCo: {e}")
                QMessageBox.critical(self, "Error", f"Failed to export for MuJoCo: {e}")

    def export_for_drake(self) -> None:
        """Export URDF optimized for Drake."""
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export for Drake",
            "golf_robot_drake.urdf",
            "URDF Files (*.urdf);;XML Files (*.xml)",
        )

        if file_path:
            try:
                # Get Drake-optimized URDF
                urdf_content = self.urdf_builder.get_urdf()
                # Note: Currently exporting generic URDF. Drake requires standard URDF.
                # Collision tags are already handled by URDFBuilder.

                Path(file_path).write_text(urdf_content, encoding="utf-8")
                self.status_bar.showMessage(f"Exported for Drake: {file_path}")
                logger.info(f"Drake export saved to: {file_path}")
            except Exception as e:
                logger.error(f"Error exporting for Drake: {e}")
                QMessageBox.critical(self, "Error", f"Failed to export for Drake: {e}")

    def export_for_pinocchio(self) -> None:
        """Export URDF optimized for Pinocchio."""
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export for Pinocchio",
            "golf_robot_pinocchio.urdf",
            "URDF Files (*.urdf);;XML Files (*.xml)",
        )

        if file_path:
            try:
                # Get Pinocchio-optimized URDF
                urdf_content = self.urdf_builder.get_urdf()
                # Note: Pinocchio uses standard URDF loader. No specific tags needed.

                Path(file_path).write_text(urdf_content, encoding="utf-8")
                self.status_bar.showMessage(f"Exported for Pinocchio: {file_path}")
                logger.info(f"Pinocchio export saved to: {file_path}")
            except Exception as e:
                logger.error(f"Error exporting for Pinocchio: {e}")
                QMessageBox.critical(
                    self, "Error", f"Failed to export for Pinocchio: {e}"
                )

    def show_about(self) -> None:
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About URDF Generator",
            "Interactive URDF Generator\n"
            "Part of the Golf Modeling Suite\n\n"
            "Create and edit URDF files with support for\n"
            "parallel kinematic configurations.\n\n"
            "Compatible with MuJoCo, Drake, and Pinocchio.",
        )

    def closeEvent(self, event: Any) -> None:
        """Handle window close event."""
        from PyQt6.QtWidgets import QMessageBox

        # Check for unsaved changes
        if self.urdf_builder.get_segment_count() > 0 and not self.current_file_path:
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
                if not self.current_file_path:  # Save was cancelled
                    event.ignore()
                    return
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

        event.accept()
        logger.info("URDF Generator window closed")


def main() -> None:
    """Main entry point for the URDF Generator."""
    from src.shared.python.logging_config import configure_gui_logging

    app = QApplication(sys.argv)
    app.setApplicationName("URDF Generator")
    app.setApplicationVersion("1.0.0")

    # Set up logging
    configure_gui_logging()

    window = URDFGeneratorWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
