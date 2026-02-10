"""URDF Editor Window - Integrated editor with all component manipulation features.

This module provides a comprehensive URDF editor that integrates:
- Component library with read-only protection
- Code editor with syntax highlighting
- Frankenstein mode for combining URDFs
- Chain manipulation tools
- End effector swap system
- Joint auto-loader and manipulation
- Mesh/STL browser and copy functionality
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QWidget,
)

from src.shared.python.logging_config import get_logger

from .chain_manipulation import ChainManipulationWidget
from .component_library import ComponentLibraryWidget
from .end_effector_manager import EndEffectorManagerWidget
from .frankenstein_editor import FrankensteinEditor
from .joint_manipulator import JointManipulatorWidget
from .mesh_browser import MeshBrowserWidget
from .urdf_code_editor import URDFCodeEditorWidget
from .visualization_widget import VisualizationWidget

logger = get_logger(__name__)


class URDFEditorWindow(QMainWindow):
    """Comprehensive URDF editor window with all editing features."""

    urdf_changed = pyqtSignal(str)  # Emitted when URDF content changes

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the URDF editor window."""
        super().__init__(parent)

    def _show_status(self, message: str) -> None:
        """Show a message in the status bar."""
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage(message)
        self.current_file: Path | None = None
        self.urdf_content: str = ""
        self._is_modified: bool = False

        self._setup_ui()
        self._setup_menu_bar()
        self._setup_dock_widgets()
        self._connect_signals()

        self.setWindowTitle("URDF Editor - Golf Modeling Suite")
        self.setMinimumSize(1400, 900)

        logger.info("URDF Editor window initialized")

    def _setup_ui(self) -> None:
        """Set up the main user interface."""
        # Main tab widget as central widget
        self.central_tabs = QTabWidget()
        self.central_tabs.setTabPosition(QTabWidget.TabPosition.South)
        self.setCentralWidget(self.central_tabs)

        # Tab 1: Code Editor
        self.code_editor = URDFCodeEditorWidget()
        self.central_tabs.addTab(self.code_editor, "Code Editor")

        # Tab 2: Frankenstein Mode
        self.frankenstein = FrankensteinEditor()
        self.central_tabs.addTab(self.frankenstein, "Frankenstein Mode")

        # Tab 3: Chain Manipulation
        self.chain_tools = ChainManipulationWidget()
        self.central_tabs.addTab(self.chain_tools, "Chain Tools")

        # Tab 4: End Effector Manager
        self.ee_manager = EndEffectorManagerWidget()
        self.central_tabs.addTab(self.ee_manager, "End Effectors")

        # Tab 5: Joint Manipulator
        self.joint_tools = JointManipulatorWidget()
        self.central_tabs.addTab(self.joint_tools, "Joints")

        # Tab 6: Mesh Browser
        self.mesh_browser = MeshBrowserWidget()
        self.central_tabs.addTab(self.mesh_browser, "Meshes")

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
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.new_urdf)
        file_menu.addAction(new_action)

        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_urdf)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_urdf)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self.save_urdf_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        close_action = QAction("&Close", self)
        close_action.setShortcut(QKeySequence.StandardKey.Close)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        if edit_menu is None:
            return

        validate_action = QAction("&Validate URDF", self)
        validate_action.setShortcut("Ctrl+Shift+V")
        validate_action.triggered.connect(self._validate_current)
        edit_menu.addAction(validate_action)

        format_action = QAction("&Format XML", self)
        format_action.setShortcut("Ctrl+Shift+F")
        format_action.triggered.connect(self._format_current)
        edit_menu.addAction(format_action)

        # View menu
        view_menu = menubar.addMenu("&View")
        if view_menu is None:
            return

        self.show_preview_action = QAction("Show &3D Preview", self)
        self.show_preview_action.setCheckable(True)
        self.show_preview_action.setChecked(True)
        self.show_preview_action.triggered.connect(self._toggle_preview)
        view_menu.addAction(self.show_preview_action)

        self.show_library_action = QAction("Show Component &Library", self)
        self.show_library_action.setCheckable(True)
        self.show_library_action.setChecked(True)
        self.show_library_action.triggered.connect(self._toggle_library)
        view_menu.addAction(self.show_library_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        if tools_menu is None:
            return

        frankenstein_action = QAction("&Frankenstein Mode", self)
        frankenstein_action.triggered.connect(
            lambda: self.central_tabs.setCurrentWidget(self.frankenstein)
        )
        tools_menu.addAction(frankenstein_action)

        chain_action = QAction("&Chain Tools", self)
        chain_action.triggered.connect(
            lambda: self.central_tabs.setCurrentWidget(self.chain_tools)
        )
        tools_menu.addAction(chain_action)

        ee_action = QAction("&End Effector Manager", self)
        ee_action.triggered.connect(
            lambda: self.central_tabs.setCurrentWidget(self.ee_manager)
        )
        tools_menu.addAction(ee_action)

        joint_action = QAction("&Joint Manipulator", self)
        joint_action.triggered.connect(
            lambda: self.central_tabs.setCurrentWidget(self.joint_tools)
        )
        tools_menu.addAction(joint_action)

        mesh_action = QAction("&Mesh Browser", self)
        mesh_action.triggered.connect(
            lambda: self.central_tabs.setCurrentWidget(self.mesh_browser)
        )
        tools_menu.addAction(mesh_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        if help_menu is None:
            return

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_dock_widgets(self) -> None:
        """Set up dock widgets for preview and library."""
        # 3D Preview dock
        self.preview_dock = QDockWidget("3D Preview", self)
        self.preview_dock.setObjectName("PreviewDock")
        self.visualization = VisualizationWidget()
        self.preview_dock.setWidget(self.visualization)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.preview_dock)

        # Component Library dock
        self.library_dock = QDockWidget("Component Library", self)
        self.library_dock.setObjectName("LibraryDock")
        self.component_library = ComponentLibraryWidget()
        self.library_dock.setWidget(self.component_library)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.library_dock)

    def _connect_signals(self) -> None:
        """Connect all signals."""
        # Code editor signals
        self.code_editor.editor.content_changed.connect(self._on_content_changed)
        self.code_editor.validation_changed.connect(self._on_validation_changed)

        # Frankenstein signals
        self.frankenstein.right_panel.tree.itemSelectionChanged.connect(
            self._on_frankenstein_update
        )

        # Chain manipulation signals
        self.chain_tools.chain_modified.connect(self._on_urdf_modified)

        # End effector signals
        self.ee_manager.urdf_modified.connect(self._on_urdf_modified)

        # Joint manipulator signals
        self.joint_tools.urdf_modified.connect(self._on_urdf_modified)
        self.joint_tools.joints_updated.connect(self._on_joints_updated)

        # Mesh browser signals
        self.mesh_browser.urdf_modified.connect(self._on_urdf_modified)

        # Component library signals
        self.component_library.component_edited.connect(
            self._on_component_edit_requested
        )

        # Tab change
        self.central_tabs.currentChanged.connect(self._on_tab_changed)

    def _on_content_changed(self, content: str) -> None:
        """Handle code editor content change."""
        self.urdf_content = content
        self._is_modified = True
        self._update_title()
        self._update_preview()

    def _on_urdf_modified(self, content: str) -> None:
        """Handle URDF modification from any tool."""
        self.urdf_content = content
        self._is_modified = True
        self._update_title()

        # Update code editor
        self.code_editor.set_content(content)

        # Update preview
        self._update_preview()

        # Update other tools with new content
        self._sync_tools_with_content()

    def _on_validation_changed(self, is_valid: bool, errors: list[str]) -> None:
        """Handle validation status change."""
        if is_valid:
            self._show_status("URDF is valid")
        else:
            self._show_status(f"Validation errors: {len(errors)}")

    def _on_frankenstein_update(self) -> None:
        """Handle Frankenstein editor update."""
        model = self.frankenstein.get_working_model()
        if model:
            content = model.to_xml()
            self._on_urdf_modified(content)

    def _on_joints_updated(self, positions: dict[str, float]) -> None:
        """Handle joint position updates for preview."""
        # This could update the 3D preview with new joint positions

    def _on_component_edit_requested(self, component: Any) -> None:
        """Handle request to edit a component."""
        # Switch to code editor and highlight the component
        self.central_tabs.setCurrentWidget(self.code_editor)
        # Could search for and highlight the component in the code

    def _on_tab_changed(self, index: int) -> None:
        """Handle tab change."""
        current_widget = self.central_tabs.currentWidget()

        # Sync content to the new tab
        if current_widget == self.chain_tools and self.urdf_content:
            self.chain_tools.load_urdf(self.urdf_content)
        elif current_widget == self.ee_manager and self.urdf_content:
            self.ee_manager.load_urdf(self.urdf_content)
        elif current_widget == self.joint_tools and self.urdf_content:
            self.joint_tools.load_urdf(self.urdf_content)
        elif current_widget == self.mesh_browser and self.urdf_content:
            self.mesh_browser.load_target_content(self.urdf_content, self.current_file)

    def _sync_tools_with_content(self) -> None:
        """Synchronize all tools with current URDF content."""
        # Only sync the currently visible tool to avoid overhead
        current_widget = self.central_tabs.currentWidget()

        if current_widget == self.chain_tools:
            self.chain_tools.load_urdf(self.urdf_content)
        elif current_widget == self.ee_manager:
            self.ee_manager.load_urdf(self.urdf_content)
        elif current_widget == self.joint_tools:
            self.joint_tools.load_urdf(self.urdf_content)

    def _update_preview(self) -> None:
        """Update the 3D preview."""
        if self.urdf_content:
            file_path = str(self.current_file) if self.current_file else None
            self.visualization.update_visualization(self.urdf_content, file_path)

    def _update_title(self) -> None:
        """Update the window title."""
        title = "URDF Editor - Golf Modeling Suite"
        if self.current_file:
            title = f"{self.current_file.name} - {title}"
        if self._is_modified:
            title = f"*{title}"
        self.setWindowTitle(title)

    def _validate_current(self) -> None:
        """Validate the current URDF."""
        if self.central_tabs.currentWidget() == self.code_editor:
            self.code_editor.editor.validate_urdf()

    def _format_current(self) -> None:
        """Format the current URDF."""
        if self.central_tabs.currentWidget() == self.code_editor:
            self.code_editor._on_format()

    def _toggle_preview(self) -> None:
        """Toggle 3D preview visibility."""
        self.preview_dock.setVisible(self.show_preview_action.isChecked())

    def _toggle_library(self) -> None:
        """Toggle component library visibility."""
        self.library_dock.setVisible(self.show_library_action.isChecked())

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About URDF Editor",
            "URDF Editor - Golf Modeling Suite\n\n"
            "Features:\n"
            "- Code editor with syntax highlighting\n"
            "- Frankenstein mode for combining URDFs\n"
            "- Chain manipulation tools\n"
            "- End effector swap system\n"
            "- Joint auto-loader and manipulation\n"
            "- Mesh/STL browser\n\n"
            "Components from library URDFs are read-only.\n"
            "Use 'Copy to Working Set' to edit them.",
        )

    def new_urdf(self) -> None:
        """Create a new URDF."""
        if self._is_modified:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "Save changes before creating new file?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )

            if reply == QMessageBox.StandardButton.Save:
                self.save_urdf()
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        # Create minimal URDF
        self.urdf_content = """<?xml version="1.0"?>
<robot name="new_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>
"""

        self.current_file = None
        self._is_modified = False
        self.code_editor.set_content(self.urdf_content)
        self._update_title()
        self._update_preview()
        self._show_status("New URDF created")

    def open_urdf(self) -> None:
        """Open a URDF file."""
        if self._is_modified:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "Save changes before opening another file?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )

            if reply == QMessageBox.StandardButton.Save:
                self.save_urdf()
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open URDF",
            "",
            "URDF Files (*.urdf);;XML Files (*.xml);;All Files (*)",
        )

        if file_path:
            self.load_file(Path(file_path))

    def load_file(self, file_path: Path) -> bool:
        """Load a URDF file."""
        try:
            self.urdf_content = file_path.read_text(encoding="utf-8")
            self.current_file = file_path
            self._is_modified = False

            self.code_editor.set_content(self.urdf_content, str(file_path))
            self._update_title()
            self._update_preview()

            # Add to library for reference
            self.component_library.load_urdf_to_library(file_path)

            self._show_status(f"Loaded: {file_path.name}")
            logger.info(f"Loaded URDF: {file_path}")
            return True

        except (RuntimeError, ValueError, OSError) as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {e}")
            logger.error(f"Failed to load URDF: {e}")
            return False

    def save_urdf(self) -> None:
        """Save the current URDF."""
        if self.current_file:
            self._save_to_file(self.current_file)
        else:
            self.save_urdf_as()

    def save_urdf_as(self) -> None:
        """Save the current URDF with a new filename."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save URDF",
            "robot.urdf",
            "URDF Files (*.urdf);;XML Files (*.xml)",
        )

        if file_path:
            self._save_to_file(Path(file_path))

    def _save_to_file(self, file_path: Path) -> None:
        """Save URDF to file."""
        try:
            # Get content from code editor (most up-to-date)
            content = self.code_editor.get_content()

            file_path.write_text(content, encoding="utf-8")
            self.current_file = file_path
            self.urdf_content = content
            self._is_modified = False

            self._update_title()
            self._show_status(f"Saved: {file_path.name}")
            logger.info(f"Saved URDF: {file_path}")

        except ImportError as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {e}")
            logger.error(f"Failed to save URDF: {e}")

    def closeEvent(self, event: Any) -> None:
        """Handle window close event."""
        if self._is_modified:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "Save changes before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )

            if reply == QMessageBox.StandardButton.Save:
                self.save_urdf()
                if self._is_modified:  # Save was cancelled
                    event.ignore()
                    return
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

        event.accept()
        logger.info("URDF Editor window closed")


def launch_urdf_editor() -> None:
    """Launch the URDF editor as a standalone application."""
    import sys

    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("URDF Editor")

    window = URDFEditorWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    launch_urdf_editor()
