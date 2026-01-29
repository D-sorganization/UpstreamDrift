"""Mesh Browser - Browse and copy mesh/STL components between URDFs.

Provides a side-by-side interface for browsing mesh files referenced
in URDFs and copying mesh references between models.
"""

from __future__ import annotations

import copy
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MeshReference:
    """Information about a mesh reference in a URDF."""

    filename: str
    absolute_path: Path | None
    link_name: str
    context: str  # 'visual' or 'collision'
    scale: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    exists: bool = False
    file_size: int = 0

    @classmethod
    def from_element(
        cls,
        mesh_elem: ET.Element,
        link_name: str,
        context: str,
        urdf_dir: Path | None,
    ) -> "MeshReference":
        """Create MeshReference from a mesh XML element.

        Args:
            mesh_elem: The <mesh> XML element
            link_name: Name of the parent link
            context: 'visual' or 'collision'
            urdf_dir: Directory containing the URDF file
        """
        filename = mesh_elem.get("filename", "")

        # Parse scale
        scale_str = mesh_elem.get("scale", "1 1 1")
        try:
            scale = tuple(float(x) for x in scale_str.split())
            if len(scale) != 3:
                scale = (1.0, 1.0, 1.0)
        except ValueError:
            scale = (1.0, 1.0, 1.0)

        # Get origin from parent geometry element's sibling
        origin_xyz = (0.0, 0.0, 0.0)
        origin_rpy = (0.0, 0.0, 0.0)

        # Resolve absolute path
        absolute_path = None
        exists = False
        file_size = 0

        if urdf_dir and filename:
            # Handle package:// URIs
            if filename.startswith("package://"):
                # Strip package:// prefix, path is relative to package
                rel_path = filename.replace("package://", "").split("/", 1)
                if len(rel_path) > 1:
                    filename = rel_path[1]

            # Handle file:// URIs
            elif filename.startswith("file://"):
                filename = filename.replace("file://", "")

            # Resolve relative paths
            if not Path(filename).is_absolute():
                absolute_path = urdf_dir / filename
            else:
                absolute_path = Path(filename)

            if absolute_path.exists():
                exists = True
                file_size = absolute_path.stat().st_size

        return cls(
            filename=filename,
            absolute_path=absolute_path,
            link_name=link_name,
            context=context,
            scale=scale,  # type: ignore
            origin_xyz=origin_xyz,  # type: ignore
            origin_rpy=origin_rpy,  # type: ignore
            exists=exists,
            file_size=file_size,
        )

    def get_file_extension(self) -> str:
        """Get the file extension."""
        return Path(self.filename).suffix.lower()

    def get_display_name(self) -> str:
        """Get a display name for the mesh."""
        return Path(self.filename).name

    def format_size(self) -> str:
        """Format file size for display."""
        if self.file_size < 1024:
            return f"{self.file_size} B"
        elif self.file_size < 1024 * 1024:
            return f"{self.file_size / 1024:.1f} KB"
        else:
            return f"{self.file_size / (1024 * 1024):.1f} MB"


class MeshExtractor:
    """Extracts mesh information from URDF files."""

    @staticmethod
    def extract_meshes(
        urdf_content: str, urdf_path: Path | None = None
    ) -> list[MeshReference]:
        """Extract all mesh references from a URDF.

        Args:
            urdf_content: URDF XML content
            urdf_path: Path to URDF file for resolving relative paths

        Returns:
            List of MeshReference objects
        """
        try:
            root = ET.fromstring(urdf_content)
        except ET.ParseError:
            return []

        urdf_dir = urdf_path.parent if urdf_path else None
        meshes = []

        for link in root.findall("link"):
            link_name = link.get("name", "unnamed")

            # Visual meshes
            for visual in link.findall("visual"):
                geometry = visual.find("geometry")
                if geometry is not None:
                    mesh = geometry.find("mesh")
                    if mesh is not None:
                        ref = MeshReference.from_element(
                            mesh, link_name, "visual", urdf_dir
                        )
                        meshes.append(ref)

            # Collision meshes
            for collision in link.findall("collision"):
                geometry = collision.find("geometry")
                if geometry is not None:
                    mesh = geometry.find("mesh")
                    if mesh is not None:
                        ref = MeshReference.from_element(
                            mesh, link_name, "collision", urdf_dir
                        )
                        meshes.append(ref)

        return meshes

    @staticmethod
    def get_unique_mesh_files(meshes: list[MeshReference]) -> list[MeshReference]:
        """Get unique mesh files (deduplicated by filename)."""
        seen = set()
        unique = []
        for mesh in meshes:
            if mesh.filename not in seen:
                seen.add(mesh.filename)
                unique.append(mesh)
        return unique


class MeshBrowserPanel(QWidget):
    """Panel for browsing meshes in a single URDF."""

    mesh_selected = pyqtSignal(object)  # MeshReference
    mesh_double_clicked = pyqtSignal(object)  # For copying

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """Initialize the mesh browser panel."""
        super().__init__(parent)
        self.title = title
        self.meshes: list[MeshReference] = []
        self.urdf_path: Path | None = None
        self.urdf_content: str = ""
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Header
        header = QHBoxLayout()
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("font-weight: bold;")
        header.addWidget(self.title_label)

        self.load_btn = QPushButton("Load URDF")
        header.addWidget(self.load_btn)
        layout.addLayout(header)

        # File info
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: gray;")
        layout.addWidget(self.file_label)

        # Mesh table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Mesh File", "Link", "Type", "Size", "Status"
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        header = self.table.horizontalHeader()
        if header:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            for i in range(1, 5):
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self.table)

        # Summary
        summary_layout = QHBoxLayout()
        self.summary_label = QLabel("No meshes")
        summary_layout.addWidget(self.summary_label)

        self.missing_label = QLabel("")
        self.missing_label.setStyleSheet("color: red;")
        summary_layout.addWidget(self.missing_label)

        layout.addLayout(summary_layout)

    def _connect_signals(self) -> None:
        """Connect signals."""
        self.load_btn.clicked.connect(self._on_load)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.cellDoubleClicked.connect(self._on_double_click)

    def _on_load(self) -> None:
        """Handle load button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load URDF",
            "",
            "URDF Files (*.urdf);;XML Files (*.xml)",
        )

        if file_path:
            self.load_file(Path(file_path))

    def load_file(self, file_path: Path) -> bool:
        """Load a URDF file."""
        try:
            self.urdf_content = file_path.read_text(encoding="utf-8")
            self.urdf_path = file_path
            self.meshes = MeshExtractor.extract_meshes(self.urdf_content, file_path)
            self._populate_table()
            self.file_label.setText(f"File: {file_path.name}")
            logger.info(f"Loaded {len(self.meshes)} meshes from {file_path}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {e}")
            return False

    def load_content(self, content: str, file_path: Path | None = None) -> None:
        """Load URDF content directly."""
        self.urdf_content = content
        self.urdf_path = file_path
        self.meshes = MeshExtractor.extract_meshes(content, file_path)
        self._populate_table()
        if file_path:
            self.file_label.setText(f"File: {file_path.name}")
        else:
            self.file_label.setText("Content loaded (no file)")

    def _populate_table(self) -> None:
        """Populate the mesh table."""
        self.table.setRowCount(len(self.meshes))

        missing_count = 0
        for row, mesh in enumerate(self.meshes):
            # Filename
            self.table.setItem(row, 0, QTableWidgetItem(mesh.get_display_name()))

            # Link
            self.table.setItem(row, 1, QTableWidgetItem(mesh.link_name))

            # Type (visual/collision)
            type_item = QTableWidgetItem(mesh.context)
            if mesh.context == "visual":
                type_item.setForeground(QColor("#006400"))
            else:
                type_item.setForeground(QColor("#0000CD"))
            self.table.setItem(row, 2, type_item)

            # Size
            size_text = mesh.format_size() if mesh.exists else "-"
            self.table.setItem(row, 3, QTableWidgetItem(size_text))

            # Status
            if mesh.exists:
                status_item = QTableWidgetItem("Found")
                status_item.setForeground(QColor("#006400"))
            else:
                status_item = QTableWidgetItem("Missing")
                status_item.setForeground(QColor("#FF0000"))
                missing_count += 1
            self.table.setItem(row, 4, status_item)

        # Update summary
        unique = len(MeshExtractor.get_unique_mesh_files(self.meshes))
        self.summary_label.setText(
            f"Total references: {len(self.meshes)} | Unique files: {unique}"
        )

        if missing_count > 0:
            self.missing_label.setText(f"Missing: {missing_count}")
        else:
            self.missing_label.setText("")

    def _on_selection_changed(self) -> None:
        """Handle table selection change."""
        row = self.table.currentRow()
        if 0 <= row < len(self.meshes):
            self.mesh_selected.emit(self.meshes[row])

    def _on_double_click(self, row: int, column: int) -> None:
        """Handle double-click for copying."""
        if 0 <= row < len(self.meshes):
            self.mesh_double_clicked.emit(self.meshes[row])

    def get_selected_mesh(self) -> MeshReference | None:
        """Get the currently selected mesh."""
        row = self.table.currentRow()
        if 0 <= row < len(self.meshes):
            return self.meshes[row]
        return None

    def get_meshes(self) -> list[MeshReference]:
        """Get all meshes."""
        return self.meshes

    def get_urdf_content(self) -> str:
        """Get the URDF content."""
        return self.urdf_content

    def get_urdf_path(self) -> Path | None:
        """Get the URDF file path."""
        return self.urdf_path


class CopyMeshDialog(QDialog):
    """Dialog for configuring mesh copy operation."""

    def __init__(
        self,
        mesh: MeshReference,
        target_links: list[str],
        target_urdf_dir: Path | None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the dialog."""
        super().__init__(parent)
        self.mesh = mesh
        self.target_urdf_dir = target_urdf_dir
        self.setWindowTitle("Copy Mesh Reference")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Source info
        source_group = QGroupBox("Source Mesh")
        source_layout = QFormLayout(source_group)
        source_layout.addRow("File:", QLabel(mesh.get_display_name()))
        source_layout.addRow("From link:", QLabel(mesh.link_name))
        source_layout.addRow("Type:", QLabel(mesh.context))

        status = "Found" if mesh.exists else "Missing (file will need to be copied)"
        status_label = QLabel(status)
        status_label.setStyleSheet(
            "color: green;" if mesh.exists else "color: orange;"
        )
        source_layout.addRow("Status:", status_label)

        layout.addWidget(source_group)

        # Target configuration
        target_group = QGroupBox("Target Configuration")
        target_layout = QFormLayout(target_group)

        self.link_combo = QComboBox()
        self.link_combo.addItems(target_links)
        target_layout.addRow("Add to link:", self.link_combo)

        self.context_combo = QComboBox()
        self.context_combo.addItems(["visual", "collision", "both"])
        self.context_combo.setCurrentText(mesh.context)
        target_layout.addRow("As:", self.context_combo)

        layout.addWidget(target_group)

        # File handling
        if mesh.exists and target_urdf_dir:
            file_group = QGroupBox("File Handling")
            file_layout = QVBoxLayout(file_group)

            self.copy_file_check = QCheckBox("Copy mesh file to target directory")
            self.copy_file_check.setChecked(True)
            file_layout.addWidget(self.copy_file_check)

            self.mesh_subdir_edit = QLineEdit("meshes")
            form = QFormLayout()
            form.addRow("Mesh subdirectory:", self.mesh_subdir_edit)
            file_layout.addLayout(form)

            layout.addWidget(file_group)
        else:
            self.copy_file_check = None
            self.mesh_subdir_edit = None

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_configuration(self) -> dict[str, Any]:
        """Get the copy configuration."""
        return {
            "target_link": self.link_combo.currentText(),
            "context": self.context_combo.currentText(),
            "copy_file": self.copy_file_check.isChecked() if self.copy_file_check else False,
            "mesh_subdir": self.mesh_subdir_edit.text() if self.mesh_subdir_edit else "meshes",
        }


class MeshBrowserWidget(QWidget):
    """Side-by-side mesh browser for copying between URDFs."""

    urdf_modified = pyqtSignal(str)  # New URDF content

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the mesh browser widget."""
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel(
            "Browse meshes in two URDFs side-by-side. "
            "Double-click or use the Copy button to copy mesh references."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(instructions)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.left_panel = MeshBrowserPanel("Source URDF")
        self.right_panel = MeshBrowserPanel("Target URDF")

        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)

        layout.addWidget(splitter)

        # Controls
        controls = QHBoxLayout()
        controls.addStretch()

        self.copy_btn = QPushButton("Copy Selected Mesh -->")
        self.copy_all_btn = QPushButton("Copy All Meshes -->")
        self.swap_btn = QPushButton("Swap Models")

        controls.addWidget(self.copy_btn)
        controls.addWidget(self.copy_all_btn)
        controls.addWidget(self.swap_btn)
        controls.addStretch()

        layout.addLayout(controls)

        # Preview
        preview_group = QGroupBox("Selected Mesh Details")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(100)
        preview_layout.addWidget(self.preview_text)
        layout.addWidget(preview_group)

        # Status
        self.status_label = QLabel("Load URDFs to browse meshes")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)

    def _connect_signals(self) -> None:
        """Connect signals."""
        self.copy_btn.clicked.connect(self._on_copy_selected)
        self.copy_all_btn.clicked.connect(self._on_copy_all)
        self.swap_btn.clicked.connect(self._on_swap)

        self.left_panel.mesh_selected.connect(self._on_mesh_selected)
        self.left_panel.mesh_double_clicked.connect(self._on_copy_mesh)
        self.right_panel.mesh_selected.connect(self._on_mesh_selected)

    def _on_mesh_selected(self, mesh: MeshReference) -> None:
        """Handle mesh selection."""
        details = f"File: {mesh.filename}\n"
        details += f"Link: {mesh.link_name}\n"
        details += f"Type: {mesh.context}\n"
        details += f"Scale: ({mesh.scale[0]:.2f}, {mesh.scale[1]:.2f}, {mesh.scale[2]:.2f})\n"
        details += f"Exists: {'Yes' if mesh.exists else 'No'}\n"
        if mesh.exists:
            details += f"Size: {mesh.format_size()}\n"
            details += f"Path: {mesh.absolute_path}"
        self.preview_text.setPlainText(details)

    def _on_copy_selected(self) -> None:
        """Copy the selected mesh from left to right."""
        mesh = self.left_panel.get_selected_mesh()
        if mesh:
            self._on_copy_mesh(mesh)
        else:
            self.status_label.setText("Select a mesh in the source panel")

    def _on_copy_mesh(self, mesh: MeshReference) -> None:
        """Copy a mesh reference to the target URDF."""
        target_content = self.right_panel.get_urdf_content()
        if not target_content:
            self.status_label.setText("Load a target URDF first")
            return

        try:
            target_root = ET.fromstring(target_content)
        except ET.ParseError:
            self.status_label.setText("Invalid target URDF")
            return

        # Get target links
        target_links = [link.get("name", "") for link in target_root.findall("link")]
        if not target_links:
            self.status_label.setText("No links in target URDF")
            return

        # Show configuration dialog
        dialog = CopyMeshDialog(
            mesh, target_links, self.right_panel.get_urdf_path(), self
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        config = dialog.get_configuration()
        self._apply_mesh_copy(mesh, config)

    def _apply_mesh_copy(self, mesh: MeshReference, config: dict[str, Any]) -> None:
        """Apply the mesh copy to the target URDF."""
        target_content = self.right_panel.get_urdf_content()
        target_path = self.right_panel.get_urdf_path()

        try:
            root = ET.fromstring(target_content)
        except ET.ParseError:
            return

        # Find target link
        target_link = None
        for link in root.findall("link"):
            if link.get("name") == config["target_link"]:
                target_link = link
                break

        if target_link is None:
            self.status_label.setText(f"Link '{config['target_link']}' not found")
            return

        # Copy mesh file if requested
        new_filename = mesh.filename
        if config["copy_file"] and mesh.exists and mesh.absolute_path and target_path:
            mesh_subdir = config["mesh_subdir"]
            target_mesh_dir = target_path.parent / mesh_subdir
            target_mesh_dir.mkdir(parents=True, exist_ok=True)

            target_mesh_path = target_mesh_dir / mesh.absolute_path.name
            try:
                shutil.copy2(mesh.absolute_path, target_mesh_path)
                new_filename = f"{mesh_subdir}/{mesh.absolute_path.name}"
                logger.info(f"Copied mesh file to {target_mesh_path}")
            except Exception as e:
                logger.error(f"Failed to copy mesh file: {e}")

        # Create mesh element(s)
        contexts = []
        if config["context"] == "both":
            contexts = ["visual", "collision"]
        else:
            contexts = [config["context"]]

        for context in contexts:
            # Create or find the context element (visual/collision)
            context_elem = ET.SubElement(target_link, context)

            # Add origin
            origin = ET.SubElement(context_elem, "origin")
            origin.set("xyz", f"{mesh.origin_xyz[0]} {mesh.origin_xyz[1]} {mesh.origin_xyz[2]}")
            origin.set("rpy", f"{mesh.origin_rpy[0]} {mesh.origin_rpy[1]} {mesh.origin_rpy[2]}")

            # Add geometry with mesh
            geometry = ET.SubElement(context_elem, "geometry")
            mesh_elem = ET.SubElement(geometry, "mesh")
            mesh_elem.set("filename", new_filename)

            if mesh.scale != (1.0, 1.0, 1.0):
                mesh_elem.set("scale", f"{mesh.scale[0]} {mesh.scale[1]} {mesh.scale[2]}")

        # Generate new URDF
        ET.indent(root, space="  ")
        new_content = ET.tostring(root, encoding="unicode", xml_declaration=True)

        # Reload right panel
        self.right_panel.load_content(new_content, target_path)
        self.urdf_modified.emit(new_content)
        self.status_label.setText(
            f"Added mesh '{mesh.get_display_name()}' to link '{config['target_link']}'"
        )

    def _on_copy_all(self) -> None:
        """Copy all meshes from source to target."""
        meshes = self.left_panel.get_meshes()
        if not meshes:
            self.status_label.setText("No meshes in source URDF")
            return

        target_content = self.right_panel.get_urdf_content()
        if not target_content:
            self.status_label.setText("Load a target URDF first")
            return

        reply = QMessageBox.question(
            self,
            "Copy All Meshes",
            f"Copy {len(meshes)} mesh reference(s) to target?\n"
            "Each mesh will be added to a link with the same name if it exists.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Get unique meshes
        unique_meshes = MeshExtractor.get_unique_mesh_files(meshes)

        try:
            root = ET.fromstring(target_content)
        except ET.ParseError:
            return

        target_links = {link.get("name"): link for link in root.findall("link")}
        copied_count = 0

        for mesh in unique_meshes:
            # Try to find matching link in target
            if mesh.link_name in target_links:
                target_link = target_links[mesh.link_name]

                # Create visual/collision element
                context_elem = ET.SubElement(target_link, mesh.context)
                geometry = ET.SubElement(context_elem, "geometry")
                mesh_elem = ET.SubElement(geometry, "mesh")
                mesh_elem.set("filename", mesh.filename)

                if mesh.scale != (1.0, 1.0, 1.0):
                    mesh_elem.set("scale", f"{mesh.scale[0]} {mesh.scale[1]} {mesh.scale[2]}")

                copied_count += 1

        # Generate new URDF
        ET.indent(root, space="  ")
        new_content = ET.tostring(root, encoding="unicode", xml_declaration=True)

        self.right_panel.load_content(new_content, self.right_panel.get_urdf_path())
        self.urdf_modified.emit(new_content)
        self.status_label.setText(f"Copied {copied_count} of {len(unique_meshes)} meshes")

    def _on_swap(self) -> None:
        """Swap the source and target models."""
        left_content = self.left_panel.get_urdf_content()
        left_path = self.left_panel.get_urdf_path()
        right_content = self.right_panel.get_urdf_content()
        right_path = self.right_panel.get_urdf_path()

        if right_content:
            self.left_panel.load_content(right_content, right_path)
        if left_content:
            self.right_panel.load_content(left_content, left_path)

        self.status_label.setText("Swapped source and target models")

    def load_source(self, file_path: Path) -> bool:
        """Load a source URDF."""
        return self.left_panel.load_file(file_path)

    def load_target(self, file_path: Path) -> bool:
        """Load a target URDF."""
        return self.right_panel.load_file(file_path)

    def load_source_content(self, content: str, file_path: Path | None = None) -> None:
        """Load source URDF from content."""
        self.left_panel.load_content(content, file_path)

    def load_target_content(self, content: str, file_path: Path | None = None) -> None:
        """Load target URDF from content."""
        self.right_panel.load_content(content, file_path)

    def get_target_content(self) -> str:
        """Get the modified target URDF content."""
        return self.right_panel.get_urdf_content()
