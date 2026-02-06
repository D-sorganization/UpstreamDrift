"""Component Library for URDF parts with read-only protection and copy-to-edit.

This module provides a library system where components from existing URDFs
are read-only but can be copied and edited. This prevents corruption of
source files while enabling reuse and modification.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as ET
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


class ComponentType(Enum):
    """Types of URDF components."""

    LINK = "link"
    JOINT = "joint"
    MATERIAL = "material"
    TRANSMISSION = "transmission"
    GAZEBO = "gazebo"
    SENSOR = "sensor"


@dataclass
class URDFComponent:
    """Represents a single URDF component (link, joint, etc.)."""

    component_type: ComponentType
    name: str
    xml_content: str
    source_file: Path | None = None
    is_library: bool = False  # True if from library (read-only)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_read_only(self) -> bool:
        """Check if component is read-only (library component)."""
        return self.is_library

    def get_hash(self) -> str:
        """Get unique hash for this component."""
        return hashlib.md5(self.xml_content.encode()).hexdigest()[:8]

    def to_dict(self) -> dict[str, Any]:
        """Convert component to dictionary."""
        return {
            "type": self.component_type.value,
            "name": self.name,
            "xml_content": self.xml_content,
            "source_file": str(self.source_file) if self.source_file else None,
            "is_library": self.is_library,
            "metadata": self.metadata,
        }

    @classmethod
    def from_xml_element(
        cls,
        element: ET.Element,
        source_file: Path | None = None,
        is_library: bool = False,
    ) -> URDFComponent:
        """Create component from XML element."""
        tag_to_type = {
            "link": ComponentType.LINK,
            "joint": ComponentType.JOINT,
            "material": ComponentType.MATERIAL,
            "transmission": ComponentType.TRANSMISSION,
            "gazebo": ComponentType.GAZEBO,
            "sensor": ComponentType.SENSOR,
        }

        component_type = tag_to_type.get(element.tag, ComponentType.LINK)
        name = element.get("name", f"unnamed_{element.tag}")

        # Convert element to string
        xml_content = ET.tostring(element, encoding="unicode")

        # Extract metadata
        metadata: dict[str, Any] = {}
        if component_type == ComponentType.LINK:
            # Extract geometry info
            visual = element.find("visual/geometry")
            if visual is not None:
                for geom in visual:
                    metadata["geometry_type"] = geom.tag
                    break
            # Extract mass
            mass_elem = element.find("inertial/mass")
            if mass_elem is not None:
                metadata["mass"] = mass_elem.get("value", "unknown")

        elif component_type == ComponentType.JOINT:
            metadata["joint_type"] = element.get("type", "unknown")
            parent = element.find("parent")
            child = element.find("child")
            if parent is not None:
                metadata["parent"] = parent.get("link", "")
            if child is not None:
                metadata["child"] = child.get("link", "")

        return cls(
            component_type=component_type,
            name=name,
            xml_content=xml_content,
            source_file=source_file,
            is_library=is_library,
            metadata=metadata,
        )


class ComponentLibrary:
    """Manages a library of URDF components with read-only protection."""

    def __init__(self) -> None:
        """Initialize the component library."""
        self._library_components: dict[str, URDFComponent] = {}
        self._working_components: dict[str, URDFComponent] = {}
        self._source_files: set[Path] = set()

    def load_urdf_as_library(self, urdf_path: Path) -> list[URDFComponent]:
        """Load a URDF file as read-only library components.

        Args:
            urdf_path: Path to URDF file

        Returns:
            List of loaded components
        """
        if not urdf_path.exists():
            logger.error(f"URDF file not found: {urdf_path}")
            return []

        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error(f"Failed to parse URDF: {e}")
            return []

        components = []
        self._source_files.add(urdf_path)

        # Extract all components
        for tag in ["link", "joint", "material", "transmission", "gazebo", "sensor"]:
            for element in root.findall(tag):
                component = URDFComponent.from_xml_element(
                    element, source_file=urdf_path, is_library=True
                )
                key = f"{urdf_path.stem}:{component.name}"
                self._library_components[key] = component
                components.append(component)

        logger.info(f"Loaded {len(components)} components from {urdf_path.name}")
        return components

    def load_urdf_as_working(self, urdf_path: Path) -> list[URDFComponent]:
        """Load a URDF file as editable working components.

        Args:
            urdf_path: Path to URDF file

        Returns:
            List of loaded components
        """
        if not urdf_path.exists():
            logger.error(f"URDF file not found: {urdf_path}")
            return []

        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error(f"Failed to parse URDF: {e}")
            return []

        components = []

        # Extract all components as editable
        for tag in ["link", "joint", "material", "transmission", "gazebo", "sensor"]:
            for element in root.findall(tag):
                component = URDFComponent.from_xml_element(
                    element, source_file=urdf_path, is_library=False
                )
                self._working_components[component.name] = component
                components.append(component)

        logger.info(
            f"Loaded {len(components)} working components from {urdf_path.name}"
        )
        return components

    def copy_to_working(
        self, library_key: str, new_name: str | None = None
    ) -> URDFComponent | None:
        """Copy a library component to working set for editing.

        Args:
            library_key: Key of library component
            new_name: Optional new name for the copy

        Returns:
            The new editable component, or None if not found
        """
        if library_key not in self._library_components:
            logger.error(f"Library component not found: {library_key}")
            return None

        original = self._library_components[library_key]

        # Create a deep copy
        new_component = copy.deepcopy(original)
        new_component.is_library = False

        # Update name if provided
        if new_name:
            new_component.name = new_name
            # Update XML content with new name
            new_component.xml_content = new_component.xml_content.replace(
                f'name="{original.name}"', f'name="{new_name}"', 1
            )

        # Add to working components
        self._working_components[new_component.name] = new_component

        logger.info(
            f"Copied library component '{library_key}' as '{new_component.name}'"
        )
        return new_component

    def get_library_components(
        self, filter_type: ComponentType | None = None
    ) -> list[URDFComponent]:
        """Get all library components, optionally filtered by type.

        Args:
            filter_type: Optional component type filter

        Returns:
            List of library components
        """
        components = list(self._library_components.values())
        if filter_type:
            components = [c for c in components if c.component_type == filter_type]
        return components

    def get_working_components(
        self, filter_type: ComponentType | None = None
    ) -> list[URDFComponent]:
        """Get all working components, optionally filtered by type.

        Args:
            filter_type: Optional component type filter

        Returns:
            List of working components
        """
        components = list(self._working_components.values())
        if filter_type:
            components = [c for c in components if c.component_type == filter_type]
        return components

    def get_component(
        self, name: str, from_library: bool = False
    ) -> URDFComponent | None:
        """Get a component by name.

        Args:
            name: Component name
            from_library: If True, search library; otherwise search working

        Returns:
            Component if found, None otherwise
        """
        if from_library:
            # Search by name in library (could be multiple files)
            for comp in self._library_components.values():
                if comp.name == name:
                    return comp
            return None
        return self._working_components.get(name)

    def update_working_component(self, name: str, xml_content: str) -> bool:
        """Update XML content of a working component.

        Args:
            name: Component name
            xml_content: New XML content

        Returns:
            True if updated successfully
        """
        if name not in self._working_components:
            logger.error(f"Working component not found: {name}")
            return False

        component = self._working_components[name]
        if component.is_read_only:
            logger.error(f"Cannot modify read-only component: {name}")
            return False

        component.xml_content = xml_content
        logger.info(f"Updated working component: {name}")
        return True

    def remove_working_component(self, name: str) -> bool:
        """Remove a component from working set.

        Args:
            name: Component name

        Returns:
            True if removed successfully
        """
        if name in self._working_components:
            del self._working_components[name]
            logger.info(f"Removed working component: {name}")
            return True
        return False

    def export_working_to_urdf(self, robot_name: str = "robot") -> str:
        """Export working components to URDF XML.

        Args:
            robot_name: Name for the robot element

        Returns:
            URDF XML string
        """
        root = ET.Element("robot", name=robot_name)

        # Sort components: materials first, then links, then joints
        order = {
            ComponentType.MATERIAL: 0,
            ComponentType.LINK: 1,
            ComponentType.JOINT: 2,
            ComponentType.TRANSMISSION: 3,
            ComponentType.GAZEBO: 4,
            ComponentType.SENSOR: 5,
        }

        sorted_components = sorted(
            self._working_components.values(),
            key=lambda c: order.get(c.component_type, 99),
        )

        for component in sorted_components:
            try:
                element = ET.fromstring(component.xml_content)
                root.append(element)
            except ET.ParseError as e:
                logger.error(f"Failed to parse component {component.name}: {e}")

        # Pretty print
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def clear_working(self) -> None:
        """Clear all working components."""
        self._working_components.clear()
        logger.info("Cleared working components")

    def clear_library(self) -> None:
        """Clear all library components."""
        self._library_components.clear()
        self._source_files.clear()
        logger.info("Cleared library components")

    def get_source_files(self) -> list[Path]:
        """Get list of loaded source files."""
        return list(self._source_files)

    def get_library_items(self) -> dict[str, URDFComponent]:
        """Get all library components as a dictionary (key -> component).

        Returns:
            Dictionary mapping keys to library components.
        """
        return dict(self._library_components)

    def get_working_items(self) -> dict[str, URDFComponent]:
        """Get all working components as a dictionary (name -> component).

        Returns:
            Dictionary mapping names to working components.
        """
        return dict(self._working_components)

    def get_library_component_by_key(self, key: str) -> URDFComponent | None:
        """Get a library component by its key.

        Args:
            key: The library key (format: "source_file:component_name")

        Returns:
            Component if found, None otherwise.
        """
        return self._library_components.get(key)


class ComponentLibraryWidget(QWidget):
    """Widget for browsing and managing component library."""

    component_selected = pyqtSignal(object)  # URDFComponent
    component_copied = pyqtSignal(object)  # URDFComponent (the copy)
    component_edited = pyqtSignal(object)  # URDFComponent

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the component library widget."""
        super().__init__(parent)
        self.library = ComponentLibrary()
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Create splitter for library and working areas
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Library section (read-only)
        library_group = QGroupBox("Component Library (Read-Only)")
        library_layout = QVBoxLayout(library_group)

        # Load button
        load_layout = QHBoxLayout()
        self.load_library_btn = QPushButton("Load URDF as Library")
        load_layout.addWidget(self.load_library_btn)
        library_layout.addLayout(load_layout)

        # Library tree
        self.library_tree = QTreeWidget()
        self.library_tree.setHeaderLabels(["Component", "Type", "Source"])
        self.library_tree.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        header = self.library_tree.header()
        if header:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        library_layout.addWidget(self.library_tree)

        # Copy button
        copy_layout = QHBoxLayout()
        self.copy_btn = QPushButton("Copy to Working Set")
        self.copy_btn.setEnabled(False)
        copy_layout.addWidget(self.copy_btn)
        library_layout.addLayout(copy_layout)

        splitter.addWidget(library_group)

        # Working section (editable)
        working_group = QGroupBox("Working Components (Editable)")
        working_layout = QVBoxLayout(working_group)

        # Working tree
        self.working_tree = QTreeWidget()
        self.working_tree.setHeaderLabels(["Component", "Type", "Status"])
        self.working_tree.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        header = self.working_tree.header()
        if header:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        working_layout.addWidget(self.working_tree)

        # Edit/Remove buttons
        btn_layout = QHBoxLayout()
        self.edit_btn = QPushButton("Edit")
        self.edit_btn.setEnabled(False)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setEnabled(False)
        btn_layout.addWidget(self.edit_btn)
        btn_layout.addWidget(self.remove_btn)
        working_layout.addLayout(btn_layout)

        splitter.addWidget(working_group)

        layout.addWidget(splitter)

        # Preview area
        preview_group = QGroupBox("Component Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(150)
        preview_layout.addWidget(self.preview_text)
        layout.addWidget(preview_group)

    def _connect_signals(self) -> None:
        """Connect signals to slots."""
        self.load_library_btn.clicked.connect(self._on_load_library)
        self.copy_btn.clicked.connect(self._on_copy_to_working)
        self.edit_btn.clicked.connect(self._on_edit_component)
        self.remove_btn.clicked.connect(self._on_remove_component)

        self.library_tree.itemSelectionChanged.connect(
            self._on_library_selection_changed
        )
        self.working_tree.itemSelectionChanged.connect(
            self._on_working_selection_changed
        )

    def _on_load_library(self) -> None:
        """Handle load library button click."""
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load URDF as Library",
            "",
            "URDF Files (*.urdf);;XML Files (*.xml);;All Files (*)",
        )

        if file_path:
            path = Path(file_path)
            components = self.library.load_urdf_as_library(path)
            self._refresh_library_tree()

            if components:
                QMessageBox.information(
                    self,
                    "Library Loaded",
                    f"Loaded {len(components)} components from {path.name}.\n"
                    "These components are read-only. Use 'Copy to Working Set' to edit.",
                )

    def _on_copy_to_working(self) -> None:
        """Handle copy to working button click."""
        current = self.library_tree.currentItem()
        if not current:
            return

        library_key = current.data(0, Qt.ItemDataRole.UserRole)
        if not library_key:
            return

        # Ask for new name
        original_name = current.text(0)
        dialog = CopyComponentDialog(original_name, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_name = dialog.get_new_name()
            component = self.library.copy_to_working(library_key, new_name)
            if component:
                self._refresh_working_tree()
                self.component_copied.emit(component)

    def _on_edit_component(self) -> None:
        """Handle edit button click."""
        current = self.working_tree.currentItem()
        if not current:
            return

        name = current.text(0)
        component = self.library.get_component(name, from_library=False)
        if component:
            self.component_edited.emit(component)

    def _on_remove_component(self) -> None:
        """Handle remove button click."""
        current = self.working_tree.currentItem()
        if not current:
            return

        name = current.text(0)
        reply = QMessageBox.question(
            self,
            "Remove Component",
            f"Remove component '{name}' from working set?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.library.remove_working_component(name)
            self._refresh_working_tree()

    def _on_library_selection_changed(self) -> None:
        """Handle library tree selection change."""
        current = self.library_tree.currentItem()
        self.copy_btn.setEnabled(
            current is not None
            and current.data(0, Qt.ItemDataRole.UserRole) is not None
        )

        if current:
            library_key = current.data(0, Qt.ItemDataRole.UserRole)
            component = self.library.get_library_component_by_key(library_key)
            if component:
                self.preview_text.setPlainText(component.xml_content)
                self.component_selected.emit(component)
        else:
            self.preview_text.clear()

    def _on_working_selection_changed(self) -> None:
        """Handle working tree selection change."""
        current = self.working_tree.currentItem()
        has_selection = current is not None
        self.edit_btn.setEnabled(has_selection)
        self.remove_btn.setEnabled(has_selection)

        if current:
            name = current.text(0)
            component = self.library.get_component(name, from_library=False)
            if component:
                self.preview_text.setPlainText(component.xml_content)
                self.component_selected.emit(component)
        else:
            self.preview_text.clear()

    def _refresh_library_tree(self) -> None:
        """Refresh the library tree widget."""
        self.library_tree.clear()

        # Group by source file
        by_source: dict[str, list[tuple[str, URDFComponent]]] = {}
        for key, component in self.library.get_library_items().items():
            source = component.source_file.name if component.source_file else "Unknown"
            if source not in by_source:
                by_source[source] = []
            by_source[source].append((key, component))

        for source, components in by_source.items():
            source_item = QTreeWidgetItem([source, "", ""])
            source_item.setFlags(source_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.library_tree.addTopLevelItem(source_item)

            for key, component in components:
                item = QTreeWidgetItem(
                    [
                        component.name,
                        component.component_type.value,
                        component.get_hash(),
                    ]
                )
                item.setData(0, Qt.ItemDataRole.UserRole, key)
                # Visual indication of read-only
                item.setForeground(0, QColor(100, 100, 100))
                source_item.addChild(item)

            source_item.setExpanded(True)

    def _refresh_working_tree(self) -> None:
        """Refresh the working tree widget."""
        self.working_tree.clear()

        # Group by type
        by_type: dict[ComponentType, list[URDFComponent]] = {}
        for component in self.library.get_working_items().values():
            if component.component_type not in by_type:
                by_type[component.component_type] = []
            by_type[component.component_type].append(component)

        for comp_type, components in by_type.items():
            type_item = QTreeWidgetItem([comp_type.value.title() + "s", "", ""])
            type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.working_tree.addTopLevelItem(type_item)

            for component in components:
                status = "Modified" if not component.source_file else "Copied"
                item = QTreeWidgetItem(
                    [
                        component.name,
                        component.component_type.value,
                        status,
                    ]
                )
                type_item.addChild(item)

            type_item.setExpanded(True)

    def load_urdf_to_library(self, urdf_path: Path) -> None:
        """Load a URDF file into the library."""
        self.library.load_urdf_as_library(urdf_path)
        self._refresh_library_tree()

    def load_urdf_to_working(self, urdf_path: Path) -> None:
        """Load a URDF file into the working set."""
        self.library.load_urdf_as_working(urdf_path)
        self._refresh_working_tree()

    def get_library(self) -> ComponentLibrary:
        """Get the underlying component library."""
        return self.library


class CopyComponentDialog(QDialog):
    """Dialog for copying a component with a new name."""

    def __init__(self, original_name: str, parent: QWidget | None = None) -> None:
        """Initialize the dialog."""
        super().__init__(parent)
        self.setWindowTitle("Copy Component")
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)

        # Name input
        layout.addWidget(QLabel(f"Original name: {original_name}"))
        layout.addWidget(QLabel("Enter name for the copy:"))

        self.name_edit = QLineEdit()
        self.name_edit.setText(f"{original_name}_copy")
        self.name_edit.selectAll()
        layout.addWidget(self.name_edit)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_new_name(self) -> str:
        """Get the new name entered by the user."""
        return self.name_edit.text().strip()
