"""Frankenstein Editor - Side-by-side URDF editor for component stealing.

Enables combining components from multiple URDF files by displaying
two models side-by-side and allowing drag-and-drop or copy-paste
of components between them.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
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


@dataclass
class URDFModel:
    """Represents a loaded URDF model."""

    file_path: Path | None
    robot_name: str
    links: dict[str, ET.Element]
    joints: dict[str, ET.Element]
    materials: dict[str, ET.Element]
    other_elements: list[ET.Element]
    is_modified: bool = False

    @classmethod
    def from_file(cls, file_path: Path) -> URDFModel:
        """Load a URDF model from file."""
        tree = ET.parse(file_path)
        root = tree.getroot()
        return cls.from_element(root, file_path)

    @classmethod
    def from_element(cls, root: ET.Element, file_path: Path | None = None) -> URDFModel:
        """Create model from XML element."""
        robot_name = root.get("name", "unnamed_robot")

        links = {}
        joints = {}
        materials = {}
        other_elements = []

        for child in root:
            name = child.get("name", "")
            if child.tag == "link":
                links[name] = child
            elif child.tag == "joint":
                joints[name] = child
            elif child.tag == "material":
                materials[name] = child
            else:
                other_elements.append(child)

        return cls(
            file_path=file_path,
            robot_name=robot_name,
            links=links,
            joints=joints,
            materials=materials,
            other_elements=other_elements,
        )

    @classmethod
    def create_empty(cls, name: str = "new_robot") -> URDFModel:
        """Create an empty URDF model."""
        return cls(
            file_path=None,
            robot_name=name,
            links={},
            joints={},
            materials={},
            other_elements=[],
        )

    def to_xml(self) -> str:
        """Convert model to XML string."""
        root = ET.Element("robot", name=self.robot_name)

        # Add materials first
        for material in self.materials.values():
            root.append(copy.deepcopy(material))

        # Add links
        for link in self.links.values():
            root.append(copy.deepcopy(link))

        # Add joints
        for joint in self.joints.values():
            root.append(copy.deepcopy(joint))

        # Add other elements
        for elem in self.other_elements:
            root.append(copy.deepcopy(elem))

        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def add_link(self, link: ET.Element, new_name: str | None = None) -> str:
        """Add a link to the model.

        Args:
            link: Link element to add
            new_name: Optional new name for the link

        Returns:
            The name used for the link
        """
        link_copy = copy.deepcopy(link)
        name = new_name or link_copy.get("name", "unnamed_link")

        # Ensure unique name
        original_name = name
        counter = 1
        while name in self.links:
            name = f"{original_name}_{counter}"
            counter += 1

        link_copy.set("name", name)
        self.links[name] = link_copy
        self.is_modified = True
        return name

    def add_joint(
        self,
        joint: ET.Element,
        new_name: str | None = None,
        parent_mapping: dict[str, str] | None = None,
    ) -> str:
        """Add a joint to the model.

        Args:
            joint: Joint element to add
            new_name: Optional new name for the joint
            parent_mapping: Optional mapping from old link names to new ones

        Returns:
            The name used for the joint
        """
        joint_copy = copy.deepcopy(joint)
        name = new_name or joint_copy.get("name", "unnamed_joint")

        # Ensure unique name
        original_name = name
        counter = 1
        while name in self.joints:
            name = f"{original_name}_{counter}"
            counter += 1

        joint_copy.set("name", name)

        # Update parent/child references if mapping provided
        if parent_mapping:
            parent = joint_copy.find("parent")
            if parent is not None:
                old_link = parent.get("link", "")
                if old_link in parent_mapping:
                    parent.set("link", parent_mapping[old_link])

            child = joint_copy.find("child")
            if child is not None:
                old_link = child.get("link", "")
                if old_link in parent_mapping:
                    child.set("link", parent_mapping[old_link])

        self.joints[name] = joint_copy
        self.is_modified = True
        return name

    def add_material(self, material: ET.Element) -> str:
        """Add a material to the model."""
        material_copy = copy.deepcopy(material)
        name = material_copy.get("name", "unnamed_material")

        if name not in self.materials:
            self.materials[name] = material_copy
            self.is_modified = True

        return name

    def remove_link(self, name: str) -> bool:
        """Remove a link and its connected joints."""
        if name not in self.links:
            return False

        del self.links[name]

        # Remove joints connected to this link
        joints_to_remove = []
        for joint_name, joint in self.joints.items():
            parent = joint.find("parent")
            child = joint.find("child")
            if (parent is not None and parent.get("link") == name) or (
                child is not None and child.get("link") == name
            ):
                joints_to_remove.append(joint_name)

        for joint_name in joints_to_remove:
            del self.joints[joint_name]

        self.is_modified = True
        return True

    def remove_joint(self, name: str) -> bool:
        """Remove a joint."""
        if name not in self.joints:
            return False

        del self.joints[name]
        self.is_modified = True
        return True

    def get_link_names(self) -> list[str]:
        """Get list of link names."""
        return list(self.links.keys())

    def get_joint_names(self) -> list[str]:
        """Get list of joint names."""
        return list(self.joints.keys())


class ModelPanel(QWidget):
    """Panel displaying a single URDF model with component tree."""

    component_selected = pyqtSignal(str, str, object)  # type, name, element
    component_double_clicked = pyqtSignal(str, str, object)  # For stealing

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """Initialize the model panel."""
        super().__init__(parent)
        self.title = title
        self.model: URDFModel | None = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Header
        header = QHBoxLayout()
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header.addWidget(self.title_label)

        self.load_btn = QPushButton("Load URDF")
        self.new_btn = QPushButton("New")
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)

        header.addWidget(self.load_btn)
        header.addWidget(self.new_btn)
        header.addWidget(self.save_btn)
        layout.addLayout(header)

        # File info
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: gray;")
        layout.addWidget(self.file_label)

        # Component tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Component", "Type", "Details"])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setDragEnabled(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header = self.tree.header()
        if header:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.tree)

        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(120)
        preview_layout.addWidget(self.preview_text)
        layout.addWidget(preview_group)

    def _connect_signals(self) -> None:
        """Connect signals."""
        self.load_btn.clicked.connect(self._on_load)
        self.new_btn.clicked.connect(self._on_new)
        self.save_btn.clicked.connect(self._on_save)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)

    def _on_load(self) -> None:
        """Handle load button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load URDF File",
            "",
            "URDF Files (*.urdf);;XML Files (*.xml);;All Files (*)",
        )

        if file_path:
            self.load_file(Path(file_path))

    def load_file(self, file_path: Path) -> bool:
        """Load a URDF file."""
        try:
            self.model = URDFModel.from_file(file_path)
            self.file_label.setText(f"File: {file_path.name}")
            self.save_btn.setEnabled(True)
            self._refresh_tree()
            logger.info(f"Loaded URDF: {file_path}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load URDF: {e}")
            logger.error(f"Failed to load URDF: {e}")
            return False

    def _on_new(self) -> None:
        """Handle new button click."""
        self.model = URDFModel.create_empty()
        self.file_label.setText("New model (unsaved)")
        self.save_btn.setEnabled(True)
        self._refresh_tree()

    def _on_save(self) -> None:
        """Handle save button click."""
        if not self.model:
            return

        if self.model.file_path:
            file_path = self.model.file_path
        else:
            file_path_str, _ = QFileDialog.getSaveFileName(
                self,
                "Save URDF File",
                "robot.urdf",
                "URDF Files (*.urdf);;XML Files (*.xml)",
            )
            if not file_path_str:
                return
            file_path = Path(file_path_str)

        try:
            content = self.model.to_xml()
            file_path.write_text(content, encoding="utf-8")
            self.model.file_path = file_path
            self.model.is_modified = False
            self.file_label.setText(f"File: {file_path.name}")
            logger.info(f"Saved URDF: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save URDF: {e}")

    def _on_selection_changed(self) -> None:
        """Handle tree selection change."""
        current = self.tree.currentItem()
        if not current:
            self.preview_text.clear()
            return

        element = current.data(0, Qt.ItemDataRole.UserRole)
        if element is not None:
            xml_str = ET.tostring(element, encoding="unicode")
            self.preview_text.setPlainText(xml_str)

            comp_type = current.data(1, Qt.ItemDataRole.UserRole) or ""
            name = current.text(0)
            self.component_selected.emit(comp_type, name, element)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle double-click for stealing component."""
        element = item.data(0, Qt.ItemDataRole.UserRole)
        if element is not None:
            comp_type = item.data(1, Qt.ItemDataRole.UserRole) or ""
            name = item.text(0)
            self.component_double_clicked.emit(comp_type, name, element)

    def _on_context_menu(self, pos: Any) -> None:
        """Show context menu for components."""
        item = self.tree.itemAt(pos)
        if not item:
            return

        element = item.data(0, Qt.ItemDataRole.UserRole)
        if element is None:
            return

        menu = QMenu(self)

        copy_action = QAction("Copy to Other Model", self)
        copy_action.triggered.connect(lambda: self._emit_copy(item))
        menu.addAction(copy_action)

        if self.model:
            remove_action = QAction("Remove", self)
            remove_action.triggered.connect(lambda: self._remove_component(item))
            menu.addAction(remove_action)

        menu.exec(self.tree.mapToGlobal(pos))

    def _emit_copy(self, item: QTreeWidgetItem) -> None:
        """Emit signal to copy component."""
        element = item.data(0, Qt.ItemDataRole.UserRole)
        if element is not None:
            comp_type = item.data(1, Qt.ItemDataRole.UserRole) or ""
            name = item.text(0)
            self.component_double_clicked.emit(comp_type, name, element)

    def _remove_component(self, item: QTreeWidgetItem) -> None:
        """Remove a component from the model."""
        if not self.model:
            return

        name = item.text(0)
        comp_type = item.data(1, Qt.ItemDataRole.UserRole)

        reply = QMessageBox.question(
            self,
            "Remove Component",
            f"Remove {comp_type} '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            if comp_type == "link":
                self.model.remove_link(name)
            elif comp_type == "joint":
                self.model.remove_joint(name)

            self._refresh_tree()

    def _refresh_tree(self) -> None:
        """Refresh the component tree."""
        self.tree.clear()
        if not self.model:
            return

        # Links
        links_item = QTreeWidgetItem(["Links", "", f"({len(self.model.links)})"])
        links_item.setFlags(links_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.tree.addTopLevelItem(links_item)

        for name, link in self.model.links.items():
            # Get geometry info
            geom_info = "unknown"
            visual = link.find("visual/geometry")
            if visual is not None:
                for child in visual:
                    geom_info = child.tag
                    break

            item = QTreeWidgetItem([name, "link", geom_info])
            item.setData(0, Qt.ItemDataRole.UserRole, link)
            item.setData(1, Qt.ItemDataRole.UserRole, "link")
            links_item.addChild(item)

        links_item.setExpanded(True)

        # Joints
        joints_item = QTreeWidgetItem(["Joints", "", f"({len(self.model.joints)})"])
        joints_item.setFlags(joints_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.tree.addTopLevelItem(joints_item)

        for name, joint in self.model.joints.items():
            joint_type = joint.get("type", "unknown")
            item = QTreeWidgetItem([name, "joint", joint_type])
            item.setData(0, Qt.ItemDataRole.UserRole, joint)
            item.setData(1, Qt.ItemDataRole.UserRole, "joint")
            joints_item.addChild(item)

        joints_item.setExpanded(True)

        # Materials
        if self.model.materials:
            materials_item = QTreeWidgetItem(
                ["Materials", "", f"({len(self.model.materials)})"]
            )
            materials_item.setFlags(
                materials_item.flags() & ~Qt.ItemFlag.ItemIsSelectable
            )
            self.tree.addTopLevelItem(materials_item)

            for name, material in self.model.materials.items():
                item = QTreeWidgetItem([name, "material", ""])
                item.setData(0, Qt.ItemDataRole.UserRole, material)
                item.setData(1, Qt.ItemDataRole.UserRole, "material")
                materials_item.addChild(item)

            materials_item.setExpanded(True)

    def add_component(
        self,
        comp_type: str,
        element: ET.Element,
        name_prefix: str = "",
    ) -> str | None:
        """Add a component to this model.

        Args:
            comp_type: Component type (link, joint, material)
            element: XML element to add
            name_prefix: Prefix for the new name

        Returns:
            The name used, or None if failed
        """
        if not self.model:
            self.model = URDFModel.create_empty()

        try:
            if comp_type == "link":
                new_name = (
                    name_prefix + element.get("name", "link") if name_prefix else None
                )
                result = self.model.add_link(element, new_name)
            elif comp_type == "joint":
                new_name = (
                    name_prefix + element.get("name", "joint") if name_prefix else None
                )
                result = self.model.add_joint(element, new_name)
            elif comp_type == "material":
                result = self.model.add_material(element)
            else:
                return None

            self._refresh_tree()
            self.save_btn.setEnabled(True)
            return result

        except Exception as e:
            logger.error(f"Failed to add component: {e}")
            return None

    def get_model(self) -> URDFModel | None:
        """Get the current model."""
        return self.model


class FrankensteinEditor(QWidget):
    """Side-by-side URDF editor for combining components from multiple files."""

    model_updated = pyqtSignal(str, object)  # panel_id, model

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the Frankenstein editor."""
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel(
            "Double-click or right-click on a component to copy it to the other model. "
            "Components are copied - source files are never modified."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        layout.addWidget(instructions)

        # Main splitter with two model panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.left_panel = ModelPanel("Source Model (Read-Only)")
        self.right_panel = ModelPanel("Working Model (Editable)")

        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)

        layout.addWidget(splitter)

        # Transfer buttons
        transfer_layout = QHBoxLayout()
        transfer_layout.addStretch()

        self.copy_selected_btn = QPushButton("Copy Selected Component -->")
        self.copy_chain_btn = QPushButton("Copy Link Chain -->")
        self.merge_all_btn = QPushButton("Merge All Components -->")

        transfer_layout.addWidget(self.copy_selected_btn)
        transfer_layout.addWidget(self.copy_chain_btn)
        transfer_layout.addWidget(self.merge_all_btn)
        transfer_layout.addStretch()

        layout.addLayout(transfer_layout)

        # Status
        self.status_label = QLabel("Ready - Load URDFs to begin")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)

    def _connect_signals(self) -> None:
        """Connect signals."""
        # Left panel signals (source)
        self.left_panel.component_double_clicked.connect(self._on_copy_to_right)

        # Button signals
        self.copy_selected_btn.clicked.connect(self._on_copy_selected)
        self.copy_chain_btn.clicked.connect(self._on_copy_chain)
        self.merge_all_btn.clicked.connect(self._on_merge_all)

    def _on_copy_to_right(self, comp_type: str, name: str, element: ET.Element) -> None:
        """Copy component from left to right panel."""
        result = self.right_panel.add_component(comp_type, element)
        if result:
            self.status_label.setText(f"Copied {comp_type} '{name}' as '{result}'")
        else:
            self.status_label.setText(f"Failed to copy {comp_type} '{name}'")

    def _on_copy_selected(self) -> None:
        """Copy currently selected component."""
        current = self.left_panel.tree.currentItem()
        if not current:
            self.status_label.setText("No component selected in source model")
            return

        element = current.data(0, Qt.ItemDataRole.UserRole)
        if element is None:
            return

        comp_type = current.data(1, Qt.ItemDataRole.UserRole) or ""
        name = current.text(0)
        self._on_copy_to_right(comp_type, name, element)

    def _on_copy_chain(self) -> None:
        """Copy a link and all its connected joints/child links."""
        current = self.left_panel.tree.currentItem()
        if not current:
            self.status_label.setText("Select a link in the source model")
            return

        comp_type = current.data(1, Qt.ItemDataRole.UserRole)
        if comp_type != "link":
            self.status_label.setText("Please select a link (not a joint)")
            return

        source_model = self.left_panel.get_model()
        if not source_model:
            return

        # Get the selected link
        link_name = current.text(0)
        copied_count = self._copy_link_chain(source_model, link_name)
        self.status_label.setText(
            f"Copied chain starting from '{link_name}': {copied_count} components"
        )

    def _copy_link_chain(
        self,
        source_model: URDFModel,
        link_name: str,
        name_mapping: dict[str, str] | None = None,
    ) -> int:
        """Recursively copy a link and its child chain.

        Args:
            source_model: Source model
            link_name: Name of link to copy
            name_mapping: Mapping of old names to new names

        Returns:
            Number of components copied
        """
        if name_mapping is None:
            name_mapping = {}

        count = 0

        # Copy the link
        if link_name in source_model.links:
            link = source_model.links[link_name]
            new_name = self.right_panel.add_component("link", link)
            if new_name:
                name_mapping[link_name] = new_name
                count += 1

        # Find joints where this link is the parent
        for joint_name, joint in source_model.joints.items():
            parent = joint.find("parent")
            if parent is not None and parent.get("link") == link_name:
                # Copy the joint
                new_joint_name = self.right_panel.add_component("joint", joint)
                if new_joint_name:
                    count += 1

                # Recursively copy the child link
                child = joint.find("child")
                if child is not None:
                    child_link = child.get("link")
                    if child_link and child_link in source_model.links:
                        count += self._copy_link_chain(
                            source_model, child_link, name_mapping
                        )

        return count

    def _on_merge_all(self) -> None:
        """Merge all components from source to working model."""
        source_model = self.left_panel.get_model()
        if not source_model:
            self.status_label.setText("No source model loaded")
            return

        reply = QMessageBox.question(
            self,
            "Merge All",
            "This will copy all links, joints, and materials from the source model. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        count = 0

        # Copy all materials first
        for name, material in source_model.materials.items():
            if self.right_panel.add_component("material", material):
                count += 1

        # Copy all links
        for name, link in source_model.links.items():
            if self.right_panel.add_component("link", link):
                count += 1

        # Copy all joints
        for name, joint in source_model.joints.items():
            if self.right_panel.add_component("joint", joint):
                count += 1

        self.status_label.setText(f"Merged {count} components from source model")

    def load_source(self, file_path: Path) -> bool:
        """Load a file into the source (left) panel."""
        return self.left_panel.load_file(file_path)

    def load_working(self, file_path: Path) -> bool:
        """Load a file into the working (right) panel."""
        return self.right_panel.load_file(file_path)

    def get_working_model(self) -> URDFModel | None:
        """Get the working model."""
        return self.right_panel.get_model()

    def get_working_xml(self) -> str | None:
        """Get the working model as XML string."""
        model = self.right_panel.get_model()
        if model:
            return model.to_xml()
        return None


class StealComponentDialog(QDialog):
    """Dialog for configuring component stealing with renaming."""

    def __init__(
        self,
        comp_type: str,
        original_name: str,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the dialog."""
        super().__init__(parent)
        self.setWindowTitle("Copy Component")
        self.setMinimumWidth(350)

        layout = QVBoxLayout(self)

        # Info
        layout.addWidget(QLabel(f"Copying {comp_type}: {original_name}"))

        # Name input
        form = QFormLayout()
        self.name_edit = QLineEdit(original_name)
        form.addRow("New name:", self.name_edit)

        # Prefix option
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("e.g., 'imported_'")
        form.addRow("Add prefix:", self.prefix_edit)

        layout.addLayout(form)

        # Include related checkbox (for links)
        if comp_type == "link":
            self.include_materials = QLabel(
                "Note: Referenced materials will also be copied"
            )
            layout.addWidget(self.include_materials)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_new_name(self) -> str:
        """Get the new name with prefix."""
        prefix = self.prefix_edit.text()
        name = self.name_edit.text()
        return prefix + name
