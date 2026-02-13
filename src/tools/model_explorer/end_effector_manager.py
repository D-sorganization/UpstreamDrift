"""End Effector Manager - Tools for swapping and managing end effectors.

Provides visual interface for easily swapping end effectors between URDFs,
changing attachment points, and managing end effector configurations.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as DefusedET
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EndEffector:
    """Represents an end effector configuration."""

    name: str
    link_element: ET.Element
    joint_element: ET.Element | None  # Joint connecting to parent
    child_links: list[ET.Element]  # Links that are part of this end effector
    child_joints: list[ET.Element]  # Joints within the end effector
    source_file: Path | None = None

    def get_all_link_names(self) -> list[str]:
        """Get names of all links in this end effector."""
        names = [self.link_element.get("name", "")]
        for link in self.child_links:
            names.append(link.get("name", ""))
        return names

    def get_attachment_joint_type(self) -> str:
        """Get the joint type for attaching to parent."""
        if self.joint_element is not None:
            return self.joint_element.get("type", "fixed")
        return "fixed"

    def to_xml_elements(self) -> tuple[list[ET.Element], list[ET.Element]]:
        """Convert to XML elements (links, joints)."""
        links = [copy.deepcopy(self.link_element)]
        links.extend(copy.deepcopy(link) for link in self.child_links)

        joints = []
        if self.joint_element is not None:
            joints.append(copy.deepcopy(self.joint_element))
        joints.extend(copy.deepcopy(joint) for joint in self.child_joints)

        return links, joints


class EndEffectorLibrary:
    """Library of available end effectors."""

    def __init__(self) -> None:
        """Initialize the library."""
        self.end_effectors: dict[str, EndEffector] = {}
        self._builtin_definitions = self._create_builtin_definitions()

    def _create_builtin_definitions(self) -> dict[str, dict[str, Any]]:
        """Create built-in end effector definitions."""
        return {
            "simple_gripper": {
                "name": "Simple Gripper",
                "description": "Basic two-finger parallel gripper",
                "link_xml": """
                    <link name="gripper_base">
                        <inertial>
                            <mass value="0.5"/>
                            <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
                        </inertial>
                        <visual>
                            <geometry><box size="0.08 0.08 0.02"/></geometry>
                            <material name="gray"><color rgba="0.5 0.5 0.5 1"/></material>
                        </visual>
                        <collision><geometry><box size="0.08 0.08 0.02"/></geometry></collision>
                    </link>
                """,
                "child_links": [
                    """<link name="left_finger">
                        <inertial><mass value="0.1"/>
                        <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/></inertial>
                        <visual><origin xyz="0 0 0.025"/><geometry><box size="0.01 0.02 0.05"/></geometry>
                        <material name="blue"><color rgba="0.2 0.2 0.8 1"/></material></visual>
                        <collision><origin xyz="0 0 0.025"/><geometry><box size="0.01 0.02 0.05"/></geometry></collision>
                    </link>""",
                    """<link name="right_finger">
                        <inertial><mass value="0.1"/>
                        <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/></inertial>
                        <visual><origin xyz="0 0 0.025"/><geometry><box size="0.01 0.02 0.05"/></geometry>
                        <material name="blue"><color rgba="0.2 0.2 0.8 1"/></material></visual>
                        <collision><origin xyz="0 0 0.025"/><geometry><box size="0.01 0.02 0.05"/></geometry></collision>
                    </link>""",
                ],
                "child_joints": [
                    """<joint name="left_finger_joint" type="prismatic">
                        <parent link="gripper_base"/><child link="left_finger"/>
                        <origin xyz="0.02 0 0.01" rpy="0 0 0"/>
                        <axis xyz="1 0 0"/>
                        <limit lower="-0.02" upper="0.02" effort="10" velocity="0.5"/>
                    </joint>""",
                    """<joint name="right_finger_joint" type="prismatic">
                        <parent link="gripper_base"/><child link="right_finger"/>
                        <origin xyz="-0.02 0 0.01" rpy="0 0 0"/>
                        <axis xyz="1 0 0"/>
                        <limit lower="-0.02" upper="0.02" effort="10" velocity="0.5"/>
                    </joint>""",
                ],
            },
            "tool_flange": {
                "name": "Tool Flange",
                "description": "Simple tool attachment flange",
                "link_xml": """
                    <link name="tool_flange">
                        <inertial>
                            <mass value="0.2"/>
                            <inertia ixx="0.0005" iyy="0.0005" izz="0.0005" ixy="0" ixz="0" iyz="0"/>
                        </inertial>
                        <visual>
                            <geometry><cylinder radius="0.04" length="0.02"/></geometry>
                            <material name="metal"><color rgba="0.7 0.7 0.7 1"/></material>
                        </visual>
                        <collision><geometry><cylinder radius="0.04" length="0.02"/></geometry></collision>
                    </link>
                """,
                "child_links": [],
                "child_joints": [],
            },
            "golf_club_attachment": {
                "name": "Golf Club Attachment",
                "description": "Attachment point for golf club grip",
                "link_xml": """
                    <link name="club_mount">
                        <inertial>
                            <mass value="0.1"/>
                            <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/>
                        </inertial>
                        <visual>
                            <geometry><cylinder radius="0.015" length="0.05"/></geometry>
                            <material name="rubber"><color rgba="0.1 0.1 0.1 1"/></material>
                        </visual>
                        <collision><geometry><cylinder radius="0.015" length="0.05"/></geometry></collision>
                    </link>
                """,
                "child_links": [],
                "child_joints": [],
            },
        }

    def get_builtin(self, key: str) -> EndEffector | None:
        """Get a built-in end effector definition."""
        if key not in self._builtin_definitions:
            return None

        definition = self._builtin_definitions[key]

        # Parse link XML
        link_elem = DefusedET.fromstring(definition["link_xml"].strip())

        # Parse child links
        child_links = []
        for link_xml in definition["child_links"]:
            child_links.append(DefusedET.fromstring(link_xml.strip()))

        # Parse child joints
        child_joints = []
        for joint_xml in definition["child_joints"]:
            child_joints.append(DefusedET.fromstring(joint_xml.strip()))

        return EndEffector(
            name=definition["name"],
            link_element=link_elem,
            joint_element=None,  # Will be created on attachment
            child_links=child_links,
            child_joints=child_joints,
        )

    def get_builtin_names(self) -> list[str]:
        """Get list of built-in end effector names."""
        return list(self._builtin_definitions.keys())

    def get_builtin_info(self, key: str) -> dict[str, str] | None:
        """Get info about a built-in end effector."""
        if key in self._builtin_definitions:
            return {
                "name": self._builtin_definitions[key]["name"],
                "description": self._builtin_definitions[key]["description"],
            }
        return None

    def extract_from_urdf(
        self,
        urdf_content: str,
        end_effector_link: str,
        source_file: Path | None = None,
    ) -> EndEffector | None:
        """Extract an end effector and its subtree from a URDF.

        Args:
            urdf_content: URDF XML content
            end_effector_link: Name of the root link of the end effector
            source_file: Source file path for reference

        Returns:
            Extracted end effector, or None if not found
        """
        try:
            root = DefusedET.fromstring(urdf_content)
        except ET.ParseError:
            return None

        # Find the end effector link
        ee_link = None
        for link in root.findall("link"):
            if link.get("name") == end_effector_link:
                ee_link = link
                break

        if ee_link is None:
            return None

        # Find the joint connecting to this link (as child)
        ee_joint = None
        for joint in root.findall("joint"):
            child = joint.find("child")
            if child is not None and child.get("link") == end_effector_link:
                ee_joint = joint
                break

        # Recursively find all child links and joints
        child_links = []
        child_joints = []

        def collect_children(parent_name: str) -> None:
            """Recursively gather child links and joints under the parent."""
            for joint in root.findall("joint"):
                parent = joint.find("parent")
                child = joint.find("child")
                if parent is not None and parent.get("link") == parent_name:
                    child_name = child.get("link") if child is not None else None
                    if child_name:
                        # Find the child link
                        for link in root.findall("link"):
                            if link.get("name") == child_name:
                                child_links.append(link)
                                child_joints.append(joint)
                                collect_children(child_name)
                                break

        collect_children(end_effector_link)

        return EndEffector(
            name=end_effector_link,
            link_element=ee_link,
            joint_element=ee_joint,
            child_links=child_links,
            child_joints=child_joints,
            source_file=source_file,
        )

    def add_to_library(self, key: str, end_effector: EndEffector) -> None:
        """Add an end effector to the library."""
        self.end_effectors[key] = end_effector

    def remove_from_library(self, key: str) -> bool:
        """Remove an end effector from the library."""
        if key in self.end_effectors:
            del self.end_effectors[key]
            return True
        return False


class AttachmentPointSelector(QDialog):
    """Dialog for selecting and configuring attachment point."""

    def __init__(
        self,
        available_links: list[str],
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the dialog."""
        super().__init__(parent)
        self.setWindowTitle("Select Attachment Point")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Attachment link selection
        attach_group = QGroupBox("Attachment Configuration")
        attach_layout = QFormLayout(attach_group)

        self.link_combo = QComboBox()
        self.link_combo.addItems(available_links)
        attach_layout.addRow("Attach to link:", self.link_combo)

        self.joint_type_combo = QComboBox()
        self.joint_type_combo.addItems(["fixed", "revolute", "prismatic", "continuous"])
        attach_layout.addRow("Joint type:", self.joint_type_combo)

        layout.addWidget(attach_group)

        # Position offset
        offset_group = QGroupBox("Position Offset")
        offset_layout = QFormLayout(offset_group)

        self.offset_x = QDoubleSpinBox()
        self.offset_x.setRange(-10, 10)
        self.offset_x.setValue(0)
        self.offset_x.setSuffix(" m")
        offset_layout.addRow("X:", self.offset_x)

        self.offset_y = QDoubleSpinBox()
        self.offset_y.setRange(-10, 10)
        self.offset_y.setValue(0)
        self.offset_y.setSuffix(" m")
        offset_layout.addRow("Y:", self.offset_y)

        self.offset_z = QDoubleSpinBox()
        self.offset_z.setRange(-10, 10)
        self.offset_z.setValue(0.1)
        self.offset_z.setSuffix(" m")
        offset_layout.addRow("Z:", self.offset_z)

        layout.addWidget(offset_group)

        # Orientation
        orient_group = QGroupBox("Orientation (RPY)")
        orient_layout = QFormLayout(orient_group)

        self.roll = QDoubleSpinBox()
        self.roll.setRange(-3.15, 3.15)
        self.roll.setValue(0)
        self.roll.setSuffix(" rad")
        orient_layout.addRow("Roll:", self.roll)

        self.pitch = QDoubleSpinBox()
        self.pitch.setRange(-3.15, 3.15)
        self.pitch.setValue(0)
        self.pitch.setSuffix(" rad")
        orient_layout.addRow("Pitch:", self.pitch)

        self.yaw = QDoubleSpinBox()
        self.yaw.setRange(-3.15, 3.15)
        self.yaw.setValue(0)
        self.yaw.setSuffix(" rad")
        orient_layout.addRow("Yaw:", self.yaw)

        layout.addWidget(orient_group)

        # Name prefix
        prefix_group = QGroupBox("Naming")
        prefix_layout = QFormLayout(prefix_group)

        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("optional prefix for link/joint names")
        prefix_layout.addRow("Name prefix:", self.prefix_edit)

        layout.addWidget(prefix_group)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_configuration(self) -> dict[str, Any]:
        """Get the attachment configuration."""
        return {
            "parent_link": self.link_combo.currentText(),
            "joint_type": self.joint_type_combo.currentText(),
            "offset": (
                self.offset_x.value(),
                self.offset_y.value(),
                self.offset_z.value(),
            ),
            "orientation": (self.roll.value(), self.pitch.value(), self.yaw.value()),
            "name_prefix": self.prefix_edit.text(),
        }


class EndEffectorManagerWidget(QWidget):
    """Widget for managing and swapping end effectors."""

    urdf_modified = pyqtSignal(str)  # Emits new URDF content

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the end effector manager."""
        super().__init__(parent)
        self.library = EndEffectorLibrary()
        self.urdf_content: str = ""
        self.current_end_effectors: list[str] = []  # Links identified as EEs
        self._setup_ui()
        self._connect_signals()
        self._populate_builtin_list()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side - current model end effectors
        current_widget = QWidget()
        current_layout = QVBoxLayout(current_widget)

        current_group = QGroupBox("Current End Effectors")
        current_inner = QVBoxLayout(current_group)

        self.current_list = QListWidget()
        current_inner.addWidget(self.current_list)

        btn_layout = QHBoxLayout()
        self.identify_btn = QPushButton("Identify EEs")
        self.remove_ee_btn = QPushButton("Remove Selected")
        self.extract_btn = QPushButton("Extract to Library")
        btn_layout.addWidget(self.identify_btn)
        btn_layout.addWidget(self.remove_ee_btn)
        btn_layout.addWidget(self.extract_btn)
        current_inner.addLayout(btn_layout)

        current_layout.addWidget(current_group)

        # Selected EE info
        info_group = QGroupBox("Selected End Effector")
        info_layout = QVBoxLayout(info_group)
        self.ee_info_text = QTextEdit()
        self.ee_info_text.setReadOnly(True)
        self.ee_info_text.setMaximumHeight(100)
        info_layout.addWidget(self.ee_info_text)
        current_layout.addWidget(info_group)

        splitter.addWidget(current_widget)

        # Right side - library
        library_widget = QWidget()
        library_layout = QVBoxLayout(library_widget)

        # Built-in end effectors
        builtin_group = QGroupBox("Built-in End Effectors")
        builtin_layout = QVBoxLayout(builtin_group)

        self.builtin_list = QListWidget()
        builtin_layout.addWidget(self.builtin_list)

        library_layout.addWidget(builtin_group)

        # Custom library
        custom_group = QGroupBox("Custom Library")
        custom_layout = QVBoxLayout(custom_group)

        self.custom_list = QListWidget()
        custom_layout.addWidget(self.custom_list)

        import_btn_layout = QHBoxLayout()
        self.import_from_file_btn = QPushButton("Import from URDF")
        import_btn_layout.addWidget(self.import_from_file_btn)
        custom_layout.addLayout(import_btn_layout)

        library_layout.addWidget(custom_group)

        # Attach button
        self.attach_btn = QPushButton("Attach Selected to Model")
        self.attach_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        library_layout.addWidget(self.attach_btn)

        splitter.addWidget(library_widget)

        layout.addWidget(splitter)

        # Status
        self.status_label = QLabel("Load a URDF to manage end effectors")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)

    def _connect_signals(self) -> None:
        """Connect signals."""
        self.identify_btn.clicked.connect(self._on_identify_end_effectors)
        self.remove_ee_btn.clicked.connect(self._on_remove_end_effector)
        self.extract_btn.clicked.connect(self._on_extract_to_library)
        self.import_from_file_btn.clicked.connect(self._on_import_from_file)
        self.attach_btn.clicked.connect(self._on_attach_end_effector)

        self.current_list.itemSelectionChanged.connect(
            self._on_current_selection_changed
        )
        self.builtin_list.itemSelectionChanged.connect(
            self._on_library_selection_changed
        )
        self.custom_list.itemSelectionChanged.connect(
            self._on_library_selection_changed
        )

    def _populate_builtin_list(self) -> None:
        """Populate the built-in end effectors list."""
        self.builtin_list.clear()
        for key in self.library.get_builtin_names():
            info = self.library.get_builtin_info(key)
            if info:
                item = QListWidgetItem(f"{info['name']}")
                item.setData(Qt.ItemDataRole.UserRole, key)
                item.setToolTip(info["description"])
                self.builtin_list.addItem(item)

    def load_urdf(self, content: str) -> None:
        """Load URDF content."""
        self.urdf_content = content
        self._on_identify_end_effectors()

    def _on_identify_end_effectors(self) -> None:
        """Identify end effectors in the current URDF."""
        if not self.urdf_content:
            return

        try:
            root = DefusedET.fromstring(self.urdf_content)
        except ET.ParseError:
            return

        # Find all links
        links = {link.get("name"): link for link in root.findall("link")}

        # Find links that are children but not parents (leaf nodes)
        parent_links = set()
        child_links = set()

        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is not None:
                parent_links.add(parent.get("link"))
            if child is not None:
                child_links.add(child.get("link"))

        # End effectors are leaves (in child_links but not in parent_links)
        end_effector_names = child_links - parent_links

        # Also check for naming hints
        ee_hints = ["hand", "gripper", "tool", "effector", "finger", "tip", "end"]

        self.current_list.clear()
        self.current_end_effectors = []

        for name in links.keys():
            if name is None:
                continue
            is_leaf = name in end_effector_names
            has_hint = any(hint in name.lower() for hint in ee_hints)

            if is_leaf or has_hint:
                self.current_end_effectors.append(name)
                item = QListWidgetItem(name)
                if is_leaf:
                    item.setForeground(QColor("#006400"))  # Green for leaves
                    item.setToolTip("Leaf link (no children)")
                else:
                    item.setForeground(QColor("#0000FF"))  # Blue for name hints
                    item.setToolTip("Identified by naming convention")
                self.current_list.addItem(item)

        self.status_label.setText(
            f"Found {len(self.current_end_effectors)} potential end effector(s)"
        )

    def _on_current_selection_changed(self) -> None:
        """Handle current EE list selection change."""
        current = self.current_list.currentItem()
        if not current:
            self.ee_info_text.clear()
            return

        link_name = current.text()

        # Extract info about this end effector
        ee = self.library.extract_from_urdf(self.urdf_content, link_name)
        if ee:
            info = f"Link: {ee.name}\n"
            info += f"Child links: {len(ee.child_links)}\n"
            info += f"Child joints: {len(ee.child_joints)}\n"
            if ee.joint_element is not None:
                info += f"Attachment joint: {ee.joint_element.get('name', 'unknown')}\n"
                info += f"Joint type: {ee.get_attachment_joint_type()}"
            self.ee_info_text.setPlainText(info)
        else:
            self.ee_info_text.setPlainText(f"Link: {link_name}")

    def _on_library_selection_changed(self) -> None:
        """Handle library selection change."""
        # Deselect in the other list
        sender = self.sender()
        if sender == self.builtin_list:
            self.custom_list.clearSelection()
        else:
            self.builtin_list.clearSelection()

    def _on_remove_end_effector(self) -> None:
        """Remove the selected end effector from the model."""
        current = self.current_list.currentItem()
        if not current:
            self.status_label.setText("Select an end effector to remove")
            return

        link_name = current.text()

        reply = QMessageBox.question(
            self,
            "Remove End Effector",
            f"Remove end effector '{link_name}' and all its children?\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Extract the EE first to get all links to remove
        ee = self.library.extract_from_urdf(self.urdf_content, link_name)
        if not ee:
            return

        try:
            root = DefusedET.fromstring(self.urdf_content)
        except ET.ParseError:
            return

        # Get all link names to remove
        links_to_remove = ee.get_all_link_names()

        # Remove links
        for link in list(root.findall("link")):
            if link.get("name") in links_to_remove:
                root.remove(link)

        # Remove joints connected to these links
        for joint in list(root.findall("joint")):
            parent = joint.find("parent")
            child = joint.find("child")
            parent_link = parent.get("link") if parent is not None else None
            child_link = child.get("link") if child is not None else None

            if parent_link in links_to_remove or child_link in links_to_remove:
                root.remove(joint)

        # Generate new URDF
        ET.indent(root, space="  ")
        new_content = ET.tostring(root, encoding="unicode", xml_declaration=True)

        self.urdf_content = new_content
        self._on_identify_end_effectors()
        self.urdf_modified.emit(new_content)
        self.status_label.setText(f"Removed end effector '{link_name}'")

    def _on_extract_to_library(self) -> None:
        """Extract selected EE to custom library."""
        current = self.current_list.currentItem()
        if not current:
            self.status_label.setText("Select an end effector to extract")
            return

        link_name = current.text()
        ee = self.library.extract_from_urdf(self.urdf_content, link_name)

        if ee:
            key = link_name.lower().replace(" ", "_")
            self.library.add_to_library(key, ee)

            item = QListWidgetItem(ee.name)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.custom_list.addItem(item)

            self.status_label.setText(f"Extracted '{link_name}' to library")

    def _on_import_from_file(self) -> None:
        """Import an end effector from another URDF file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select URDF with End Effector",
            "",
            "URDF Files (*.urdf);;XML Files (*.xml)",
        )

        if not file_path:
            return

        try:
            content = Path(file_path).read_text(encoding="utf-8")
            root = DefusedET.fromstring(content)
        except (FileNotFoundError, OSError) as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {e}")
            return

        # Get list of links
        links = [link.get("name", "") for link in root.findall("link")]

        if not links:
            QMessageBox.warning(self, "No Links", "No links found in the file.")
            return

        # Let user select which link to import as EE
        link_name, ok = self._select_from_list(
            "Select End Effector Link",
            "Select the root link of the end effector:",
            links,
        )

        if ok and link_name:
            ee = self.library.extract_from_urdf(content, link_name, Path(file_path))
            if ee:
                key = f"imported_{link_name}".lower().replace(" ", "_")
                self.library.add_to_library(key, ee)

                item = QListWidgetItem(f"{ee.name} (imported)")
                item.setData(Qt.ItemDataRole.UserRole, key)
                self.custom_list.addItem(item)

                self.status_label.setText(
                    f"Imported '{link_name}' from {Path(file_path).name}"
                )

    def _select_from_list(
        self, title: str, label: str, items: list[str]
    ) -> tuple[str, bool]:
        """Show a simple selection dialog."""
        from PyQt6.QtWidgets import QInputDialog

        item, ok = QInputDialog.getItem(self, title, label, items, 0, False)
        return item, bool(ok)

    def _on_attach_end_effector(self) -> None:
        """Attach selected library EE to the model."""
        # Check which list has selection
        builtin_item = self.builtin_list.currentItem()
        custom_item = self.custom_list.currentItem()

        ee: EndEffector | None = None

        if builtin_item:
            key = builtin_item.data(Qt.ItemDataRole.UserRole)
            ee = self.library.get_builtin(key)
        elif custom_item:
            key = custom_item.data(Qt.ItemDataRole.UserRole)
            ee = self.library.end_effectors.get(key)

        if not ee:
            self.status_label.setText("Select an end effector from the library")
            return

        if not self.urdf_content:
            self.status_label.setText("Load a URDF first")
            return

        # Get available links for attachment
        try:
            root = DefusedET.fromstring(self.urdf_content)
        except ET.ParseError:
            return

        available_links = [link.get("name", "") for link in root.findall("link")]

        if not available_links:
            self.status_label.setText("No links available for attachment")
            return

        # Show attachment dialog
        dialog = AttachmentPointSelector(available_links, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        config = dialog.get_configuration()
        self._attach_end_effector(ee, config)

    def _attach_end_effector(self, ee: EndEffector, config: dict[str, Any]) -> None:
        """Attach an end effector to the model."""
        try:
            root = DefusedET.fromstring(self.urdf_content)
        except ET.ParseError:
            return

        prefix = config["name_prefix"]
        parent_link = config["parent_link"]

        # Get link and joint elements
        links, joints = ee.to_xml_elements()

        # Rename if prefix is specified
        name_mapping: dict[str, str] = {}
        if prefix:
            for link in links:
                old_name = link.get("name", "")
                new_name = prefix + old_name
                link.set("name", new_name)
                name_mapping[old_name] = new_name

            for joint in joints:
                old_name = joint.get("name", "")
                joint.set("name", prefix + old_name)

                # Update parent/child references
                parent = joint.find("parent")
                child = joint.find("child")
                if parent is not None:
                    old_link = parent.get("link", "")
                    if old_link in name_mapping:
                        parent.set("link", name_mapping[old_link])
                if child is not None:
                    old_link = child.get("link", "")
                    if old_link in name_mapping:
                        child.set("link", name_mapping[old_link])

        # Add links to model
        for link in links:
            root.append(link)

        # Add joints to model
        for joint in joints:
            root.append(joint)

        # Create attachment joint
        ee_root_name = links[0].get("name", "end_effector")
        attachment_joint = ET.Element(
            "joint",
            name=f"{prefix}attachment_joint" if prefix else "attachment_joint",
            type=config["joint_type"],
        )

        ET.SubElement(attachment_joint, "parent", link=parent_link)
        ET.SubElement(attachment_joint, "child", link=ee_root_name)

        offset = config["offset"]
        orient = config["orientation"]
        ET.SubElement(
            attachment_joint,
            "origin",
            xyz=f"{offset[0]} {offset[1]} {offset[2]}",
            rpy=f"{orient[0]} {orient[1]} {orient[2]}",
        )

        if config["joint_type"] in ["revolute", "prismatic"]:
            ET.SubElement(attachment_joint, "axis", xyz="0 0 1")
            ET.SubElement(
                attachment_joint,
                "limit",
                lower="-3.14",
                upper="3.14",
                effort="100",
                velocity="10",
            )

        root.append(attachment_joint)

        # Generate new URDF
        ET.indent(root, space="  ")
        new_content = ET.tostring(root, encoding="unicode", xml_declaration=True)

        self.urdf_content = new_content
        self._on_identify_end_effectors()
        self.urdf_modified.emit(new_content)
        self.status_label.setText(
            f"Attached end effector '{ee.name}' to '{parent_link}'"
        )

    def get_urdf_content(self) -> str:
        """Get the current URDF content."""
        return self.urdf_content
