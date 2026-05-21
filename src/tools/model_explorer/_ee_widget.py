from __future__ import annotations

import xml.etree.ElementTree as ET  # stdlib retained for Element/SubElement
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as DefusedET  # noqa: S314  # Security: defusedxml prevents XML attacks
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QListWidgetItem,
    QMessageBox,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger
from src.tools.model_explorer._attachment_dialog import AttachmentPointSelector
from src.tools.model_explorer._ee_library import EndEffectorLibrary
from src.tools.model_explorer._ee_model import EndEffector
from src.tools.model_explorer._ee_widget_ui import _EndEffectorManagerWidgetUIMixin

logger = get_logger(__name__)


class EndEffectorManagerWidget(_EndEffectorManagerWidgetUIMixin, QWidget):
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

    def load_urdf(self, content: str) -> None:
        """Load URDF content."""
        if content is None:
            raise ValueError("content must be provided")
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

        for name in links:
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
        if title is None:
            raise ValueError("title must be provided")
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

    def _attach_end_effector(self, ee: EndEffector, config: dict[str, Any]) -> None:  # noqa: C901
        """Attach an end effector to the model."""
        if ee is None:
            raise ValueError("ee must be provided")
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
