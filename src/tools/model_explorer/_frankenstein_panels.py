"""ModelPanel and StealComponentDialog for the Frankenstein Editor."""

from __future__ import annotations

import xml.etree.ElementTree as ET  # stdlib retained for Element/SubElement
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
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger

from ._frankenstein_model import URDFModel

logger = get_logger(__name__)


class ModelPanel(QWidget):
    """Panel displaying a single URDF model with component tree."""

    component_selected = pyqtSignal(str, str, object)  # type, name, element
    component_double_clicked = pyqtSignal(str, str, object)  # For stealing

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """Initialize the model panel."""
        if title is None:
            raise ValueError("title must be provided")
        super().__init__(parent)
        self.title = title
        self.model: URDFModel | None = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Header
        header_layout = QHBoxLayout()
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(self.title_label)

        self.load_btn = QPushButton("Load URDF")
        self.new_btn = QPushButton("New")
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)

        header_layout.addWidget(self.load_btn)
        header_layout.addWidget(self.new_btn)
        header_layout.addWidget(self.save_btn)
        layout.addLayout(header_layout)

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
        except (RuntimeError, ValueError, OSError) as e:
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
        except (RuntimeError, ValueError, OSError) as e:
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
        if item is None:
            raise ValueError("item must be provided")
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
        if item is None:
            raise ValueError("item must be provided")
        element = item.data(0, Qt.ItemDataRole.UserRole)
        if element is not None:
            comp_type = item.data(1, Qt.ItemDataRole.UserRole) or ""
            name = item.text(0)
            self.component_double_clicked.emit(comp_type, name, element)

    def _remove_component(self, item: QTreeWidgetItem) -> None:
        """Remove a component from the model."""
        if item is None:
            raise ValueError("item must be provided")
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
        if comp_type is None:
            raise ValueError("comp_type must be provided")
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

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Failed to add component: {e}")
            return None

    def get_model(self) -> URDFModel | None:
        """Get the current model."""
        return self.model


class StealComponentDialog(QDialog):
    """Dialog for configuring component stealing with renaming."""

    def __init__(
        self,
        comp_type: str,
        original_name: str,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the dialog."""
        if comp_type is None:
            raise ValueError("comp_type must be provided")
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
