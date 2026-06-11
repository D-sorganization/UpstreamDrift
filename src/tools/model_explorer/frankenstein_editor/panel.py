import xml.etree.ElementTree as ET  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
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

from .model import URDFModel

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelPanelSelection:
    """Public description of the currently selected model component."""

    comp_type: str
    name: str
    element: ET.Element


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

        validation_group = QGroupBox("Validation Findings")
        validation_layout = QVBoxLayout(validation_group)
        self.validation_list = QListWidget()
        self.validation_list.setMaximumHeight(100)
        validation_layout.addWidget(self.validation_list)
        layout.addWidget(validation_group)

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
            self.set_model(
                URDFModel.from_file(file_path),
                label=f"File: {file_path.name}",
                dirty=False,
            )
            logger.info(f"Loaded URDF: {file_path}")
            return True
        except (RuntimeError, ValueError, OSError) as e:
            QMessageBox.critical(self, "Error", f"Failed to load URDF: {e}")
            logger.error(f"Failed to load URDF: {e}")
            return False

    def _on_new(self) -> None:
        """Handle new button click."""
        self.set_model(
            URDFModel.create_empty(),
            label="New model (unsaved)",
            dirty=True,
        )

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
            self.mark_clean()
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
            self.set_dirty(True)

    def _refresh_tree(self) -> None:
        """Refresh the component tree."""
        self.tree.clear()
        self._refresh_validation_findings()
        if not self.model:
            return
        model = self.model

        # Links
        links_item = QTreeWidgetItem(["Links", "", f"({len(model.links)})"])
        links_item.setFlags(links_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.tree.addTopLevelItem(links_item)

        for name, link in model.links.items():
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
        joints_item = QTreeWidgetItem(["Joints", "", f"({len(model.joints)})"])
        joints_item.setFlags(joints_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.tree.addTopLevelItem(joints_item)

        for name, joint in model.joints.items():
            joint_type = joint.get("type", "unknown")
            item = QTreeWidgetItem([name, "joint", joint_type])
            item.setData(0, Qt.ItemDataRole.UserRole, joint)
            item.setData(1, Qt.ItemDataRole.UserRole, "joint")
            joints_item.addChild(item)

        joints_item.setExpanded(True)

        # Materials
        if model.materials:
            materials_item = QTreeWidgetItem(
                ["Materials", "", f"({len(model.materials)})"]
            )
            materials_item.setFlags(
                materials_item.flags() & ~Qt.ItemFlag.ItemIsSelectable
            )
            self.tree.addTopLevelItem(materials_item)

            for name, material in model.materials.items():
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
            self.set_dirty(True)
            return result

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Failed to add component: {e}")
            return None

    def selected_component(self) -> ModelPanelSelection | None:
        """Return the selected component without exposing tree widget internals."""
        current = self.tree.currentItem()
        if not current:
            return None

        element = current.data(0, Qt.ItemDataRole.UserRole)
        if element is None:
            return None

        return ModelPanelSelection(
            comp_type=current.data(1, Qt.ItemDataRole.UserRole) or "",
            name=current.text(0),
            element=element,
        )

    def selected_link_name(self) -> str | None:
        """Return the selected link name when the current selection is a link."""
        selection = self.selected_component()
        if selection is None or selection.comp_type != "link":
            return None
        return selection.name

    def set_model(
        self,
        model: URDFModel | None,
        *,
        label: str | None = None,
        dirty: bool | None = None,
    ) -> None:
        """Replace the panel model and refresh associated panel state."""
        self.model = model
        self.file_label.setText(label or self._model_label(model))
        if dirty is not None and self.model is not None:
            self.model.is_modified = dirty
        self.save_btn.setEnabled(model is not None)
        self.refresh()

    def refresh(self) -> None:
        """Refresh public panel presentation after model mutation."""
        self._refresh_tree()

    def set_dirty(self, dirty: bool = True) -> None:
        """Record whether the current model has unsaved changes."""
        if self.model is not None:
            self.model.is_modified = dirty
        self.save_btn.setEnabled(self.model is not None)

    def mark_clean(self) -> None:
        """Mark the current model as saved without exposing the save button."""
        self.set_dirty(False)

    def get_model(self) -> URDFModel | None:
        """Get the current model."""
        return self.model

    def _model_label(self, model: URDFModel | None) -> str:
        if model is None:
            return "No file loaded"
        if model.file_path:
            return f"File: {model.file_path.name}"
        return "New model"

    def _refresh_validation_findings(self) -> None:
        """Surface current composition findings in the panel."""
        self.validation_list.clear()
        if not self.model:
            self.validation_list.addItem("No model loaded")
            return

        result = self.model.validate_composition()
        if not result.findings:
            self.validation_list.addItem("No validation findings")
            return

        for finding in result.findings:
            item = QListWidgetItem(
                f"{finding.severity.upper()} {finding.code}: {finding.message}"
            )
            item.setToolTip(", ".join(finding.elements))
            self.validation_list.addItem(item)
