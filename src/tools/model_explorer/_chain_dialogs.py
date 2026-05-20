from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.tools.model_explorer._chain_model import KinematicTree


class InsertSegmentDialog(QDialog):
    """Dialog for inserting a new segment into the chain."""

    reparent_list: QListWidget | None

    def __init__(
        self,
        tree: KinematicTree,
        insert_after: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the dialog."""
        if tree is None:
            raise ValueError("tree must be provided")
        super().__init__(parent)
        self.tree = tree
        self.setWindowTitle("Insert Segment")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        layout.addWidget(self._create_insertion_group(tree, insert_after))
        layout.addWidget(self._create_link_group())
        layout.addWidget(self._create_joint_group())
        layout.addWidget(self._create_reparent_group(tree))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.parent_combo.currentTextChanged.connect(self._update_reparent_list)

    def _create_insertion_group(
        self, tree: KinematicTree, insert_after: str | None
    ) -> QGroupBox:
        if tree is None:
            raise ValueError("tree must be provided")
        insertion_group = QGroupBox("Insertion Point")
        insertion_layout = QFormLayout(insertion_group)

        self.parent_combo = QComboBox()
        for name in tree.nodes:
            self.parent_combo.addItem(name)
        if insert_after:
            index = self.parent_combo.findText(insert_after)
            if index >= 0:
                self.parent_combo.setCurrentIndex(index)
        insertion_layout.addRow("Insert after link:", self.parent_combo)
        return insertion_group

    def _create_link_group(self) -> QGroupBox:
        link_group = QGroupBox("New Link")
        link_layout = QFormLayout(link_group)

        self.link_name_edit = QLineEdit()
        self.link_name_edit.setPlaceholderText("new_link")
        link_layout.addRow("Link name:", self.link_name_edit)

        self.geometry_combo = QComboBox()
        self.geometry_combo.addItems(["box", "cylinder", "sphere", "capsule"])
        link_layout.addRow("Geometry:", self.geometry_combo)

        self.mass_spin = QDoubleSpinBox()
        self.mass_spin.setRange(0.001, 1000)
        self.mass_spin.setValue(1.0)
        self.mass_spin.setSuffix(" kg")
        link_layout.addRow("Mass:", self.mass_spin)
        return link_group

    def _create_joint_group(self) -> QGroupBox:
        joint_group = QGroupBox("New Joint")
        joint_layout = QFormLayout(joint_group)

        self.joint_name_edit = QLineEdit()
        self.joint_name_edit.setPlaceholderText("new_joint")
        joint_layout.addRow("Joint name:", self.joint_name_edit)

        self.joint_type_combo = QComboBox()
        self.joint_type_combo.addItems(["fixed", "revolute", "prismatic", "continuous"])
        joint_layout.addRow("Joint type:", self.joint_type_combo)

        axis_layout = QHBoxLayout()
        self.axis_x = QDoubleSpinBox()
        self.axis_x.setRange(-1, 1)
        self.axis_x.setValue(0)
        self.axis_y = QDoubleSpinBox()
        self.axis_y.setRange(-1, 1)
        self.axis_y.setValue(0)
        self.axis_z = QDoubleSpinBox()
        self.axis_z.setRange(-1, 1)
        self.axis_z.setValue(1)
        axis_layout.addWidget(QLabel("X:"))
        axis_layout.addWidget(self.axis_x)
        axis_layout.addWidget(QLabel("Y:"))
        axis_layout.addWidget(self.axis_y)
        axis_layout.addWidget(QLabel("Z:"))
        axis_layout.addWidget(self.axis_z)
        joint_layout.addRow("Axis:", axis_layout)
        return joint_group

    def _create_reparent_group(self, tree: KinematicTree) -> QGroupBox:
        if tree is None:
            raise ValueError("tree must be provided")
        reparent_group = QGroupBox("Re-parent Children")
        reparent_layout = QVBoxLayout(reparent_group)

        parent_name = self.parent_combo.currentText()
        if parent_name in tree.nodes:
            node = tree.nodes[parent_name]
            if node.children:
                self.reparent_list = QListWidget()
                self.reparent_list.setSelectionMode(
                    QListWidget.SelectionMode.MultiSelection
                )
                for child in node.children:
                    item = QListWidgetItem(child.name)
                    item.setSelected(True)
                    self.reparent_list.addItem(item)
                reparent_layout.addWidget(
                    QLabel("Select children to re-parent to new link:")
                )
                reparent_layout.addWidget(self.reparent_list)
            else:
                reparent_layout.addWidget(QLabel("No children to re-parent"))
                self.reparent_list = None
        else:
            reparent_layout.addWidget(QLabel("Select a parent link first"))
            self.reparent_list = None
        return reparent_group

    def _update_reparent_list(self, parent_name: str) -> None:
        """Update the reparent list when parent selection changes."""
        if parent_name is None:
            raise ValueError("parent_name must be provided")
        if self.reparent_list is None:
            return

        self.reparent_list.clear()

        if parent_name in self.tree.nodes:
            node = self.tree.nodes[parent_name]
            for child in node.children:
                item = QListWidgetItem(child.name)
                item.setSelected(True)
                self.reparent_list.addItem(item)

    def get_configuration(self) -> dict[str, Any]:
        """Get the dialog configuration."""
        children_to_reparent = []
        if self.reparent_list:
            for i in range(self.reparent_list.count()):
                item = self.reparent_list.item(i)
                if item and item.isSelected():
                    children_to_reparent.append(item.text())

        return {
            "parent_link": self.parent_combo.currentText(),
            "link_name": self.link_name_edit.text() or "new_link",
            "geometry": self.geometry_combo.currentText(),
            "mass": self.mass_spin.value(),
            "joint_name": self.joint_name_edit.text() or "new_joint",
            "joint_type": self.joint_type_combo.currentText(),
            "axis": (self.axis_x.value(), self.axis_y.value(), self.axis_z.value()),
            "reparent_children": children_to_reparent,
        }
