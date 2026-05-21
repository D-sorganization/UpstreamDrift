"""Frankenstein Editor - Side-by-side URDF editor for component stealing.

Enables combining components from multiple URDF files by displaying
two models side-by-side and allowing drag-and-drop or copy-paste
of components between them.

Implementation split across:
- _frankenstein_model.py: URDFModel data class
- _frankenstein_panels.py: ModelPanel, StealComponentDialog
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET  # stdlib retained for Element/SubElement
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger

# Re-export public names for backward compatibility
from ._frankenstein_model import URDFModel
from ._frankenstein_panels import ModelPanel, StealComponentDialog

logger = get_logger(__name__)


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

        # Transfer buttons (left-to-right operations)
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

        # Comparison/manipulation buttons
        compare_layout = QHBoxLayout()
        compare_layout.addStretch()

        self.swap_btn = QPushButton("⇄ Swap Models")
        self.swap_btn.setToolTip("Exchange left and right models")

        self.copy_right_as_left_btn = QPushButton("← Copy Right as Source")
        self.copy_right_as_left_btn.setToolTip(
            "Load the working model into the source panel for comparison"
        )

        self.replace_subtree_btn = QPushButton("Replace Subtree")
        self.replace_subtree_btn.setToolTip(
            "Replace selected subtree in working model with source selection"
        )

        self.diff_btn = QPushButton("Show Diff")
        self.diff_btn.setToolTip("Show differences between models")

        compare_layout.addWidget(self.swap_btn)
        compare_layout.addWidget(self.copy_right_as_left_btn)
        compare_layout.addWidget(self.replace_subtree_btn)
        compare_layout.addWidget(self.diff_btn)
        compare_layout.addStretch()

        layout.addLayout(compare_layout)

        # Status
        self.status_label = QLabel("Ready - Load URDFs to begin")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)

    def _connect_signals(self) -> None:
        """Connect signals."""
        # Left panel signals (source)
        self.left_panel.component_double_clicked.connect(self._on_copy_to_right)

        # Transfer button signals
        self.copy_selected_btn.clicked.connect(self._on_copy_selected)
        self.copy_chain_btn.clicked.connect(self._on_copy_chain)
        self.merge_all_btn.clicked.connect(self._on_merge_all)

        # Comparison/manipulation button signals
        self.swap_btn.clicked.connect(self._on_swap_models)
        self.copy_right_as_left_btn.clicked.connect(self._on_copy_right_as_left)
        self.replace_subtree_btn.clicked.connect(self._on_replace_subtree)
        self.diff_btn.clicked.connect(self._on_show_diff)

    def _on_copy_to_right(self, comp_type: str, name: str, element: ET.Element) -> None:
        """Copy component from left to right panel."""
        if comp_type is None:
            raise ValueError("comp_type must be provided")
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

    def _copy_link_chain(  # noqa: C901
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
        if source_model is None:
            raise ValueError("source_model must be provided")
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
        for joint in source_model.joints.values():
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
        for material in source_model.materials.values():
            if self.right_panel.add_component("material", material):
                count += 1

        # Copy all links
        for link in source_model.links.values():
            if self.right_panel.add_component("link", link):
                count += 1

        # Copy all joints
        for joint in source_model.joints.values():
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

    def _on_swap_models(self) -> None:
        """Swap the left and right models."""
        left_model = self.left_panel.get_model()
        right_model = self.right_panel.get_model()

        if not left_model and not right_model:
            self.status_label.setText("No models to swap")
            return

        # Swap models
        self.left_panel.model = right_model
        self.right_panel.model = left_model

        # Update file labels
        if right_model and right_model.file_path:
            self.left_panel.file_label.setText(f"File: {right_model.file_path.name}")
        else:
            self.left_panel.file_label.setText(
                "No file" if not right_model else "New model"
            )

        if left_model and left_model.file_path:
            self.right_panel.file_label.setText(f"File: {left_model.file_path.name}")
        else:
            self.right_panel.file_label.setText(
                "No file" if not left_model else "New model"
            )

        # Refresh trees
        self.left_panel._refresh_tree()
        self.right_panel._refresh_tree()

        # Update button states
        self.left_panel.save_btn.setEnabled(left_model is not None)
        self.right_panel.save_btn.setEnabled(right_model is not None)

        self.status_label.setText("Models swapped")
        logger.info("Swapped left and right models")

    def _on_copy_right_as_left(self) -> None:
        """Copy the working (right) model as the source (left) model."""
        right_model = self.right_panel.get_model()

        if not right_model:
            self.status_label.setText("No working model to copy")
            return

        # Create a deep copy of the right model
        left_model = URDFModel(
            file_path=None,
            robot_name=right_model.robot_name + "_copy",
            links={k: copy.deepcopy(v) for k, v in right_model.links.items()},
            joints={k: copy.deepcopy(v) for k, v in right_model.joints.items()},
            materials={k: copy.deepcopy(v) for k, v in right_model.materials.items()},
            other_elements=[copy.deepcopy(e) for e in right_model.other_elements],
        )

        self.left_panel.model = left_model
        self.left_panel.file_label.setText("Copied from working model")
        self.left_panel.save_btn.setEnabled(True)
        self.left_panel._refresh_tree()

        self.status_label.setText("Working model copied to source panel for comparison")
        logger.info("Copied working model to source panel")

    def _on_replace_subtree(self) -> None:
        """Replace a subtree in the working model with one from the source."""
        source_model = self.left_panel.get_model()
        target_model = self.right_panel.get_model()

        if not source_model or not target_model:
            self.status_label.setText("Both models must be loaded to replace subtree")
            return

        # Get selected link from source
        source_item = self.left_panel.tree.currentItem()
        if not source_item:
            self.status_label.setText("Select a link in the source model")
            return

        source_type = source_item.data(1, Qt.ItemDataRole.UserRole)
        if source_type != "link":
            self.status_label.setText("Please select a link (not a joint) from source")
            return

        source_link_name = source_item.text(0)

        # Get selected link from target to replace
        target_item = self.right_panel.tree.currentItem()
        if not target_item:
            self.status_label.setText("Select a link in the working model to replace")
            return

        target_type = target_item.data(1, Qt.ItemDataRole.UserRole)
        if target_type != "link":
            self.status_label.setText(
                "Please select a link (not a joint) from working model"
            )
            return

        target_link_name = target_item.text(0)

        # Confirm replacement
        reply = QMessageBox.question(
            self,
            "Replace Subtree",
            f"Replace '{target_link_name}' subtree with '{source_link_name}' subtree?\n\n"
            "This will remove the target link and all its children, then copy the source subtree.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Remove target subtree
        self._remove_subtree(target_model, target_link_name)

        # Copy source subtree
        count = self._copy_link_chain(source_model, source_link_name)

        self.right_panel._refresh_tree()
        self.status_label.setText(
            f"Replaced '{target_link_name}' with '{source_link_name}' ({count} components)"
        )
        logger.info(f"Replaced subtree {target_link_name} -> {source_link_name}")

    def _remove_subtree(self, model: URDFModel, link_name: str) -> int:  # noqa: C901
        """Recursively remove a link and all its children.

        Args:
            model: Model to remove from
            link_name: Root link name to remove

        Returns:
            Number of components removed
        """
        if model is None:
            raise ValueError("model must be provided")
        count = 0

        # Find child links
        child_links = []
        joints_to_remove = []

        for joint_name, joint in model.joints.items():
            parent = joint.find("parent")
            child = joint.find("child")

            if parent is not None and parent.get("link") == link_name:
                joints_to_remove.append(joint_name)
                if child is not None:
                    child_link = child.get("link")
                    if child_link:
                        child_links.append(child_link)

        # Recursively remove children
        for child_link in child_links:
            count += self._remove_subtree(model, child_link)

        # Remove joints
        for joint_name in joints_to_remove:
            if joint_name in model.joints:
                del model.joints[joint_name]
                count += 1

        # Remove the link
        if link_name in model.links:
            del model.links[link_name]
            count += 1

        return count

    def _on_show_diff(self) -> None:
        """Show differences between source and working models."""
        source_model = self.left_panel.get_model()
        target_model = self.right_panel.get_model()

        if not source_model or not target_model:
            self.status_label.setText("Both models must be loaded to show diff")
            return

        # Calculate differences
        source_links = set(source_model.links.keys())
        target_links = set(target_model.links.keys())
        source_joints = set(source_model.joints.keys())
        target_joints = set(target_model.joints.keys())

        links_only_source = source_links - target_links
        links_only_target = target_links - source_links
        links_both = source_links & target_links

        joints_only_source = source_joints - target_joints
        joints_only_target = target_joints - source_joints

        # Build diff message
        diff_lines = [
            "=== Model Comparison ===",
            "",
            f"Source: {source_model.robot_name} ({len(source_links)} links, {len(source_joints)} joints)",
            f"Working: {target_model.robot_name} ({len(target_links)} links, {len(target_joints)} joints)",
            "",
        ]

        if links_only_source:
            diff_lines.append(
                f"Links only in source: {', '.join(sorted(links_only_source))}"
            )
        if links_only_target:
            diff_lines.append(
                f"Links only in working: {', '.join(sorted(links_only_target))}"
            )
        if links_both:
            diff_lines.append(f"Links in both: {', '.join(sorted(links_both))}")

        diff_lines.append("")

        if joints_only_source:
            diff_lines.append(
                f"Joints only in source: {', '.join(sorted(joints_only_source))}"
            )
        if joints_only_target:
            diff_lines.append(
                f"Joints only in working: {', '.join(sorted(joints_only_target))}"
            )

        # Show in dialog
        diff_dialog = QDialog(self)
        diff_dialog.setWindowTitle("Model Comparison")
        diff_dialog.setMinimumSize(500, 400)

        layout = QVBoxLayout(diff_dialog)

        diff_text = QTextEdit()
        diff_text.setReadOnly(True)
        diff_text.setPlainText("\n".join(diff_lines))
        layout.addWidget(diff_text)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(diff_dialog.accept)
        layout.addWidget(close_btn)

        diff_dialog.exec()
        self.status_label.setText("Diff comparison shown")


__all__ = [
    "FrankensteinEditor",
    "ModelPanel",
    "StealComponentDialog",
    "URDFModel",
]
