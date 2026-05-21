from __future__ import annotations

import copy
import xml.etree.ElementTree as ET  # stdlib retained for Element/SubElement
from typing import Any

import defusedxml.ElementTree as DefusedET  # noqa: S314  # Security: defusedxml prevents XML attacks
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QGroupBox,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.core.contracts import precondition
from src.tools.model_explorer._chain_dialogs import InsertSegmentDialog
from src.tools.model_explorer._chain_model import KinematicTree
from src.tools.model_explorer._chain_visualizer import ChainVisualizer


class ChainManipulationWidget(QWidget):
    """Widget for manipulating kinematic chains."""

    chain_modified = pyqtSignal(str)  # Emits new URDF content

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the chain manipulation widget."""
        super().__init__(parent)
        self.tree = KinematicTree()
        self.urdf_content: str = ""
        self.selected_node: str | None = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side - chain info and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Chain info
        info_group = QGroupBox("Chain Information")
        info_layout = QVBoxLayout(info_group)

        self.links_label = QLabel("Links: 0")
        self.joints_label = QLabel("Joints: 0")
        self.branches_label = QLabel("Branches: 0")
        self.end_effectors_label = QLabel("End effectors: 0")

        info_layout.addWidget(self.links_label)
        info_layout.addWidget(self.joints_label)
        info_layout.addWidget(self.branches_label)
        info_layout.addWidget(self.end_effectors_label)

        left_layout.addWidget(info_group)

        # Chain list
        chains_group = QGroupBox("Kinematic Chains")
        chains_layout = QVBoxLayout(chains_group)

        self.chains_list = QListWidget()
        chains_layout.addWidget(self.chains_list)

        left_layout.addWidget(chains_group)

        # Controls
        controls_group = QGroupBox("Chain Operations")
        controls_layout = QVBoxLayout(controls_group)

        self.insert_btn = QPushButton("Insert Segment")
        self.remove_btn = QPushButton("Remove Segment")
        self.split_chain_btn = QPushButton("Split Chain")
        self.merge_chains_btn = QPushButton("Merge Chains")

        controls_layout.addWidget(self.insert_btn)
        controls_layout.addWidget(self.remove_btn)
        controls_layout.addWidget(self.split_chain_btn)
        controls_layout.addWidget(self.merge_chains_btn)

        left_layout.addWidget(controls_group)

        splitter.addWidget(left_widget)

        # Right side - visualization
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        right_layout.addWidget(
            QLabel("Chain Visualization (double-click to select insertion point):")
        )

        self.visualizer = ChainVisualizer()
        self.visualizer.setMinimumSize(400, 300)
        right_layout.addWidget(self.visualizer)

        # Selected node info
        self.selected_label = QLabel("Selected: None")
        right_layout.addWidget(self.selected_label)

        splitter.addWidget(right_widget)

        layout.addWidget(splitter)

    def _connect_signals(self) -> None:
        """Connect signals."""
        self.insert_btn.clicked.connect(self._on_insert_segment)
        self.remove_btn.clicked.connect(self._on_remove_segment)
        self.split_chain_btn.clicked.connect(self._on_split_chain)
        self.merge_chains_btn.clicked.connect(self._on_merge_chains)

        self.visualizer.node_selected.connect(self._on_node_selected)
        self.visualizer.node_double_clicked.connect(self._on_node_double_clicked)

    @precondition(
        lambda self, content: content is not None and len(content.strip()) > 0,
        "URDF content must be a non-empty string",
    )
    def load_urdf(self, content: str) -> None:
        """Load URDF content and build the kinematic tree."""
        if content is None:
            raise ValueError("content must be provided")
        self.urdf_content = content
        self.tree.build_from_urdf(content)
        self._update_info()
        self._update_chains_list()
        self.visualizer.set_tree(self.tree)

    def _update_info(self) -> None:
        """Update the chain information display."""
        self.links_label.setText(f"Links: {len(self.tree.nodes)}")

        joint_count = sum(
            1 for n in self.tree.nodes.values() if n.joint_to_parent is not None
        )
        self.joints_label.setText(f"Joints: {joint_count}")

        branches = len(self.tree.get_branch_points())
        self.branches_label.setText(f"Branch points: {branches}")

        end_effectors = len(self.tree.get_end_effectors())
        self.end_effectors_label.setText(f"End effectors: {end_effectors}")

    def _update_chains_list(self) -> None:
        """Update the chains list widget."""
        self.chains_list.clear()

        chains = self.tree.get_all_chains()
        for i, chain in enumerate(chains):
            chain_str = " -> ".join(n.name for n in chain)
            self.chains_list.addItem(f"Chain {i + 1}: {chain_str}")

    def _on_node_selected(self, name: str) -> None:
        """Handle node selection."""
        if name is None:
            raise ValueError("name must be provided")
        self.selected_node = name
        node = self.tree.nodes.get(name)
        if node:
            info = f"Selected: {name}"
            if node.joint_to_parent:
                joint_type = node.joint_to_parent.get("type", "unknown")
                info += f" (joint: {joint_type})"
            if node.is_end_effector():
                info += " [End Effector]"
            self.selected_label.setText(info)

    def _on_node_double_clicked(self, name: str) -> None:
        """Handle node double-click for insertion."""
        self._show_insert_dialog(name)

    def _on_insert_segment(self) -> None:
        """Handle insert segment button."""
        # Get currently selected node, if any
        selected = None
        if self.tree.root:
            selected = self.tree.root.name

        self._show_insert_dialog(selected)

    def _show_insert_dialog(self, insert_after: str | None) -> None:
        """Show the insert segment dialog."""
        if not self.tree.nodes:
            QMessageBox.warning(self, "No Model", "Load a URDF first.")
            return

        dialog = InsertSegmentDialog(self.tree, insert_after, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_configuration()
            self._insert_segment(config)

    def _insert_segment(self, config: dict[str, Any]) -> None:  # noqa: C901
        """Insert a new segment into the URDF."""
        if config is None:
            raise ValueError("config must be provided")
        try:
            root = DefusedET.fromstring(self.urdf_content)
        except ET.ParseError:
            return

        parent_link = config["parent_link"]
        link_name = config["link_name"]
        joint_name = config["joint_name"]

        # Create new link
        new_link = ET.Element("link", name=link_name)

        # Add inertial
        inertial = ET.SubElement(new_link, "inertial")
        ET.SubElement(inertial, "mass", value=str(config["mass"]))
        ET.SubElement(
            inertial,
            "inertia",
            ixx="0.01",
            iyy="0.01",
            izz="0.01",
            ixy="0",
            ixz="0",
            iyz="0",
        )

        # Add visual
        visual = ET.SubElement(new_link, "visual")
        geometry = ET.SubElement(visual, "geometry")
        if config["geometry"] == "box":
            ET.SubElement(geometry, "box", size="0.1 0.1 0.1")
        elif config["geometry"] == "cylinder":
            ET.SubElement(geometry, "cylinder", radius="0.05", length="0.1")
        elif config["geometry"] == "sphere":
            ET.SubElement(geometry, "sphere", radius="0.05")
        else:
            ET.SubElement(geometry, "box", size="0.1 0.1 0.1")

        # Add collision (same as visual)
        collision = ET.SubElement(new_link, "collision")
        collision_geom = ET.SubElement(collision, "geometry")
        collision_geom.append(copy.deepcopy(list(geometry)[0]))

        root.append(new_link)

        # Create new joint connecting parent to new link
        new_joint = ET.Element("joint", name=joint_name, type=config["joint_type"])
        ET.SubElement(new_joint, "parent", link=parent_link)
        ET.SubElement(new_joint, "child", link=link_name)
        ET.SubElement(new_joint, "origin", xyz="0 0 0.1", rpy="0 0 0")

        axis = config["axis"]
        ET.SubElement(new_joint, "axis", xyz=f"{axis[0]} {axis[1]} {axis[2]}")

        if config["joint_type"] in ["revolute", "prismatic"]:
            ET.SubElement(
                new_joint,
                "limit",
                lower="-3.14",
                upper="3.14",
                effort="100",
                velocity="10",
            )

        root.append(new_joint)

        # Re-parent children if specified
        for child_name in config["reparent_children"]:
            # Find the joint that connects parent to this child
            for joint in root.findall("joint"):
                parent_elem = joint.find("parent")
                child_elem = joint.find("child")
                if (
                    parent_elem is not None
                    and child_elem is not None
                    and parent_elem.get("link") == parent_link
                    and child_elem.get("link") == child_name
                ):
                    # Change the parent to the new link
                    parent_elem.set("link", link_name)
                    break

        # Generate new URDF
        ET.indent(root, space="  ")
        new_content = ET.tostring(root, encoding="unicode", xml_declaration=True)

        self.urdf_content = new_content
        self.load_urdf(new_content)
        self.chain_modified.emit(new_content)

    def _on_remove_segment(self) -> None:
        """Handle remove segment button.

        Note: This feature requires careful handling of child re-parenting
        and is planned for a future release.
        """
        if not self.selected_node:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select a segment in the visualizer first.",
            )
            return

        QMessageBox.information(
            self,
            "Feature Coming Soon",
            "Remove Segment is planned for a future release.\n\n"
            "This feature will:\n"
            "• Remove the selected segment\n"
            "• Re-parent children to the removed segment's parent\n"
            "• Update all joint references automatically",
        )

    def _on_split_chain(self) -> None:
        """Handle split chain button.

        Note: This feature creates branch points and is planned
        for a future release.
        """
        if not self.selected_node:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select a link in the visualizer first.",
            )
            return

        QMessageBox.information(
            self,
            "Feature Coming Soon",
            "Split Chain is planned for a future release.\n\n"
            "This feature will:\n"
            "• Create a new branch point at the selected link\n"
            "• Allow duplicating children as a new branch\n"
            "• Support creating parallel kinematic chains",
        )

    def _on_merge_chains(self) -> None:
        """Handle merge chains button.

        Note: This feature merges leaf nodes and is planned
        for a future release.
        """
        QMessageBox.information(
            self,
            "Feature Coming Soon",
            "Merge Chains is planned for a future release.\n\n"
            "This feature will:\n"
            "• Allow selecting two leaf nodes\n"
            "• Connect the end of one chain to another\n"
            "• Create closed kinematic loops if needed",
        )

    def get_urdf_content(self) -> str:
        """Get the current URDF content."""
        return self.urdf_content
