"""Chain Manipulation Tools for URDF kinematic chain editing.

Provides tools for inserting segments into chains, editing branch structures,
and managing the kinematic hierarchy of URDF models.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ChainNode:
    """Represents a node in the kinematic chain."""

    name: str
    link_element: ET.Element | None = None
    joint_to_parent: ET.Element | None = None
    parent: ChainNode | None = None
    children: list[ChainNode] = field(default_factory=list)
    depth: int = 0

    def get_chain_to_root(self) -> list[ChainNode]:
        """Get the chain from this node to the root."""
        chain = [self]
        current = self.parent
        while current:
            chain.append(current)
            current = current.parent
        return list(reversed(chain))

    def get_all_descendants(self) -> list[ChainNode]:
        """Get all descendant nodes."""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0

    def is_end_effector(self) -> bool:
        """Check if this could be an end effector (leaf with common naming)."""
        if not self.is_leaf():
            return False
        lower_name = self.name.lower()
        end_effector_hints = [
            "hand",
            "gripper",
            "tool",
            "effector",
            "finger",
            "tip",
            "end",
            "head",
            "foot",
            "palm",
        ]
        return any(hint in lower_name for hint in end_effector_hints)


class KinematicTree:
    """Represents the kinematic tree structure of a URDF."""

    def __init__(self) -> None:
        """Initialize the kinematic tree."""
        self.root: ChainNode | None = None
        self.nodes: dict[str, ChainNode] = {}

    def build_from_urdf(self, urdf_content: str) -> None:
        """Build the tree from URDF XML content."""
        try:
            root_elem = ET.fromstring(urdf_content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse URDF: {e}")
            return

        self.nodes.clear()
        self.root = None

        # Extract links
        links = {}
        for link in root_elem.findall("link"):
            name = link.get("name", "")
            links[name] = link
            self.nodes[name] = ChainNode(name=name, link_element=link)

        # Extract joints and build hierarchy
        child_links = set()
        for joint in root_elem.findall("joint"):
            parent_elem = joint.find("parent")
            child_elem = joint.find("child")

            if parent_elem is None or child_elem is None:
                continue

            parent_name = parent_elem.get("link", "")
            child_name = child_elem.get("link", "")

            if parent_name in self.nodes and child_name in self.nodes:
                parent_node = self.nodes[parent_name]
                child_node = self.nodes[child_name]

                child_node.parent = parent_node
                child_node.joint_to_parent = joint
                parent_node.children.append(child_node)
                child_links.add(child_name)

        # Find root (link that is never a child)
        for name, node in self.nodes.items():
            if name not in child_links:
                if self.root is None:
                    self.root = node
                else:
                    # Multiple roots - use first one
                    logger.warning(
                        f"Multiple root links found. Using '{self.root.name}'"
                    )

        # Calculate depths
        self._calculate_depths()

    def _calculate_depths(self) -> None:
        """Calculate depth for each node."""
        if self.root is None:
            return

        def set_depth(node: ChainNode, depth: int) -> None:
            node.depth = depth
            for child in node.children:
                set_depth(child, depth + 1)

        set_depth(self.root, 0)

    def get_chain(self, from_link: str, to_link: str) -> list[ChainNode]:
        """Get the chain between two links.

        Args:
            from_link: Starting link name
            to_link: Ending link name

        Returns:
            List of nodes in the chain (may be empty if no path exists)
        """
        if from_link not in self.nodes or to_link not in self.nodes:
            return []

        from_node = self.nodes[from_link]
        to_node = self.nodes[to_link]

        # Get paths to root
        from_path = from_node.get_chain_to_root()
        to_path = to_node.get_chain_to_root()

        # Find common ancestor
        from_set = set(n.name for n in from_path)
        common_ancestor = None
        for node in to_path:
            if node.name in from_set:
                common_ancestor = node
                break

        if common_ancestor is None:
            return []

        # Build path
        chain = []

        # From from_link to common ancestor
        for node in from_path:
            chain.append(node)
            if node.name == common_ancestor.name:
                break

        # From common ancestor to to_link (reversed)
        to_chain = []
        for node in to_path:
            if node.name == common_ancestor.name:
                break
            to_chain.append(node)

        chain.extend(reversed(to_chain))
        return chain

    def get_all_chains(self) -> list[list[ChainNode]]:
        """Get all chains from root to leaves."""
        chains = []

        def collect_chains(node: ChainNode, current_chain: list[ChainNode]) -> None:
            current_chain = current_chain + [node]
            if node.is_leaf():
                chains.append(current_chain)
            else:
                for child in node.children:
                    collect_chains(child, current_chain)

        if self.root:
            collect_chains(self.root, [])

        return chains

    def get_end_effectors(self) -> list[ChainNode]:
        """Get all potential end effectors."""
        return [node for node in self.nodes.values() if node.is_leaf()]

    def get_branch_points(self) -> list[ChainNode]:
        """Get all branch points (nodes with multiple children)."""
        return [node for node in self.nodes.values() if len(node.children) > 1]


class ChainVisualizer(QGraphicsView):
    """Visual representation of the kinematic chain."""

    node_selected = pyqtSignal(str)  # Link name
    node_double_clicked = pyqtSignal(str)  # For insertion point

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the visualizer."""
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.tree: KinematicTree | None = None
        self.node_items: dict[str, QGraphicsEllipseItem] = {}

        # Visual settings
        self.node_radius = 20
        self.level_height = 80
        self.sibling_spacing = 60

    def set_tree(self, tree: KinematicTree) -> None:
        """Set the kinematic tree to visualize."""
        self.tree = tree
        self._render_tree()

    def _render_tree(self) -> None:
        """Render the kinematic tree."""
        self.scene.clear()
        self.node_items.clear()

        if self.tree is None or self.tree.root is None:
            return

        # Calculate positions
        positions = self._calculate_positions()

        # Draw edges first (so they appear behind nodes)
        for name, node in self.tree.nodes.items():
            if node.parent and name in positions and node.parent.name in positions:
                x1, y1 = positions[node.parent.name]
                x2, y2 = positions[name]
                line = QGraphicsLineItem(x1, y1, x2, y2)
                line.setPen(QColor("#888888"))
                self.scene.addItem(line)

        # Draw nodes
        for name, (x, y) in positions.items():
            node = self.tree.nodes[name]
            self._draw_node(node, x, y)

        # Fit view
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _calculate_positions(self) -> dict[str, tuple[float, float]]:
        """Calculate node positions using a simple tree layout."""
        positions: dict[str, tuple[float, float]] = {}

        if self.tree is None or self.tree.root is None:
            return positions

        # Count nodes at each depth
        depth_counts: dict[int, int] = {}
        depth_indices: dict[int, int] = {}

        for node in self.tree.nodes.values():
            if node.depth not in depth_counts:
                depth_counts[node.depth] = 0
                depth_indices[node.depth] = 0
            depth_counts[node.depth] += 1

        # Assign positions
        def assign_position(node: ChainNode) -> None:
            depth = node.depth
            count = depth_counts[depth]
            index = depth_indices[depth]

            x = (index - (count - 1) / 2) * self.sibling_spacing
            y = depth * self.level_height

            positions[node.name] = (x, y)
            depth_indices[depth] += 1

            for child in node.children:
                assign_position(child)

        assign_position(self.tree.root)
        return positions

    def _draw_node(self, node: ChainNode, x: float, y: float) -> None:
        """Draw a single node."""
        r = self.node_radius

        # Determine color based on node type
        if node.is_end_effector():
            color = QColor("#FF6B6B")  # Red for end effectors
        elif len(node.children) > 1:
            color = QColor("#4ECDC4")  # Teal for branch points
        elif node.parent is None:
            color = QColor("#45B7D1")  # Blue for root
        else:
            color = QColor("#96CEB4")  # Green for regular nodes

        # Draw ellipse
        ellipse = QGraphicsEllipseItem(x - r, y - r, r * 2, r * 2)
        ellipse.setBrush(color)
        ellipse.setPen(QColor("#333333"))
        ellipse.setData(0, node.name)
        self.scene.addItem(ellipse)
        self.node_items[node.name] = ellipse

        # Draw label
        text = QGraphicsTextItem(node.name)
        text.setPos(x - text.boundingRect().width() / 2, y + r + 2)
        font = text.font()
        font.setPointSize(8)
        text.setFont(font)
        self.scene.addItem(text)

    def mousePressEvent(self, event: Any) -> None:
        """Handle mouse press for node selection."""
        super().mousePressEvent(event)

        item = self.itemAt(event.pos())
        if isinstance(item, QGraphicsEllipseItem):
            name = item.data(0)
            if name:
                self.node_selected.emit(name)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        """Handle double-click for insertion point selection."""
        super().mouseDoubleClickEvent(event)

        item = self.itemAt(event.pos())
        if isinstance(item, QGraphicsEllipseItem):
            name = item.data(0)
            if name:
                self.node_double_clicked.emit(name)


class InsertSegmentDialog(QDialog):
    """Dialog for inserting a new segment into the chain."""

    def __init__(
        self,
        tree: KinematicTree,
        insert_after: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the dialog."""
        super().__init__(parent)
        self.tree = tree
        self.setWindowTitle("Insert Segment")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Insertion point
        insertion_group = QGroupBox("Insertion Point")
        insertion_layout = QFormLayout(insertion_group)

        self.parent_combo = QComboBox()
        for name in tree.nodes.keys():
            self.parent_combo.addItem(name)
        if insert_after:
            index = self.parent_combo.findText(insert_after)
            if index >= 0:
                self.parent_combo.setCurrentIndex(index)
        insertion_layout.addRow("Insert after link:", self.parent_combo)

        layout.addWidget(insertion_group)

        # New link properties
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

        layout.addWidget(link_group)

        # New joint properties
        joint_group = QGroupBox("New Joint")
        joint_layout = QFormLayout(joint_group)

        self.joint_name_edit = QLineEdit()
        self.joint_name_edit.setPlaceholderText("new_joint")
        joint_layout.addRow("Joint name:", self.joint_name_edit)

        self.joint_type_combo = QComboBox()
        self.joint_type_combo.addItems(["fixed", "revolute", "prismatic", "continuous"])
        joint_layout.addRow("Joint type:", self.joint_type_combo)

        # Axis
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

        layout.addWidget(joint_group)

        # Re-parenting option
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
                    item.setSelected(True)  # Select all by default
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

        layout.addWidget(reparent_group)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Update reparent list when parent changes
        self.parent_combo.currentTextChanged.connect(self._update_reparent_list)

    def _update_reparent_list(self, parent_name: str) -> None:
        """Update the reparent list when parent selection changes."""
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


class ChainManipulationWidget(QWidget):
    """Widget for manipulating kinematic chains."""

    chain_modified = pyqtSignal(str)  # Emits new URDF content

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the chain manipulation widget."""
        super().__init__(parent)
        self.tree = KinematicTree()
        self.urdf_content: str = ""
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

    def load_urdf(self, content: str) -> None:
        """Load URDF content and build the kinematic tree."""
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

    def _insert_segment(self, config: dict[str, Any]) -> None:
        """Insert a new segment into the URDF."""
        try:
            root = ET.fromstring(self.urdf_content)
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
