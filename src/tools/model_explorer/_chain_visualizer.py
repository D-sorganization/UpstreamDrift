from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QWidget,
)

from src.tools.model_explorer._chain_model import ChainNode, KinematicTree


class ChainVisualizer(QGraphicsView):
    """Visual representation of the kinematic chain."""

    node_selected = pyqtSignal(str)  # Link name
    node_double_clicked = pyqtSignal(str)  # For insertion point

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the visualizer."""
        super().__init__(parent)
        self._graphics_scene = QGraphicsScene(self)
        self.setScene(self._graphics_scene)
        self.tree: KinematicTree | None = None
        self.node_items: dict[str, QGraphicsEllipseItem] = {}

        # Visual settings
        self.node_radius = 20
        self.level_height = 80
        self.sibling_spacing = 60

    def set_tree(self, tree: KinematicTree) -> None:
        """Set the kinematic tree to visualize."""
        if tree is None:
            raise ValueError("tree must be provided")
        self.tree = tree
        self._render_tree()

    def _render_tree(self) -> None:
        """Render the kinematic tree."""
        self._graphics_scene.clear()
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
                self._graphics_scene.addItem(line)

        # Draw nodes
        for name, (x, y) in positions.items():
            node = self.tree.nodes[name]
            self._draw_node(node, x, y)

        # Fit view
        self.fitInView(
            self._graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

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
            """Assign x/y coordinates to a node based on its depth and index."""
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
        if node is None:
            raise ValueError("node must be provided")
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
        self._graphics_scene.addItem(ellipse)
        self.node_items[node.name] = ellipse

        # Draw label
        text = QGraphicsTextItem(node.name)
        text.setPos(x - text.boundingRect().width() / 2, y + r + 2)
        font = text.font()
        font.setPointSize(8)
        text.setFont(font)
        self._graphics_scene.addItem(text)

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
