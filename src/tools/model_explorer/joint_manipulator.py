"""Joint Manipulator - Auto-load and manipulate URDF joints.

Provides tools for automatically loading joints from URDFs,
visualizing joint configurations, and interactively manipulating
joint parameters.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import defusedxml.ElementTree as DefusedET
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.core.contracts import postcondition, precondition
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class JointInfo:
    """Information about a URDF joint."""

    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower_limit: float | None
    upper_limit: float | None
    effort_limit: float | None
    velocity_limit: float | None
    damping: float
    friction: float
    current_position: float = 0.0

    @classmethod
    def from_element(cls, element: ET.Element) -> JointInfo:
        """Create JointInfo from XML element."""
        name = element.get("name", "unnamed")
        joint_type = element.get("type", "fixed")

        # Parent and child
        parent = element.find("parent")
        child = element.find("child")
        parent_link = parent.get("link", "") if parent is not None else ""
        child_link = child.get("link", "") if child is not None else ""

        # Origin
        origin = element.find("origin")
        if origin is not None:
            xyz_str = origin.get("xyz", "0 0 0")
            rpy_str = origin.get("rpy", "0 0 0")
            xyz = tuple(float(x) for x in xyz_str.split())
            rpy = tuple(float(x) for x in rpy_str.split())
        else:
            xyz = (0.0, 0.0, 0.0)
            rpy = (0.0, 0.0, 0.0)

        # Axis
        axis_elem = element.find("axis")
        if axis_elem is not None:
            axis_str = axis_elem.get("xyz", "0 0 1")
            axis = tuple(float(x) for x in axis_str.split())
        else:
            axis = (0.0, 0.0, 1.0)

        # Limits
        limit = element.find("limit")
        if limit is not None:
            lower = float(limit.get("lower", "0"))
            upper = float(limit.get("upper", "0"))
            effort = float(limit.get("effort", "0"))
            velocity = float(limit.get("velocity", "0"))
        else:
            lower = None
            upper = None
            effort = None
            velocity = None

        # Dynamics
        dynamics = element.find("dynamics")
        if dynamics is not None:
            damping = float(dynamics.get("damping", "0"))
            friction = float(dynamics.get("friction", "0"))
        else:
            damping = 0.0
            friction = 0.0

        return cls(
            name=name,
            joint_type=joint_type,
            parent_link=parent_link,
            child_link=child_link,
            origin_xyz=xyz,  # type: ignore
            origin_rpy=rpy,  # type: ignore
            axis=axis,  # type: ignore
            lower_limit=lower,
            upper_limit=upper,
            effort_limit=effort,
            velocity_limit=velocity,
            damping=damping,
            friction=friction,
        )

    def is_movable(self) -> bool:
        """Check if this joint type allows movement."""
        return self.joint_type in ["revolute", "prismatic", "continuous"]

    def get_position_range(self) -> tuple[float, float]:
        """Get the valid position range for this joint."""
        if self.joint_type == "continuous":
            return (-math.pi * 2, math.pi * 2)
        elif self.lower_limit is not None and self.upper_limit is not None:
            return (self.lower_limit, self.upper_limit)
        else:
            return (-math.pi, math.pi)

    def get_position_unit(self) -> str:
        """Get the unit for joint position."""
        if self.joint_type == "prismatic":
            return "m"
        else:
            return "rad"


class JointSliderWidget(QWidget):
    """Widget for controlling a single joint with a slider."""

    value_changed = pyqtSignal(str, float)  # joint_name, value

    def __init__(self, joint: JointInfo, parent: QWidget | None = None) -> None:
        """Initialize the joint slider widget."""
        super().__init__(parent)
        self.joint = joint
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)

        # Joint name
        name_label = QLabel(self.joint.name)
        name_label.setMinimumWidth(120)
        name_label.setToolTip(
            f"Type: {self.joint.joint_type}\n"
            f"Parent: {self.joint.parent_link}\n"
            f"Child: {self.joint.child_link}"
        )
        layout.addWidget(name_label)

        # Type indicator
        type_label = QLabel(f"[{self.joint.joint_type[0].upper()}]")
        type_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(type_label)

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimumWidth(150)

        min_val, max_val = self.joint.get_position_range()
        # Scale to integer for slider (100 steps per unit)
        self.scale = 100
        self.slider.setMinimum(int(min_val * self.scale))
        self.slider.setMaximum(int(max_val * self.scale))
        self.slider.setValue(int(self.joint.current_position * self.scale))

        layout.addWidget(self.slider)

        # Value spinbox
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setValue(self.joint.current_position)
        self.spinbox.setSingleStep(0.01)
        self.spinbox.setDecimals(3)
        self.spinbox.setSuffix(f" {self.joint.get_position_unit()}")
        self.spinbox.setMinimumWidth(100)
        layout.addWidget(self.spinbox)

        # Reset button
        self.reset_btn = QPushButton("0")
        self.reset_btn.setFixedWidth(30)
        self.reset_btn.setToolTip("Reset to zero")
        layout.addWidget(self.reset_btn)

        # Disable if not movable
        if not self.joint.is_movable():
            self.slider.setEnabled(False)
            self.spinbox.setEnabled(False)
            self.reset_btn.setEnabled(False)

    def _connect_signals(self) -> None:
        """Connect signals."""
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)
        self.reset_btn.clicked.connect(self._on_reset)

    def _on_slider_changed(self, value: int) -> None:
        """Handle slider value change."""
        float_value = value / self.scale
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(float_value)
        self.spinbox.blockSignals(False)
        self.joint.current_position = float_value
        self.value_changed.emit(self.joint.name, float_value)

    def _on_spinbox_changed(self, value: float) -> None:
        """Handle spinbox value change."""
        self.slider.blockSignals(True)
        self.slider.setValue(int(value * self.scale))
        self.slider.blockSignals(False)
        self.joint.current_position = value
        self.value_changed.emit(self.joint.name, value)

    def _on_reset(self) -> None:
        """Reset joint to zero."""
        self.spinbox.setValue(0.0)

    def set_value(self, value: float) -> None:
        """Set the joint value programmatically."""
        self.spinbox.setValue(value)

    def get_value(self) -> float:
        """Get the current joint value."""
        return self.spinbox.value()


class JointTableWidget(QWidget):
    """Table view of all joints with editing capabilities."""

    joint_modified = pyqtSignal(str, dict)  # joint_name, new_properties

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the joint table widget."""
        super().__init__(parent)
        self.joints: dict[str, JointInfo] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Type", "Parent", "Child", "Lower", "Upper", "Axis", "Movable"]
        )

        header = self.table.horizontalHeader()
        if header:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            for i in range(1, 8):
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self.table)

        # Summary
        self.summary_label = QLabel("No joints loaded")
        layout.addWidget(self.summary_label)

    def load_joints(self, joints: dict[str, JointInfo]) -> None:
        """Load joints into the table."""
        self.joints = joints
        self.table.setRowCount(len(joints))

        movable_count = 0
        for row, (name, joint) in enumerate(joints.items()):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(joint.joint_type))
            self.table.setItem(row, 2, QTableWidgetItem(joint.parent_link))
            self.table.setItem(row, 3, QTableWidgetItem(joint.child_link))

            lower = f"{joint.lower_limit:.3f}" if joint.lower_limit is not None else "-"
            upper = f"{joint.upper_limit:.3f}" if joint.upper_limit is not None else "-"
            self.table.setItem(row, 4, QTableWidgetItem(lower))
            self.table.setItem(row, 5, QTableWidgetItem(upper))

            axis_str = (
                f"({joint.axis[0]:.1f}, {joint.axis[1]:.1f}, {joint.axis[2]:.1f})"
            )
            self.table.setItem(row, 6, QTableWidgetItem(axis_str))

            movable = "Yes" if joint.is_movable() else "No"
            item = QTableWidgetItem(movable)
            if joint.is_movable():
                item.setForeground(QColor("#006400"))
                movable_count += 1
            else:
                item.setForeground(QColor("#888888"))
            self.table.setItem(row, 7, item)

        self.summary_label.setText(
            f"Total joints: {len(joints)} | Movable: {movable_count} | "
            f"Fixed: {len(joints) - movable_count}"
        )


class JointEditorPanel(QWidget):
    """Panel for editing individual joint properties."""

    joint_updated = pyqtSignal(str, ET.Element)  # joint_name, new_element

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the joint editor panel."""
        super().__init__(parent)
        self.current_joint: JointInfo | None = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Joint selection
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("Edit Joint:"))
        self.joint_combo = QComboBox()
        select_layout.addWidget(self.joint_combo)
        layout.addLayout(select_layout)

        # Basic properties
        basic_group = QGroupBox("Basic Properties")
        basic_layout = QFormLayout(basic_group)

        self.name_edit = QLineEdit()
        basic_layout.addRow("Name:", self.name_edit)

        self.type_combo = QComboBox()
        self.type_combo.addItems(
            ["fixed", "revolute", "prismatic", "continuous", "floating", "planar"]
        )
        basic_layout.addRow("Type:", self.type_combo)

        layout.addWidget(basic_group)

        # Axis
        axis_group = QGroupBox("Joint Axis")
        axis_layout = QHBoxLayout(axis_group)

        self.axis_x = QDoubleSpinBox()
        self.axis_x.setRange(-1, 1)
        self.axis_x.setSingleStep(0.1)
        axis_layout.addWidget(QLabel("X:"))
        axis_layout.addWidget(self.axis_x)

        self.axis_y = QDoubleSpinBox()
        self.axis_y.setRange(-1, 1)
        self.axis_y.setSingleStep(0.1)
        axis_layout.addWidget(QLabel("Y:"))
        axis_layout.addWidget(self.axis_y)

        self.axis_z = QDoubleSpinBox()
        self.axis_z.setRange(-1, 1)
        self.axis_z.setSingleStep(0.1)
        axis_layout.addWidget(QLabel("Z:"))
        axis_layout.addWidget(self.axis_z)

        layout.addWidget(axis_group)

        # Limits
        limits_group = QGroupBox("Joint Limits")
        limits_layout = QFormLayout(limits_group)

        self.lower_spin = QDoubleSpinBox()
        self.lower_spin.setRange(-100, 100)
        self.lower_spin.setDecimals(4)
        limits_layout.addRow("Lower:", self.lower_spin)

        self.upper_spin = QDoubleSpinBox()
        self.upper_spin.setRange(-100, 100)
        self.upper_spin.setDecimals(4)
        limits_layout.addRow("Upper:", self.upper_spin)

        self.effort_spin = QDoubleSpinBox()
        self.effort_spin.setRange(0, 10000)
        limits_layout.addRow("Effort:", self.effort_spin)

        self.velocity_spin = QDoubleSpinBox()
        self.velocity_spin.setRange(0, 1000)
        limits_layout.addRow("Velocity:", self.velocity_spin)

        layout.addWidget(limits_group)

        # Dynamics
        dynamics_group = QGroupBox("Dynamics")
        dynamics_layout = QFormLayout(dynamics_group)

        self.damping_spin = QDoubleSpinBox()
        self.damping_spin.setRange(0, 1000)
        self.damping_spin.setDecimals(4)
        dynamics_layout.addRow("Damping:", self.damping_spin)

        self.friction_spin = QDoubleSpinBox()
        self.friction_spin.setRange(0, 1000)
        self.friction_spin.setDecimals(4)
        dynamics_layout.addRow("Friction:", self.friction_spin)

        layout.addWidget(dynamics_group)

        # Apply button
        self.apply_btn = QPushButton("Apply Changes")
        self.apply_btn.setEnabled(False)
        layout.addWidget(self.apply_btn)

        layout.addStretch()

    def _connect_signals(self) -> None:
        """Connect signals."""
        self.joint_combo.currentTextChanged.connect(self._on_joint_selected)
        self.apply_btn.clicked.connect(self._on_apply)

    def load_joints(self, joints: dict[str, JointInfo]) -> None:
        """Load joints into the combo box."""
        self.joint_combo.clear()
        self.joint_combo.addItems(joints.keys())

    def _on_joint_selected(self, name: str) -> None:
        """Handle joint selection."""
        # This would load the joint's current values into the editor
        self.apply_btn.setEnabled(bool(name))

    @precondition(
        lambda self, joint: joint is not None and hasattr(joint, "name"),
        "Joint must be a valid JointInfo object",
    )
    def set_joint(self, joint: JointInfo) -> None:
        """Set the joint to edit."""
        self.current_joint = joint

        self.name_edit.setText(joint.name)
        index = self.type_combo.findText(joint.joint_type)
        if index >= 0:
            self.type_combo.setCurrentIndex(index)

        self.axis_x.setValue(joint.axis[0])
        self.axis_y.setValue(joint.axis[1])
        self.axis_z.setValue(joint.axis[2])

        if joint.lower_limit is not None:
            self.lower_spin.setValue(joint.lower_limit)
        if joint.upper_limit is not None:
            self.upper_spin.setValue(joint.upper_limit)
        if joint.effort_limit is not None:
            self.effort_spin.setValue(joint.effort_limit)
        if joint.velocity_limit is not None:
            self.velocity_spin.setValue(joint.velocity_limit)

        self.damping_spin.setValue(joint.damping)
        self.friction_spin.setValue(joint.friction)

        self.apply_btn.setEnabled(True)

    def _on_apply(self) -> None:
        """Apply changes to the joint."""
        if self.current_joint is None:
            return

        # Build new joint element
        joint_elem = ET.Element(
            "joint",
            name=self.name_edit.text(),
            type=self.type_combo.currentText(),
        )

        ET.SubElement(joint_elem, "parent", link=self.current_joint.parent_link)
        ET.SubElement(joint_elem, "child", link=self.current_joint.child_link)

        xyz = self.current_joint.origin_xyz
        rpy = self.current_joint.origin_rpy
        ET.SubElement(
            joint_elem,
            "origin",
            xyz=f"{xyz[0]} {xyz[1]} {xyz[2]}",
            rpy=f"{rpy[0]} {rpy[1]} {rpy[2]}",
        )

        axis = f"{self.axis_x.value()} {self.axis_y.value()} {self.axis_z.value()}"
        ET.SubElement(joint_elem, "axis", xyz=axis)

        if self.type_combo.currentText() in ["revolute", "prismatic"]:
            ET.SubElement(
                joint_elem,
                "limit",
                lower=str(self.lower_spin.value()),
                upper=str(self.upper_spin.value()),
                effort=str(self.effort_spin.value()),
                velocity=str(self.velocity_spin.value()),
            )

        if self.damping_spin.value() > 0 or self.friction_spin.value() > 0:
            ET.SubElement(
                joint_elem,
                "dynamics",
                damping=str(self.damping_spin.value()),
                friction=str(self.friction_spin.value()),
            )

        self.joint_updated.emit(self.current_joint.name, joint_elem)


class JointManipulatorWidget(QWidget):
    """Main widget for joint manipulation with auto-loading."""

    joints_updated = pyqtSignal(dict)  # dict[str, float] joint positions
    urdf_modified = pyqtSignal(str)  # new URDF content

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the joint manipulator widget."""
        super().__init__(parent)
        self.joints: dict[str, JointInfo] = {}
        self.urdf_content: str = ""
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Header with controls
        header = QHBoxLayout()

        self.auto_load_btn = QPushButton("Auto-Load Joints")
        header.addWidget(self.auto_load_btn)

        self.reset_all_btn = QPushButton("Reset All")
        header.addWidget(self.reset_all_btn)

        self.random_btn = QPushButton("Random Pose")
        header.addWidget(self.random_btn)

        header.addStretch()

        # Filter
        header.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Movable Only", "Fixed Only"])
        header.addWidget(self.filter_combo)

        layout.addLayout(header)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side - sliders
        sliders_widget = QWidget()
        sliders_layout = QVBoxLayout(sliders_widget)

        sliders_layout.addWidget(QLabel("Joint Sliders (interactive control):"))

        # Scroll area for sliders
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.sliders_container = QWidget()
        self.sliders_layout = QVBoxLayout(self.sliders_container)
        self.sliders_layout.addStretch()
        scroll.setWidget(self.sliders_container)

        sliders_layout.addWidget(scroll)

        splitter.addWidget(sliders_widget)

        # Right side - table and editor
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Table
        self.table_widget = JointTableWidget()
        right_layout.addWidget(self.table_widget)

        # Editor
        self.editor_panel = JointEditorPanel()
        right_layout.addWidget(self.editor_panel)

        splitter.addWidget(right_widget)

        layout.addWidget(splitter)

        # Status
        self.status_label = QLabel("Load a URDF to auto-detect joints")
        layout.addWidget(self.status_label)

    def _connect_signals(self) -> None:
        """Connect signals."""
        self.auto_load_btn.clicked.connect(self._on_auto_load)
        self.reset_all_btn.clicked.connect(self._on_reset_all)
        self.random_btn.clicked.connect(self._on_random_pose)
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        self.editor_panel.joint_updated.connect(self._on_joint_updated)

    @precondition(
        lambda self, content: content is not None and len(content.strip()) > 0,
        "URDF content must be a non-empty string",
    )
    def load_urdf(self, content: str) -> None:
        """Load URDF content and auto-detect joints."""
        self.urdf_content = content
        self._on_auto_load()

    def _on_auto_load(self) -> None:
        """Auto-load joints from the URDF."""
        if not self.urdf_content:
            self.status_label.setText("No URDF loaded")
            return

        try:
            root = DefusedET.fromstring(self.urdf_content)
        except ET.ParseError as e:
            self.status_label.setText(f"Parse error: {e}")
            return

        self.joints.clear()

        # Extract all joints
        for joint_elem in root.findall("joint"):
            joint_info = JointInfo.from_element(joint_elem)
            self.joints[joint_info.name] = joint_info

        # Update UI
        self._populate_sliders()
        self.table_widget.load_joints(self.joints)
        self.editor_panel.load_joints(self.joints)

        movable = sum(1 for j in self.joints.values() if j.is_movable())
        self.status_label.setText(
            f"Loaded {len(self.joints)} joints ({movable} movable)"
        )

    def _populate_sliders(self) -> None:
        """Populate the sliders container."""
        # Clear existing sliders
        while self.sliders_layout.count() > 1:  # Keep the stretch
            item = self.sliders_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        # Add sliders for movable joints
        filter_mode = self.filter_combo.currentText()

        for joint in self.joints.values():
            if filter_mode == "Movable Only" and not joint.is_movable():
                continue
            if filter_mode == "Fixed Only" and joint.is_movable():
                continue

            slider_widget = JointSliderWidget(joint)
            slider_widget.value_changed.connect(self._on_joint_value_changed)
            self.sliders_layout.insertWidget(
                self.sliders_layout.count() - 1, slider_widget
            )

    def _on_filter_changed(self, filter_text: str) -> None:
        """Handle filter change."""
        self._populate_sliders()

    def _on_joint_value_changed(self, name: str, value: float) -> None:
        """Handle joint value change from slider."""
        if name in self.joints:
            self.joints[name].current_position = value

        # Emit all current positions
        positions = {name: j.current_position for name, j in self.joints.items()}
        self.joints_updated.emit(positions)

    def _on_reset_all(self) -> None:
        """Reset all joints to zero."""
        for i in range(self.sliders_layout.count() - 1):
            item = self.sliders_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if isinstance(widget, JointSliderWidget):
                    widget.set_value(0.0)

    def _on_random_pose(self) -> None:
        """Set random poses for all movable joints."""
        import random

        for i in range(self.sliders_layout.count() - 1):
            item = self.sliders_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if isinstance(widget, JointSliderWidget) and widget.joint.is_movable():
                    min_val, max_val = widget.joint.get_position_range()
                    random_val = random.uniform(min_val, max_val)
                    widget.set_value(random_val)

    def _on_joint_updated(self, name: str, new_element: ET.Element) -> None:
        """Handle joint update from editor."""
        if not self.urdf_content:
            return

        try:
            root = DefusedET.fromstring(self.urdf_content)
        except ET.ParseError:
            return

        # Find and replace the joint
        for joint in root.findall("joint"):
            if joint.get("name") == name:
                root.remove(joint)
                root.append(new_element)
                break

        # Generate new URDF
        ET.indent(root, space="  ")
        new_content = ET.tostring(root, encoding="unicode", xml_declaration=True)

        self.urdf_content = new_content
        self._on_auto_load()  # Reload joints
        self.urdf_modified.emit(new_content)
        self.status_label.setText(f"Updated joint '{name}'")

    @postcondition(
        lambda result: result is not None and isinstance(result, dict),
        "Joint positions must be returned as a non-None dictionary",
    )
    def get_joint_positions(self) -> dict[str, float]:
        """Get current positions of all joints."""
        return {name: j.current_position for name, j in self.joints.items()}

    @precondition(
        lambda self, positions: positions is not None and isinstance(positions, dict),
        "Joint positions must be a non-None dictionary",
    )
    def set_joint_positions(self, positions: dict[str, float]) -> None:
        """Set joint positions."""
        for i in range(self.sliders_layout.count() - 1):
            item = self.sliders_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if (
                    isinstance(widget, JointSliderWidget)
                    and widget.joint.name in positions
                ):
                    widget.set_value(positions[widget.joint.name])

    def get_urdf_content(self) -> str:
        """Get the current URDF content."""
        return self.urdf_content
