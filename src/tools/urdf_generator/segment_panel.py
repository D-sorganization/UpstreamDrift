"""Segment management panel for the URDF Generator."""

from src.shared.python.logging_config import get_logger

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

logger = get_logger(__name__)


class SegmentPanel(QWidget):
    """Panel for managing URDF segments."""

    # Signals
    segment_added = pyqtSignal(dict)
    segment_removed = pyqtSignal(str)
    segment_modified = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None):
        """Initialize the segment panel.

        Args:
            parent: Parent widget, if any.
        """
        super().__init__(parent)
        self.segments: list[dict] = []
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Segment list
        self._setup_segment_list(layout)

        # Segment editor tabs
        self._setup_segment_editor(layout)

        # Control buttons
        self._setup_control_buttons(layout)

    def _setup_segment_list(self, parent_layout: QVBoxLayout) -> None:
        """Set up the segment list widget.

        Args:
            parent_layout: Parent layout to add to.
        """
        group = QGroupBox("Segments")
        layout = QVBoxLayout(group)

        self.segment_list = QListWidget()
        self.segment_list.setMaximumHeight(150)
        layout.addWidget(self.segment_list)

        parent_layout.addWidget(group)

    def _setup_segment_editor(self, parent_layout: QVBoxLayout) -> None:
        """Set up the segment editor tabs.

        Args:
            parent_layout: Parent layout to add to.
        """
        self.editor_tabs = QTabWidget()

        # Basic properties tab
        self._setup_basic_tab()

        # Geometry tab
        self._setup_geometry_tab()

        # Physics tab
        self._setup_physics_tab()

        # Joint tab
        self._setup_joint_tab()

        parent_layout.addWidget(self.editor_tabs)

    def _setup_basic_tab(self) -> None:
        """Set up the basic properties tab."""
        tab = QWidget()
        layout = QFormLayout(tab)

        # Segment name
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter segment name")
        layout.addRow("Name:", self.name_edit)

        # Segment type
        self.type_combo = QComboBox()
        self.type_combo.addItems(
            [
                "Link",
                "Joint",
                "Golf Club Shaft",
                "Golf Club Head",
                "Golf Ball",
                "Tee",
                "Ground Plane",
            ]
        )
        layout.addRow("Type:", self.type_combo)

        # Parent segment
        self.parent_combo = QComboBox()
        self.parent_combo.addItem("None (Root)")
        layout.addRow("Parent:", self.parent_combo)

        self.editor_tabs.addTab(tab, "Basic")

    def _setup_geometry_tab(self) -> None:
        """Set up the geometry tab."""
        tab = QWidget()
        layout = QFormLayout(tab)

        # Shape type
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Box", "Cylinder", "Sphere", "Mesh", "Capsule"])
        layout.addRow("Shape:", self.shape_combo)

        # Dimensions
        dimensions_group = QGroupBox("Dimensions")
        dim_layout = QFormLayout(dimensions_group)

        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(0.001, 10.0)
        self.length_spin.setValue(1.0)
        self.length_spin.setSuffix(" m")
        dim_layout.addRow("Length:", self.length_spin)

        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.001, 10.0)
        self.width_spin.setValue(0.1)
        self.width_spin.setSuffix(" m")
        dim_layout.addRow("Width:", self.width_spin)

        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.001, 10.0)
        self.height_spin.setValue(0.1)
        self.height_spin.setSuffix(" m")
        dim_layout.addRow("Height:", self.height_spin)

        layout.addRow(dimensions_group)

        # Position
        position_group = QGroupBox("Position")
        pos_layout = QFormLayout(position_group)

        self.pos_x_spin = QDoubleSpinBox()
        self.pos_x_spin.setRange(-10.0, 10.0)
        self.pos_x_spin.setSuffix(" m")
        pos_layout.addRow("X:", self.pos_x_spin)

        self.pos_y_spin = QDoubleSpinBox()
        self.pos_y_spin.setRange(-10.0, 10.0)
        self.pos_y_spin.setSuffix(" m")
        pos_layout.addRow("Y:", self.pos_y_spin)

        self.pos_z_spin = QDoubleSpinBox()
        self.pos_z_spin.setRange(-10.0, 10.0)
        self.pos_z_spin.setSuffix(" m")
        pos_layout.addRow("Z:", self.pos_z_spin)

        layout.addRow(position_group)

        # Orientation
        orientation_group = QGroupBox("Orientation (RPY)")
        ori_layout = QFormLayout(orientation_group)

        self.roll_spin = QDoubleSpinBox()
        self.roll_spin.setRange(-180.0, 180.0)
        self.roll_spin.setSuffix("°")
        ori_layout.addRow("Roll:", self.roll_spin)

        self.pitch_spin = QDoubleSpinBox()
        self.pitch_spin.setRange(-180.0, 180.0)
        self.pitch_spin.setSuffix("°")
        ori_layout.addRow("Pitch:", self.pitch_spin)

        self.yaw_spin = QDoubleSpinBox()
        self.yaw_spin.setRange(-180.0, 180.0)
        self.yaw_spin.setSuffix("°")
        ori_layout.addRow("Yaw:", self.yaw_spin)

        layout.addRow(orientation_group)

        self.editor_tabs.addTab(tab, "Geometry")

    def _setup_physics_tab(self) -> None:
        """Set up the physics properties tab."""
        tab = QWidget()
        layout = QFormLayout(tab)

        # Mass
        self.mass_spin = QDoubleSpinBox()
        self.mass_spin.setRange(0.001, 1000.0)
        self.mass_spin.setValue(1.0)
        self.mass_spin.setSuffix(" kg")
        layout.addRow("Mass:", self.mass_spin)

        # Inertia (simplified)
        inertia_group = QGroupBox("Inertia")
        inertia_layout = QFormLayout(inertia_group)

        self.ixx_spin = QDoubleSpinBox()
        self.ixx_spin.setRange(0.0, 100.0)
        self.ixx_spin.setValue(0.1)
        inertia_layout.addRow("Ixx:", self.ixx_spin)

        self.iyy_spin = QDoubleSpinBox()
        self.iyy_spin.setRange(0.0, 100.0)
        self.iyy_spin.setValue(0.1)
        inertia_layout.addRow("Iyy:", self.iyy_spin)

        self.izz_spin = QDoubleSpinBox()
        self.izz_spin.setRange(0.0, 100.0)
        self.izz_spin.setValue(0.1)
        inertia_layout.addRow("Izz:", self.izz_spin)

        layout.addRow(inertia_group)

        # Material properties
        material_group = QGroupBox("Material")
        material_layout = QFormLayout(material_group)

        self.material_name_edit = QLineEdit()
        self.material_name_edit.setPlaceholderText("Material name")
        material_layout.addRow("Name:", self.material_name_edit)

        # Color (RGBA)
        color_group = QGroupBox("Color (RGBA)")
        color_layout = QFormLayout(color_group)

        self.color_r_spin = QDoubleSpinBox()
        self.color_r_spin.setRange(0.0, 1.0)
        self.color_r_spin.setValue(0.8)
        color_layout.addRow("Red:", self.color_r_spin)

        self.color_g_spin = QDoubleSpinBox()
        self.color_g_spin.setRange(0.0, 1.0)
        self.color_g_spin.setValue(0.8)
        color_layout.addRow("Green:", self.color_g_spin)

        self.color_b_spin = QDoubleSpinBox()
        self.color_b_spin.setRange(0.0, 1.0)
        self.color_b_spin.setValue(0.8)
        color_layout.addRow("Blue:", self.color_b_spin)

        self.color_a_spin = QDoubleSpinBox()
        self.color_a_spin.setRange(0.0, 1.0)
        self.color_a_spin.setValue(1.0)
        color_layout.addRow("Alpha:", self.color_a_spin)

        material_layout.addRow(color_group)
        layout.addRow(material_group)

        self.editor_tabs.addTab(tab, "Physics")

    def _setup_joint_tab(self) -> None:
        """Set up the joint properties tab."""
        tab = QWidget()
        layout = QFormLayout(tab)

        # Joint type
        self.joint_type_combo = QComboBox()
        self.joint_type_combo.addItems(
            ["fixed", "revolute", "prismatic", "continuous", "floating", "planar"]
        )
        layout.addRow("Joint Type:", self.joint_type_combo)

        # Joint axis
        axis_group = QGroupBox("Joint Axis")
        axis_layout = QFormLayout(axis_group)

        self.axis_x_spin = QDoubleSpinBox()
        self.axis_x_spin.setRange(-1.0, 1.0)
        self.axis_x_spin.setValue(0.0)
        axis_layout.addRow("X:", self.axis_x_spin)

        self.axis_y_spin = QDoubleSpinBox()
        self.axis_y_spin.setRange(-1.0, 1.0)
        self.axis_y_spin.setValue(0.0)
        axis_layout.addRow("Y:", self.axis_y_spin)

        self.axis_z_spin = QDoubleSpinBox()
        self.axis_z_spin.setRange(-1.0, 1.0)
        self.axis_z_spin.setValue(1.0)
        axis_layout.addRow("Z:", self.axis_z_spin)

        layout.addRow(axis_group)

        # Joint limits
        limits_group = QGroupBox("Joint Limits")
        limits_layout = QFormLayout(limits_group)

        self.lower_limit_spin = QDoubleSpinBox()
        self.lower_limit_spin.setRange(-360.0, 360.0)
        self.lower_limit_spin.setValue(-180.0)
        self.lower_limit_spin.setSuffix("°")
        limits_layout.addRow("Lower:", self.lower_limit_spin)

        self.upper_limit_spin = QDoubleSpinBox()
        self.upper_limit_spin.setRange(-360.0, 360.0)
        self.upper_limit_spin.setValue(180.0)
        self.upper_limit_spin.setSuffix("°")
        limits_layout.addRow("Upper:", self.upper_limit_spin)

        self.velocity_limit_spin = QDoubleSpinBox()
        self.velocity_limit_spin.setRange(0.0, 1000.0)
        self.velocity_limit_spin.setValue(10.0)
        self.velocity_limit_spin.setSuffix(" rad/s")
        limits_layout.addRow("Velocity:", self.velocity_limit_spin)

        self.effort_limit_spin = QDoubleSpinBox()
        self.effort_limit_spin.setRange(0.0, 10000.0)
        self.effort_limit_spin.setValue(100.0)
        self.effort_limit_spin.setSuffix(" N⋅m")
        limits_layout.addRow("Effort:", self.effort_limit_spin)

        layout.addRow(limits_group)

        self.editor_tabs.addTab(tab, "Joint")

    def _setup_control_buttons(self, parent_layout: QVBoxLayout) -> None:
        """Set up the control buttons.

        Args:
            parent_layout: Parent layout to add to.
        """
        button_layout = QHBoxLayout()

        self.add_button = QPushButton("Add Segment")
        self.add_button.clicked.connect(self._add_segment)
        button_layout.addWidget(self.add_button)

        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self._remove_segment)
        self.remove_button.setEnabled(False)
        button_layout.addWidget(self.remove_button)

        self.update_button = QPushButton("Update")
        self.update_button.clicked.connect(self._update_segment)
        self.update_button.setEnabled(False)
        button_layout.addWidget(self.update_button)

        parent_layout.addLayout(button_layout)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self.segment_list.currentItemChanged.connect(self._on_segment_selected)
        self.type_combo.currentTextChanged.connect(self._on_type_changed)

    def _on_segment_selected(
        self, current: QListWidgetItem, previous: QListWidgetItem
    ) -> None:
        """Handle segment selection.

        Args:
            current: Currently selected item.
            previous: Previously selected item.
        """
        if current:
            self.remove_button.setEnabled(True)
            self.update_button.setEnabled(True)
            # Load segment data into editor
            segment_name = current.text()
            self._load_segment_data(segment_name)
        else:
            self.remove_button.setEnabled(False)
            self.update_button.setEnabled(False)

    def _on_type_changed(self, segment_type: str) -> None:
        """Handle segment type change.

        Args:
            segment_type: New segment type.
        """
        # Update default values based on type
        if segment_type == "Golf Club Shaft":
            self.length_spin.setValue(1.0)
            self.width_spin.setValue(0.02)
            self.height_spin.setValue(0.02)
            self.mass_spin.setValue(0.3)
        elif segment_type == "Golf Club Head":
            self.length_spin.setValue(0.1)
            self.width_spin.setValue(0.05)
            self.height_spin.setValue(0.03)
            self.mass_spin.setValue(0.2)
        elif segment_type == "Golf Ball":
            self.shape_combo.setCurrentText("Sphere")
            self.length_spin.setValue(0.043)  # Golf ball diameter
            self.width_spin.setValue(0.043)
            self.height_spin.setValue(0.043)
            self.mass_spin.setValue(0.046)  # Golf ball mass

    def _add_segment(self) -> None:
        """Add a new segment."""
        segment_data = self._get_segment_data()

        if not segment_data["name"]:
            return  # Name is required

        # Check for duplicate names
        if any(seg["name"] == segment_data["name"] for seg in self.segments):
            return  # Duplicate name

        self.segments.append(segment_data)

        # Add to list widget
        item = QListWidgetItem(segment_data["name"])
        self.segment_list.addItem(item)

        # Update parent combo for other segments
        self.parent_combo.addItem(segment_data["name"])

        self.segment_added.emit(segment_data)
        logger.info(f"Segment added: {segment_data['name']}")

    def _remove_segment(self) -> None:
        """Remove the selected segment."""
        current_item = self.segment_list.currentItem()
        if not current_item:
            return

        segment_name = current_item.text()

        # Remove from segments list
        self.segments = [seg for seg in self.segments if seg["name"] != segment_name]

        # Remove from list widget
        row = self.segment_list.row(current_item)
        self.segment_list.takeItem(row)

        # Remove from parent combo
        index = self.parent_combo.findText(segment_name)
        if index >= 0:
            self.parent_combo.removeItem(index)

        self.segment_removed.emit(segment_name)
        logger.info(f"Segment removed: {segment_name}")

    def _update_segment(self) -> None:
        """Update the selected segment."""
        current_item = self.segment_list.currentItem()
        if not current_item:
            return

        segment_data = self._get_segment_data()
        old_name = current_item.text()

        # Update segments list
        for i, seg in enumerate(self.segments):
            if seg["name"] == old_name:
                self.segments[i] = segment_data
                break

        # Update list widget if name changed
        if segment_data["name"] != old_name:
            current_item.setText(segment_data["name"])

            # Update parent combo
            index = self.parent_combo.findText(old_name)
            if index >= 0:
                self.parent_combo.setItemText(index, segment_data["name"])

        self.segment_modified.emit(segment_data)
        logger.info(f"Segment updated: {segment_data['name']}")

    def _get_segment_data(self) -> dict:
        """Get segment data from the editor.

        Returns:
            Dictionary containing segment data.
        """
        return {
            "name": self.name_edit.text(),
            "type": self.type_combo.currentText(),
            "parent": (
                self.parent_combo.currentText()
                if self.parent_combo.currentText() != "None (Root)"
                else None
            ),
            "geometry": {
                "shape": self.shape_combo.currentText(),
                "dimensions": {
                    "length": self.length_spin.value(),
                    "width": self.width_spin.value(),
                    "height": self.height_spin.value(),
                },
                "position": {
                    "x": self.pos_x_spin.value(),
                    "y": self.pos_y_spin.value(),
                    "z": self.pos_z_spin.value(),
                },
                "orientation": {
                    "roll": self.roll_spin.value(),
                    "pitch": self.pitch_spin.value(),
                    "yaw": self.yaw_spin.value(),
                },
            },
            "physics": {
                "mass": self.mass_spin.value(),
                "inertia": {
                    "ixx": self.ixx_spin.value(),
                    "iyy": self.iyy_spin.value(),
                    "izz": self.izz_spin.value(),
                },
                "material": {
                    "name": self.material_name_edit.text(),
                    "color": {
                        "r": self.color_r_spin.value(),
                        "g": self.color_g_spin.value(),
                        "b": self.color_b_spin.value(),
                        "a": self.color_a_spin.value(),
                    },
                },
            },
            "joint": {
                "type": self.joint_type_combo.currentText(),
                "axis": {
                    "x": self.axis_x_spin.value(),
                    "y": self.axis_y_spin.value(),
                    "z": self.axis_z_spin.value(),
                },
                "limits": {
                    "lower": self.lower_limit_spin.value(),
                    "upper": self.upper_limit_spin.value(),
                    "velocity": self.velocity_limit_spin.value(),
                    "effort": self.effort_limit_spin.value(),
                },
            },
        }

    def _load_segment_data(self, segment_name: str) -> None:
        """Load segment data into the editor.

        Args:
            segment_name: Name of the segment to load.
        """
        segment = next(
            (seg for seg in self.segments if seg["name"] == segment_name), None
        )
        if not segment:
            return

        # Load basic properties
        self.name_edit.setText(segment["name"])
        self.type_combo.setCurrentText(segment["type"])
        parent_text = segment["parent"] if segment["parent"] else "None (Root)"
        self.parent_combo.setCurrentText(parent_text)

        # Load geometry
        geom = segment["geometry"]
        self.shape_combo.setCurrentText(geom["shape"])
        self.length_spin.setValue(geom["dimensions"]["length"])
        self.width_spin.setValue(geom["dimensions"]["width"])
        self.height_spin.setValue(geom["dimensions"]["height"])
        self.pos_x_spin.setValue(geom["position"]["x"])
        self.pos_y_spin.setValue(geom["position"]["y"])
        self.pos_z_spin.setValue(geom["position"]["z"])
        self.roll_spin.setValue(geom["orientation"]["roll"])
        self.pitch_spin.setValue(geom["orientation"]["pitch"])
        self.yaw_spin.setValue(geom["orientation"]["yaw"])

        # Load physics
        physics = segment["physics"]
        self.mass_spin.setValue(physics["mass"])
        self.ixx_spin.setValue(physics["inertia"]["ixx"])
        self.iyy_spin.setValue(physics["inertia"]["iyy"])
        self.izz_spin.setValue(physics["inertia"]["izz"])
        self.material_name_edit.setText(physics["material"]["name"])
        self.color_r_spin.setValue(physics["material"]["color"]["r"])
        self.color_g_spin.setValue(physics["material"]["color"]["g"])
        self.color_b_spin.setValue(physics["material"]["color"]["b"])
        self.color_a_spin.setValue(physics["material"]["color"]["a"])

        # Load joint
        joint = segment["joint"]
        self.joint_type_combo.setCurrentText(joint["type"])
        self.axis_x_spin.setValue(joint["axis"]["x"])
        self.axis_y_spin.setValue(joint["axis"]["y"])
        self.axis_z_spin.setValue(joint["axis"]["z"])
        self.lower_limit_spin.setValue(joint["limits"]["lower"])
        self.upper_limit_spin.setValue(joint["limits"]["upper"])
        self.velocity_limit_spin.setValue(joint["limits"]["velocity"])
        self.effort_limit_spin.setValue(joint["limits"]["effort"])

    def clear(self) -> None:
        """Clear all segments."""
        self.segments.clear()
        self.segment_list.clear()

        # Reset parent combo to just "None (Root)"
        self.parent_combo.clear()
        self.parent_combo.addItem("None (Root)")

        # Clear editor fields
        self.name_edit.clear()
        self.type_combo.setCurrentIndex(0)
        self.parent_combo.setCurrentIndex(0)
