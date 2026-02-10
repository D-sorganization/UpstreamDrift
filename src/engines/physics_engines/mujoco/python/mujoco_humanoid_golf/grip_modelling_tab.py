"""Grip Modelling Tab for Advanced Hand Models.

Issue #757: Contact-based hand-grip model in MuJoCo with pressure visualization.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from src.shared.python.grip_contact_model import (
    GripContactExporter,
    GripContactModel,
    GripParameters,
    PressureVisualizationData,
    compute_pressure_visualization,
)
from src.shared.python.logging_config import get_logger

from .sim_widget import MuJoCoSimWidget

logger = get_logger(__name__)


class PressureVisualizationWidget(QtWidgets.QWidget):
    """Widget for visualizing grip pressure distribution.

    Issue #757: Pressure distribution visualization available in the UI.
    Displays pressure as a 2D heatmap (unwrapped grip cylinder).
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize pressure visualization widget."""
        super().__init__(parent)
        self.setMinimumSize(200, 150)
        self.pressure_data: PressureVisualizationData | None = None

        # Color map (blue -> green -> yellow -> red)
        self.color_stops = [
            (0.0, QtGui.QColor(0, 0, 255)),  # Blue (low)
            (0.33, QtGui.QColor(0, 255, 0)),  # Green
            (0.66, QtGui.QColor(255, 255, 0)),  # Yellow
            (1.0, QtGui.QColor(255, 0, 0)),  # Red (high)
        ]

    def update_pressure(self, data: PressureVisualizationData) -> None:
        """Update displayed pressure data.

        Args:
            data: New pressure visualization data
        """
        self.pressure_data = data
        self.update()

    def clear(self) -> None:
        """Clear pressure display."""
        self.pressure_data = None
        self.update()

    def _get_color_for_value(self, normalized_value: float) -> QtGui.QColor:
        """Get color from gradient for normalized value [0, 1]."""
        normalized_value = max(0.0, min(1.0, normalized_value))

        # Find surrounding color stops
        for i in range(len(self.color_stops) - 1):
            t1, c1 = self.color_stops[i]
            t2, c2 = self.color_stops[i + 1]

            if t1 <= normalized_value <= t2:
                # Interpolate
                t = (normalized_value - t1) / (t2 - t1) if t2 > t1 else 0
                r = int(c1.red() + t * (c2.red() - c1.red()))
                g = int(c1.green() + t * (c2.green() - c1.green()))
                b = int(c1.blue() + t * (c2.blue() - c1.blue()))
                return QtGui.QColor(r, g, b)

        return self.color_stops[-1][1]

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        """Paint the pressure visualization."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor(40, 40, 40))

        if self.pressure_data is None or len(self.pressure_data.pressures) == 0:
            painter.setPen(QtGui.QColor(150, 150, 150))
            painter.drawText(
                rect, QtCore.Qt.AlignmentFlag.AlignCenter, "No contact data"
            )
            return

        # Draw title
        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.drawText(10, 20, f"Max: {self.pressure_data.max_pressure:.0f} Pa")
        painter.drawText(10, 35, f"Mean: {self.pressure_data.mean_pressure:.0f} Pa")

        # Draw pressure points
        margin = 50
        plot_rect = rect.adjusted(margin, margin, -margin, -20)

        if plot_rect.width() <= 0 or plot_rect.height() <= 0:
            return

        # Map grip axis position to x, angular position to y
        axis_pos = self.pressure_data.grip_axis_positions
        angles = self.pressure_data.angular_positions

        if len(axis_pos) == 0:
            return

        # Normalize positions for display
        axis_min, axis_max = np.min(axis_pos), np.max(axis_pos)
        axis_range = axis_max - axis_min if axis_max > axis_min else 1.0

        for i in range(len(self.pressure_data.pressures)):
            # Map to widget coordinates
            x_norm = (axis_pos[i] - axis_min) / axis_range
            y_norm = (angles[i] + np.pi) / (2 * np.pi)

            x = int(plot_rect.left() + x_norm * plot_rect.width())
            y = int(plot_rect.top() + y_norm * plot_rect.height())

            # Size based on pressure (larger = more pressure)
            size = int(5 + 15 * self.pressure_data.normalized_pressures[i])

            # Color based on pressure
            norm_val = self.pressure_data.normalized_pressures[i]
            color = self._get_color_for_value(norm_val)
            painter.setBrush(QtGui.QBrush(color))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawEllipse(x - size // 2, y - size // 2, size, size)

        # Draw axes labels
        painter.setPen(QtGui.QColor(200, 200, 200))
        painter.drawText(plot_rect.left(), rect.bottom() - 5, "Butt")
        painter.drawText(plot_rect.right() - 20, rect.bottom() - 5, "Tip")

        # Draw color legend
        legend_rect = QtCore.QRect(rect.right() - 30, margin, 15, plot_rect.height())
        for i in range(legend_rect.height()):
            t = i / legend_rect.height()
            color = self._get_color_for_value(1.0 - t)  # Flip so high is at top
            painter.setPen(color)
            painter.drawLine(
                legend_rect.left(),
                legend_rect.top() + i,
                legend_rect.right(),
                legend_rect.top() + i,
            )


class ContactMetricsWidget(QtWidgets.QWidget):
    """Widget displaying contact metrics summary.

    Issue #757: Shows contact forces, slip detection status.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize metrics widget."""
        super().__init__(parent)
        layout = QtWidgets.QFormLayout(self)

        self.lbl_normal_force = QtWidgets.QLabel("0.0 N")
        self.lbl_tangent_force = QtWidgets.QLabel("0.0 N")
        self.lbl_num_contacts = QtWidgets.QLabel("0")
        self.lbl_slip_status = QtWidgets.QLabel("No slip")
        self.lbl_slip_margin = QtWidgets.QLabel("N/A")
        self.lbl_equilibrium = QtWidgets.QLabel("Unknown")

        layout.addRow("Normal Force:", self.lbl_normal_force)
        layout.addRow("Tangent Force:", self.lbl_tangent_force)
        layout.addRow("Active Contacts:", self.lbl_num_contacts)
        layout.addRow("Slip Status:", self.lbl_slip_status)
        layout.addRow("Min Slip Margin:", self.lbl_slip_margin)
        layout.addRow("Equilibrium:", self.lbl_equilibrium)

    def update_metrics(
        self,
        normal_force: float,
        tangent_force: float,
        num_contacts: int,
        num_slipping: int,
        slip_margin: float,
        equilibrium: bool,
    ) -> None:
        """Update displayed metrics."""
        self.lbl_normal_force.setText(f"{normal_force:.1f} N")
        self.lbl_tangent_force.setText(f"{tangent_force:.1f} N")
        self.lbl_num_contacts.setText(str(num_contacts))

        if num_slipping > 0:
            self.lbl_slip_status.setText(f"SLIPPING ({num_slipping})")
            self.lbl_slip_status.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.lbl_slip_status.setText("No slip")
            self.lbl_slip_status.setStyleSheet("color: green;")

        self.lbl_slip_margin.setText(f"{slip_margin:.2%}")

        if equilibrium:
            self.lbl_equilibrium.setText("Stable")
            self.lbl_equilibrium.setStyleSheet("color: green;")
        else:
            self.lbl_equilibrium.setText("Unstable")
            self.lbl_equilibrium.setStyleSheet("color: orange;")


class GripModellingTab(QtWidgets.QWidget):
    """Tab for manipulating advanced hand models (Shadow, Allegro)."""

    def connect_sim_widget(self, sim_widget: MuJoCoSimWidget) -> None:
        """Connect to an external simulation widget.

        Args:
           sim_widget: The main simulation widget to connect to.
        """
        # For now, we just store the reference, but we maintain our own internal widget
        # for independent visualization of the hand models.
        # Future work: Unify visualization if possible.
        self.external_sim_widget = sim_widget
        logger.info("Connected GripModellingTab to external sim widget")

    def __init__(self) -> None:
        """Initialize the grip modelling tab."""
        super().__init__()
        self.main_layout = QtWidgets.QHBoxLayout(self)

        # --- Left Control Panel ---
        self.control_panel = QtWidgets.QWidget()
        self.control_panel.setFixedWidth(300)
        self.control_layout = QtWidgets.QVBoxLayout(self.control_panel)

        # Model Selection
        self.control_layout.addWidget(QtWidgets.QLabel("<b>Hand Model Selection</b>"))
        self.combo_hand = QtWidgets.QComboBox()
        self.combo_hand.addItems(
            [
                "Shadow Hand Right",
                "Shadow Hand Left",
                "Shadow Hand Both",
                "Allegro Hand Right",
                "Allegro Hand Left",
            ]
        )

        self.combo_hand.currentIndexChanged.connect(self.load_current_hand_model)
        self.control_layout.addWidget(self.combo_hand)

        self.control_layout.addSpacing(10)

        # Physics Controls
        self.control_layout.addWidget(QtWidgets.QLabel("<b>Physics Controls</b>"))
        self.chk_kinematic = QtWidgets.QCheckBox("Kinematic Mode (Pose Only)")
        self.chk_kinematic.setToolTip(
            "Disable physics integration to pose hands without gravity/collisions"
        )
        self.chk_kinematic.setChecked(True)  # Default to kinematic for posing
        self.chk_kinematic.toggled.connect(self._on_kinematic_toggled)
        self.control_layout.addWidget(self.chk_kinematic)

        # Contact monitoring checkbox
        self.chk_contact_monitor = QtWidgets.QCheckBox("Monitor Contacts")
        self.chk_contact_monitor.setToolTip(
            "Enable contact force and slip monitoring (Issue #757)"
        )
        self.chk_contact_monitor.setChecked(False)
        self.chk_contact_monitor.toggled.connect(self._on_contact_monitor_toggled)
        self.control_layout.addWidget(self.chk_contact_monitor)

        self.control_layout.addSpacing(10)
        self.control_layout.addWidget(QtWidgets.QLabel("<b>Joint Controls</b>"))

        # Sliders Area
        self.sliders_area = QtWidgets.QScrollArea()
        self.sliders_area.setWidgetResizable(True)
        self.sliders_widget = QtWidgets.QWidget()
        self.sliders_layout = QtWidgets.QVBoxLayout(self.sliders_widget)
        self.sliders_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.sliders_area.setWidget(self.sliders_widget)

        self.control_layout.addWidget(self.sliders_area)

        self.main_layout.addWidget(self.control_panel)

        # --- Center Simulation Widget ---
        self.sim_widget = MuJoCoSimWidget(width=600, height=600)
        self.main_layout.addWidget(self.sim_widget, 2)

        # --- Right Panel: Contact Visualization (Issue #757) ---
        self.contact_panel = QtWidgets.QWidget()
        self.contact_panel.setFixedWidth(250)
        self.contact_layout = QtWidgets.QVBoxLayout(self.contact_panel)

        self.contact_layout.addWidget(QtWidgets.QLabel("<b>Contact Analysis</b>"))

        # Contact metrics widget
        self.metrics_widget = ContactMetricsWidget()
        self.contact_layout.addWidget(self.metrics_widget)

        self.contact_layout.addSpacing(10)
        self.contact_layout.addWidget(QtWidgets.QLabel("<b>Pressure Distribution</b>"))

        # Pressure visualization widget
        self.pressure_widget = PressureVisualizationWidget()
        self.pressure_widget.setMinimumHeight(200)
        self.contact_layout.addWidget(self.pressure_widget)

        # Export button
        self.btn_export_contacts = QtWidgets.QPushButton("Export Contact Data")
        self.btn_export_contacts.clicked.connect(self._export_contact_data)
        self.contact_layout.addWidget(self.btn_export_contacts)

        self.contact_layout.addStretch()
        self.main_layout.addWidget(self.contact_panel)

        # Internal state for sliders
        self.joint_sliders: list[QtWidgets.QSlider] = []
        self.joint_spinboxes: list[QtWidgets.QDoubleSpinBox] = []

        # Contact model (Issue #757)
        self.grip_contact_model = GripContactModel(GripParameters())
        self.contact_exporter = GripContactExporter(self.grip_contact_model)
        self.contact_timer: QtCore.QTimer | None = None

        # Initial Load
        QtCore.QTimer.singleShot(100, self.load_current_hand_model)

    def _on_kinematic_toggled(self, checked: bool) -> None:
        """Handle kinematic mode toggle."""
        if self.sim_widget:
            mode = "kinematic" if checked else "dynamic"
            self.sim_widget.set_operating_mode(mode)

    def load_current_hand_model(self) -> None:
        """Load the selected hand model with a test cylinder."""
        model_name = self.combo_hand.currentText()
        logger.info("Loading hand model: %s", model_name)

        base_path = Path(__file__).parent / "hand_assets"

        is_shadow = "Shadow" in model_name
        is_right = "Right" in model_name
        is_both = "Both" in model_name

        if is_shadow:
            folder = "shadow_hand"
            if is_both:
                scene_file = "scene_both.xml"
            else:
                scene_file = "scene_right.xml" if is_right else "scene_left.xml"
        else:
            folder = "wonik_allegro"
            scene_file = "scene_right.xml" if is_right else "scene_left.xml"

        scene_path = base_path / folder / scene_file
        folder_path = base_path / folder

        if not scene_path.exists():
            logger.error("Scene file not found: %s", scene_path)
            return

        try:
            xml_content = self._prepare_scene_xml(scene_path, folder_path, is_both)
        except (RuntimeError, ValueError, OSError):
            logger.exception("Failed to prepare XML model from %s", scene_path)
            return

        # Load into widget
        try:
            # Change directory to scene file location so relative assets (meshdir) work
            current_dir = os.getcwd()
            os.chdir(scene_path.parent)
            try:
                self.sim_widget.load_model_from_xml(xml_content)
            finally:
                os.chdir(current_dir)
        except (RuntimeError, ValueError, OSError):
            logger.exception("Failed to load XML model")
            return

        # Rebuild controls
        self.rebuild_joint_controls()

        # Apply initial kinematic state
        self._on_kinematic_toggled(self.chk_kinematic.isChecked())

    def _prepare_scene_xml(
        self, scene_path: Path, folder_path: Path, is_both: bool = False
    ) -> str:
        """Read scene file and inject absolute paths and cylinder object."""
        xml_content = scene_path.read_text("utf-8")

        # 1. Create movable versions of hand files
        # We need to inject <freejoint/> into the hand XMLs so they can be moved
        # by the mocap bodies we will add.

        # Helper function
        def get_hand_content(filename: str, body_name_pattern: str) -> str:
            full_path = folder_path / filename
            if not full_path.exists():
                return ""

            try:
                content = full_path.read_text("utf-8")

                # Check if freejoint already exists
                if "freejoint" not in content:
                    pattern = f'(<body[^>]*name="{body_name_pattern}"[^>]*>)'
                    match = re.search(pattern, content)
                    if match:
                        logger.info("Injecting freejoint into %s", filename)
                        insertion = match.group(1) + "\n      <freejoint/>"
                        content = content.replace(match.group(1), insertion)
                    else:
                        logger.warning(
                            "Could not find body '%s' in %s to inject freejoint",
                            body_name_pattern,
                            filename,
                        )

                # Strip <mujoco> tags to allow embedding
                content = re.sub(r"<mujoco[^>]*>", "", content)
                content = content.replace("</mujoco>", "")

                # When merging both hands, prefix default class names to avoid
                # collisions
                if is_both:
                    hand_prefix = "right" if "right" in filename.lower() else "left"
                    # Find all default class names
                    class_names = re.findall(r'<default class="([^"]+)">', content)
                    for class_name in set(class_names):
                        new_name = f"{hand_prefix}_{class_name}"
                        # Update definition
                        content = content.replace(
                            f'class="{class_name}"', f'class="{new_name}"'
                        )
                        # We don't need to update references inside the hand file
                        # because they are typically local to the <default> block
                        # or used in geoms/joints within the same subtree.
                        # However, to be safe, we replace all class="..." strings
                        # (this is simple but effective for these hand models).

                return content
            except (RuntimeError, ValueError, OSError):
                logger.exception("Failed to process hand file %s", filename)
                return ""  # Return empty only on catastrophic failure

        extracted_bodies = []

        def extract_worldbody_content(filename: str, body_pattern: str) -> str:
            content = get_hand_content(filename, body_pattern)
            # Extract worldbody content
            bodies_match = re.search(
                r"<worldbody[^>]*>(.*?)</worldbody>", content, re.DOTALL
            )
            if bodies_match:
                extracted_bodies.append(bodies_match.group(1))
                # Remove worldbody from content, leaving defaults/assets
                content = re.sub(
                    r"<worldbody[^>]*>.*?</worldbody>", "", content, flags=re.DOTALL
                )
            return content

        if is_both:
            right_defs = extract_worldbody_content("right_hand.xml", "rh_forearm")
            left_defs = extract_worldbody_content("left_hand.xml", "lh_forearm")

            xml_content = re.sub(
                r'<include[^>]*file="right_hand.xml"[^>]*/>', right_defs, xml_content
            )
            xml_content = re.sub(
                r'<include[^>]*file="left_hand.xml"[^>]*/>', left_defs, xml_content
            )
        else:
            if 'file="right_hand.xml"' in xml_content:
                target_body = "rh_forearm"
                if "allegro" in str(folder_path).lower():
                    target_body = "right_hand"

                defs = extract_worldbody_content("right_hand.xml", target_body)
                xml_content = re.sub(
                    r'<include[^>]*file="right_hand.xml"[^>]*/>',
                    defs,
                    xml_content,
                )
            elif 'file="left_hand.xml"' in xml_content:
                target_body = "lh_forearm"
                if "allegro" in str(folder_path):
                    target_body = "left_hand"

                defs = extract_worldbody_content("left_hand.xml", target_body)
                xml_content = re.sub(
                    r'<include[^>]*file="left_hand.xml"[^>]*/>',
                    defs,
                    xml_content,
                )

        # Inject extracted bodies into the scene's worldbody
        if extracted_bodies:
            bodies_str = "\n".join(extracted_bodies)
            # Insert at the beginning of worldbody
            xml_content = re.sub(
                r"(<worldbody[^>]*>)", r"\1\n" + bodies_str, xml_content, count=1
            )

        # Ensure offscreen framebuffer is large enough for renderer
        offscreen_global = '<global offwidth="1920" offheight="1080"/>'
        if "<visual>" in xml_content:
            if "<global" in xml_content:
                # Update existing global: strip slash, remove old attrs, add new ones
                def update_global_tag(m: re.Match) -> str:
                    # m.group(1) usually contains 'azimuth="..." /'
                    attrs = m.group(1).replace("/", "").strip()
                    attrs = re.sub(r'offwidth="[^"]*"', "", attrs)
                    attrs = re.sub(r'offheight="[^"]*"', "", attrs)
                    return f'<global {attrs} offwidth="1920" offheight="1080"/>'

                xml_content = re.sub(
                    r"<global([^>]*)>", update_global_tag, xml_content, count=1
                )
            else:
                # Insert global into visual
                xml_content = xml_content.replace(
                    "<visual>",
                    f"<visual>\n    {offscreen_global}",
                )
        else:
            # Add new visual section
            xml_content = xml_content.replace(
                "</mujoco>",
                f"<visual>\n  {offscreen_global}\n</visual>\n</mujoco>",
            )

        # 2. Inject Cylinder Object (only if not present)
        # Check for both the object name and unique geometry characteristics
        if (
            "club_handle" not in xml_content
            and 'name="club_handle"' not in xml_content
            and 'name="object"' not in xml_content
        ):
            cylinder_body = """
    <body name="club_handle" pos="0.3 0 0.1">
      <freejoint/>
      <geom type="cylinder" size="0.015 0.15" rgba="0.8 0.2 0.2 1"
            mass="0.3" condim="4" friction="1 0.5 0.5"/>
    </body>
        """
            # Insert before the last </worldbody>
            last_worldbody_end = xml_content.rfind("</worldbody>")
            if last_worldbody_end != -1:
                xml_content = (
                    xml_content[:last_worldbody_end]
                    + f"{cylinder_body}\n  "
                    + xml_content[last_worldbody_end:]
                )

        # 3. Inject Mocap Bodies and Welds for Hands
        # This allows moving the hands "around in space" using the mocap bodies
        mocap_xml = ""
        equality_xml = "<equality>\n"

        # Right Hand Mocap (only add if not already present)
        if (
            is_both or "right" in str(scene_path).lower()
        ) and 'name="rh_mocap"' not in xml_content:
            mocap_xml += """
    <body name="rh_mocap" mocap="true" pos="0 0 0">
        <geom type="box" size="0.02 0.02 0.02" rgba="0 1 0 0.5" contype="0"
              conaffinity="0"/>
    </body>
            """
            equality_xml += (
                '    <weld body1="rh_mocap" body2="rh_forearm" solref="0.02 1" '
                'solimp="0.9 0.95 0.001"/>\n'
            )

        # Left Hand Mocap (only add if not already present)
        if (
            is_both or "left" in str(scene_path).lower()
        ) and 'name="lh_mocap"' not in xml_content:
            mocap_xml += """
    <body name="lh_mocap" mocap="true" pos="0 0 0">
        <geom type="box" size="0.02 0.02 0.02" rgba="1 0 0 0.5" contype="0"
              conaffinity="0"/>
    </body>
            """
            equality_xml += (
                '    <weld body1="lh_mocap" body2="lh_forearm" solref="0.02 1" '
                'solimp="0.9 0.95 0.001"/>\n'
            )

        equality_xml += "  </equality>"

        # Insert Mocap bodies before the last </worldbody>
        if mocap_xml:
            last_worldbody_end = xml_content.rfind("</worldbody>")
            if last_worldbody_end != -1:
                xml_content = (
                    xml_content[:last_worldbody_end]
                    + f"{mocap_xml}\n  "
                    + xml_content[last_worldbody_end:]
                )

        # Insert Equality section before </mujoco> (or merge if exists)
        if "</equality>" in xml_content:
            # If equality section exists, insert inside
            # (Simplified check, might need robust parsing if complex)
            # If equality section exists, insert inside
            equality_content = (
                equality_xml.strip()
                .replace("<equality>", "")
                .replace("</equality>", "")
            )
            xml_content = xml_content.replace(
                "</equality>", f"{equality_content}\n  </equality>"
            )

        else:
            xml_content = xml_content.replace("</mujoco>", f"{equality_xml}\n</mujoco>")

        logger.info(
            "Successfully prepared scene XML with movable hands and mocap bodies."
        )
        return xml_content

    def rebuild_joint_controls(self) -> None:
        """Rebuild the joint control widgets for the current model."""
        # Clear existing
        while self.sliders_layout.count():
            item = self.sliders_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget:
                    widget.deleteLater()

        self.joint_sliders.clear()
        self.joint_spinboxes.clear()

        if self.sim_widget.model is None or self.sim_widget.data is None:
            return

        # Iterate joints
        model = self.sim_widget.model

        for i in range(model.njnt):
            self._add_joint_control_row(i, model)

    def _add_joint_control_row(
        self, i: int, model: mujoco.MjModel
    ) -> None:  # noqa: PLR0915
        """Create a control row for a single joint."""
        if self.sim_widget.data is None:
            return

        # Skip free joints and ball joints (multi-dof)
        jnt_type = model.jnt_type[i]
        if jnt_type in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            return

        if self.sim_widget.data is None:
            return

        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not name:
            name = f"Joint {i}"

        # Create UI row
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        label = QtWidgets.QLabel(name)
        label.setFixedWidth(120)
        row_layout.addWidget(label)

        # Range
        range_min, range_max = self._get_joint_range(i, model)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, 1000)

        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(range_min, range_max)
        spin.setSingleStep(0.01)

        # Initial value (qpos) - Assuming qpos address matches joint id for 1-dof joints
        # Need strict qpos address.
        qpos_adr = model.jnt_qposadr[i]
        init_val = self.sim_widget.data.qpos[qpos_adr]

        slider.setValue(self._val_to_slider(init_val, range_min, range_max))
        spin.setValue(init_val)

        # Connect
        def _on_slider_change(
            v: int,
            s: Any = spin,
            amin: float = range_min,
            amax: float = range_max,
            idx: int = qpos_adr,
        ) -> None:
            self._on_slider(v, s, amin, amax, idx)

        slider.valueChanged.connect(_on_slider_change)

        def _on_spin_change(
            v: float,
            s: Any = slider,
            amin: float = range_min,
            amax: float = range_max,
            idx: int = qpos_adr,
        ) -> None:
            self._on_spin(v, s, amin, amax, idx)

        spin.valueChanged.connect(_on_spin_change)

        row_layout.addWidget(slider)
        row_layout.addWidget(spin)

        self.sliders_layout.addWidget(row)
        self.joint_sliders.append(slider)
        self.joint_spinboxes.append(spin)

    def _val_to_slider(self, val: float, min_v: float, max_v: float) -> int:
        """Convert float value to slider integer position."""
        ratio = (val - min_v) / (max_v - min_v) if max_v > min_v else 0.5
        return int(ratio * 1000)

    def _slider_to_val(self, slider_val: int, min_v: float, max_v: float) -> float:
        """Convert slider integer position to float value."""
        ratio = slider_val / 1000.0
        return min_v + ratio * (max_v - min_v)

    def _update_joint(self, q_idx: int, val: float) -> None:
        """Update joint value in simulation."""
        if self.sim_widget.model is None or self.sim_widget.data is None:
            return
        self.sim_widget.data.qpos[q_idx] = val
        mujoco.mj_forward(self.sim_widget.model, self.sim_widget.data)
        self.sim_widget.render()

    def _on_slider(  # noqa: PLR0913
        self,
        val_int: int,
        spin: QtWidgets.QDoubleSpinBox,
        min_v: float,
        max_v: float,
        q_idx: int,
    ) -> None:
        """Handle slider value change."""
        val = self._slider_to_val(val_int, min_v, max_v)
        spin.blockSignals(True)  # noqa: FBT003
        spin.setValue(val)
        spin.blockSignals(False)  # noqa: FBT003
        self._update_joint(q_idx, val)

    def _on_spin(  # noqa: PLR0913
        self,
        val: float,
        slider: QtWidgets.QSlider,
        min_v: float,
        max_v: float,
        q_idx: int,
    ) -> None:
        """Handle spinbox value change."""
        slider_val = self._val_to_slider(val, min_v, max_v)
        slider.blockSignals(True)  # noqa: FBT003
        slider.setValue(slider_val)
        slider.blockSignals(False)  # noqa: FBT003
        self._update_joint(q_idx, val)

    def _get_joint_range(self, i: int, model: mujoco.MjModel) -> tuple[float, float]:
        """Get valid joint range, providing defaults if undefined."""
        range_min, range_max = (
            model.jnt_range[i] if model.jnt_range is not None else (-np.pi, np.pi)
        )
        if range_min == 0 and range_max == 0:
            return -np.pi, np.pi
        return range_min, range_max

    # -------------------------------------------------------------------------
    # Contact Monitoring Methods (Issue #757)
    # -------------------------------------------------------------------------

    def _on_contact_monitor_toggled(self, checked: bool) -> None:
        """Handle contact monitoring toggle."""
        if checked:
            self._start_contact_monitoring()
        else:
            self._stop_contact_monitoring()

    def _start_contact_monitoring(self) -> None:
        """Start periodic contact monitoring."""
        if self.contact_timer is None:
            self.contact_timer = QtCore.QTimer(self)
            self.contact_timer.timeout.connect(self._update_contact_data)

        self.contact_exporter.reset()
        self.contact_timer.start(50)  # 20 Hz update rate
        logger.info("Contact monitoring started")

    def _stop_contact_monitoring(self) -> None:
        """Stop contact monitoring."""
        if self.contact_timer is not None:
            self.contact_timer.stop()
        logger.info("Contact monitoring stopped")

    def _update_contact_data(self) -> None:
        """Update contact data from MuJoCo simulation.

        Extracts contact information from MuJoCo and updates visualizations.
        """
        if self.sim_widget.model is None or self.sim_widget.data is None:
            return

        model = self.sim_widget.model
        data = self.sim_widget.data

        # Extract contacts from MuJoCo
        n_contacts = data.ncon

        if n_contacts == 0:
            self.pressure_widget.clear()
            self.metrics_widget.update_metrics(0, 0, 0, 0, 0.0, False)
            return

        # Collect contact data
        positions = []
        normals = []
        forces = []
        velocities = []
        body_names = []

        for i in range(n_contacts):
            contact = data.contact[i]

            # Get contact position and normal
            pos = contact.pos.copy()
            normal = contact.frame[:3].copy()  # First 3 elements are normal

            # Get contact force (need to use mj_contactForce)
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            contact_force = force[:3]  # Linear force components

            # Estimate velocity at contact (simplified)
            vel = np.zeros(3)  # Would need body velocities for accurate computation

            # Get body names
            geom1 = contact.geom1
            geom2 = contact.geom2
            body1_id = model.geom_bodyid[geom1]
            body2_id = model.geom_bodyid[geom2]
            body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

            # Filter for hand-grip contacts (simplified heuristic)
            is_hand_contact = any(
                name and ("hand" in name.lower() or "finger" in name.lower())
                for name in [body1_name, body2_name]
            )

            if is_hand_contact:
                positions.append(pos)
                normals.append(normal)
                forces.append(contact_force)
                velocities.append(vel)
                body_names.append(body1_name or "unknown")

        if not positions:
            self.pressure_widget.clear()
            self.metrics_widget.update_metrics(0, 0, 0, 0, 0.0, False)
            return

        # Update grip contact model
        positions_arr = np.array(positions)
        normals_arr = np.array(normals)
        forces_arr = np.array(forces)
        velocities_arr = np.array(velocities)
        timestamp = data.time

        state = self.grip_contact_model.update_from_mujoco(
            positions_arr,
            normals_arr,
            forces_arr,
            velocities_arr,
            body_names,
            timestamp,
        )

        # Capture for export
        self.contact_exporter.capture_timestep()

        # Update pressure visualization
        if len(positions_arr) > 0:
            grip_center = np.mean(positions_arr, axis=0)
        else:
            grip_center = np.zeros(3)
        pressure_data = compute_pressure_visualization(
            state.contacts,
            grip_center,
            contact_area=self.grip_contact_model.params.hand_contact_area,
        )
        self.pressure_widget.update_pressure(pressure_data)

        # Update metrics display
        margins = self.grip_contact_model.check_slip_margin()
        # ~3N typical club weight
        equilibrium = self.grip_contact_model.check_static_equilibrium(3.0)

        self.metrics_widget.update_metrics(
            normal_force=state.total_normal_force,
            tangent_force=float(np.linalg.norm(state.total_tangent_force)),
            num_contacts=len(state.contacts),
            num_slipping=state.num_slipping,
            slip_margin=margins["min_margin"],
            equilibrium=equilibrium.get("equilibrium", False),
        )

    def _export_contact_data(self) -> None:
        """Export captured contact data to file."""
        if not self.contact_exporter.timesteps:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No contact data captured. Enable contact monitoring first.",
            )
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Contact Data", "", "JSON Files (*.json);;CSV Files (*.csv)"
        )

        if not filename:
            return

        try:
            if filename.endswith(".csv"):
                import csv

                data = self.contact_exporter.export_to_csv_data()
                if data:
                    with open(filename, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
            else:
                import json

                data = self.contact_exporter.export_to_dict()
                with open(filename, "w") as f:
                    json.dump(data, f, indent=2)

            # Show summary
            summary = self.contact_exporter.get_summary_statistics()
            QtWidgets.QMessageBox.information(
                self,
                "Export Complete",
                f"Contact data exported to {filename}\n\n"
                f"Timesteps: {summary['num_timesteps']}\n"
                f"Duration: {summary['duration']:.2f}s\n"
                f"Mean Force: {summary['force_mean']:.1f}N\n"
                f"Slip Detected: {'Yes' if summary['any_slip_detected'] else 'No'}",
            )

        except ImportError as e:
            logger.exception("Failed to export contact data")
            QtWidgets.QMessageBox.critical(
                self, "Export Failed", f"Failed to export: {e}"
            )
