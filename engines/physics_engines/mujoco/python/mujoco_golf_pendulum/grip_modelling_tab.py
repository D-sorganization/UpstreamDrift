"""Grip Modelling Tab for Advanced Hand Models."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import mujoco
import numpy as np
from PyQt6 import QtCore, QtWidgets

from .sim_widget import MuJoCoSimWidget

logger = logging.getLogger(__name__)


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

        # --- Right Simulation Widget ---
        self.sim_widget = MuJoCoSimWidget(width=800, height=800)
        self.main_layout.addWidget(self.sim_widget, 1)

        # Internal state for sliders
        self.joint_sliders: list[QtWidgets.QSlider] = []
        self.joint_spinboxes: list[QtWidgets.QDoubleSpinBox] = []

        # Initial Load
        QtCore.QTimer.singleShot(100, self.load_current_hand_model)

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
        except Exception:
            logger.exception("Failed to prepare XML model")
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
        except Exception:
            logger.exception("Failed to load XML model")
            return

        # Rebuild controls
        self.rebuild_joint_controls()

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

                # Strip <mujoco> tags to allow embedding
                content = re.sub(r"<mujoco[^>]*>", "", content)
                content = content.replace("</mujoco>", "")
                return content
            except Exception:
                logger.exception("Failed to process hand file %s", filename)
                return ""

        if is_both:
            right_content = get_hand_content("right_hand.xml", "rh_forearm")
            left_content = get_hand_content("left_hand.xml", "lh_forearm")
            xml_content = re.sub(
                r'<include[^>]*file="right_hand.xml"[^>]*/>', right_content, xml_content
            )
            xml_content = re.sub(
                r'<include[^>]*file="left_hand.xml"[^>]*/>', left_content, xml_content
            )
        else:
            if 'file="right_hand.xml"' in xml_content:
                hand_content = get_hand_content("right_hand.xml", "rh_forearm")
                xml_content = re.sub(
                    r'<include[^>]*file="right_hand.xml"[^>]*/>',
                    hand_content,
                    xml_content,
                )
            elif 'file="left_hand.xml"' in xml_content:
                hand_content = get_hand_content("left_hand.xml", "lh_forearm")
                xml_content = re.sub(
                    r'<include[^>]*file="left_hand.xml"[^>]*/>',
                    hand_content,
                    xml_content,
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
            # Insert before </worldbody>
            xml_content = xml_content.replace(
                "</worldbody>", f"{cylinder_body}\n  </worldbody>"
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
        <geom type="box" size="0.02 0.02 0.02" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
    </body>
            """
            equality_xml += '    <weld body1="rh_mocap" body2="rh_forearm" solref="0.02 1" solimp="0.9 0.95 0.001"/>\n'

        # Left Hand Mocap (only add if not already present)
        if (
            is_both or "left" in str(scene_path).lower()
        ) and 'name="lh_mocap"' not in xml_content:
            mocap_xml += """
    <body name="lh_mocap" mocap="true" pos="0 0 0">
        <geom type="box" size="0.02 0.02 0.02" rgba="1 0 0 0.5" contype="0" conaffinity="0"/>
    </body>
            """
            equality_xml += '    <weld body1="lh_mocap" body2="lh_forearm" solref="0.02 1" solimp="0.9 0.95 0.001"/>\n'

        equality_xml += "  </equality>"

        # Insert Mocap bodies before </worldbody>
        xml_content = xml_content.replace(
            "</worldbody>", f"{mocap_xml}\n  </worldbody>"
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
        slider.valueChanged.connect(
            lambda v, s=spin, amin=range_min, amax=range_max, idx=qpos_adr: self._on_slider(
                v, s, amin, amax, idx
            )
        )
        spin.valueChanged.connect(
            lambda v, s=slider, amin=range_min, amax=range_max, idx=qpos_adr: (
                self._on_spin(v, s, amin, amax, idx)
            )
        )

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
