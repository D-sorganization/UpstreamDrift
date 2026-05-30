# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.  # noqa: E501
# It requires domain-aware structural extraction to isolate its internal classes appropriately.  # noqa: E501

"""Grip Modelling Tab for Advanced Hand Models.

Issue #757: Contact-based hand-grip model in MuJoCo with pressure visualization.

Implementation split across:
- _grip_modelling_widgets.py: PressureVisualizationWidget, ContactMetricsWidget
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PyQt6 import QtCore, QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.physics.grip_contact_model import (
    GripContactExporter,
    GripContactModel,
    GripParameters,
    compute_pressure_visualization,
)

from ._grip_modelling_synergies import (
    AddSynergyDialog,
    Synergy,
    SynergyJointBinding,
    get_descriptive_joint_name,
)

# Re-export public names for backward compatibility
from ._grip_modelling_widgets import ContactMetricsWidget, PressureVisualizationWidget
from ._grip_modelling_xml import prepare_scene_xml
from .sim_widget import MuJoCoSimWidget

logger = get_logger(__name__)


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

        if sim_widget is None:
            raise ValueError("sim_widget must be provided")
        self.external_sim_widget = sim_widget
        logger.info("Connected GripModellingTab to external sim widget")

    def __init__(self) -> None:
        """Initialize the grip modelling tab."""
        super().__init__()
        self.main_layout = QtWidgets.QHBoxLayout(self)

        self._setup_left_control_panel()
        self._setup_center_sim_widget()
        self._setup_right_contact_panel()
        self._init_internal_state()

        QtCore.QTimer.singleShot(100, self.load_current_hand_model)

    def _setup_left_control_panel(self) -> None:
        self.control_panel = QtWidgets.QWidget()
        self.control_panel.setFixedWidth(450)
        self.control_layout = QtWidgets.QVBoxLayout(self.control_panel)

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

        self._setup_physics_controls()
        self._setup_synergy_area()
        self._setup_sliders_area()

        self.main_layout.addWidget(self.control_panel)

    def _setup_physics_controls(self) -> None:
        self.control_layout.addWidget(QtWidgets.QLabel("<b>Physics Controls</b>"))
        self.chk_kinematic = QtWidgets.QCheckBox("Kinematic Mode (Pose Only)")
        self.chk_kinematic.setToolTip(
            "Disable physics integration to pose hands without gravity/collisions"
        )
        self.chk_kinematic.setChecked(True)
        self.chk_kinematic.toggled.connect(self._on_kinematic_toggled)
        self.control_layout.addWidget(self.chk_kinematic)

        self.chk_contact_monitor = QtWidgets.QCheckBox("Monitor Contacts")
        self.chk_contact_monitor.setToolTip(
            "Enable contact force and slip monitoring (Issue #757)"
        )  # noqa: E501
        self.chk_contact_monitor.setChecked(False)
        self.chk_contact_monitor.toggled.connect(self._on_contact_monitor_toggled)
        self.control_layout.addWidget(self.chk_contact_monitor)

        self.control_layout.addSpacing(10)
        self.control_layout.addWidget(QtWidgets.QLabel("<b>Joint Controls</b>"))

    def _setup_synergy_area(self) -> None:
        self.control_layout.addWidget(
            QtWidgets.QLabel("<b>Synergy (Linked) Sliders</b>")
        )
        self.synergy_area = QtWidgets.QScrollArea()
        self.synergy_area.setWidgetResizable(True)
        self.synergy_widget = QtWidgets.QWidget()
        self.synergy_layout = QtWidgets.QVBoxLayout(self.synergy_widget)
        self.synergy_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.synergy_area.setWidget(self.synergy_widget)
        self.synergy_area.setFixedHeight(120)
        self.control_layout.addWidget(self.synergy_area)

        self.btn_add_synergy = QtWidgets.QPushButton("+ Add Custom Synergy")
        self.btn_add_synergy.clicked.connect(self._on_add_custom_synergy)
        self.control_layout.addWidget(self.btn_add_synergy)
        self.control_layout.addSpacing(10)

    def _setup_sliders_area(self) -> None:
        self.sliders_area = QtWidgets.QScrollArea()
        self.sliders_area.setWidgetResizable(True)
        self.sliders_widget = QtWidgets.QWidget()
        self.sliders_layout = QtWidgets.QVBoxLayout(self.sliders_widget)
        self.sliders_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.sliders_area.setWidget(self.sliders_widget)
        self.control_layout.addWidget(self.sliders_area)

    def _setup_center_sim_widget(self) -> None:
        self.sim_widget = MuJoCoSimWidget(width=600, height=600)
        self.main_layout.addWidget(self.sim_widget, 2)

    def _setup_right_contact_panel(self) -> None:
        self.contact_panel = QtWidgets.QWidget()
        self.contact_panel.setFixedWidth(250)
        self.contact_layout = QtWidgets.QVBoxLayout(self.contact_panel)

        self.contact_layout.addWidget(QtWidgets.QLabel("<b>Contact Analysis</b>"))

        self.metrics_widget = ContactMetricsWidget()
        self.contact_layout.addWidget(self.metrics_widget)

        self.contact_layout.addSpacing(10)
        self.contact_layout.addWidget(QtWidgets.QLabel("<b>Pressure Distribution</b>"))

        self.pressure_widget = PressureVisualizationWidget()
        self.pressure_widget.setMinimumHeight(200)
        self.contact_layout.addWidget(self.pressure_widget)

        self.btn_export_contacts = QtWidgets.QPushButton("Export Contact Data")
        self.btn_export_contacts.clicked.connect(self._export_contact_data)
        self.contact_layout.addWidget(self.btn_export_contacts)

        self.contact_layout.addStretch()
        self.main_layout.addWidget(self.contact_panel)

    def _init_internal_state(self) -> None:
        self.joint_sliders: list[QtWidgets.QSlider] = []
        self.joint_spinboxes: list[QtWidgets.QDoubleSpinBox] = []
        self.synergy_sliders: list[QtWidgets.QSlider] = []
        self.joint_controls: dict[
            int,
            tuple[
                QtWidgets.QSlider,
                QtWidgets.QDoubleSpinBox,
                float,
                float,
            ],
        ] = {}

        self.grip_contact_model = GripContactModel(GripParameters())
        self.contact_exporter = GripContactExporter(self.grip_contact_model)
        self.contact_timer: QtCore.QTimer | None = None

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
            xml_content = prepare_scene_xml(scene_path, folder_path, is_both)
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
        self.joint_controls.clear()

        if self.sim_widget.model is None or self.sim_widget.data is None:
            return

        # Iterate joints
        model = self.sim_widget.model

        for i in range(model.njnt):
            self._add_joint_control_row(i, model)

        self.rebuild_synergy_controls()

    def _add_joint_control_row(self, i: int, model: mujoco.MjModel) -> None:  # noqa: PLR0915
        """Create a control row for a single joint."""
        if i is None:
            raise ValueError("i must be provided")
        if self.sim_widget.data is None:
            return

        # Skip free joints and ball joints (multi-dof)
        jnt_type = model.jnt_type[i]
        if jnt_type in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            return

        if self.sim_widget.data is None:
            return

        raw_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not raw_name:
            name = f"Joint {i}"
        else:
            name = get_descriptive_joint_name(raw_name)

        # Create UI row
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        label = QtWidgets.QLabel(name)
        label.setFixedWidth(180)
        row_layout.addWidget(label)

        # Range
        range_min, range_max = self._get_joint_range(i, model)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, 1000)
        slider.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(range_min, range_max)
        spin.setSingleStep(0.01)

        # Initial value (qpos) - Assuming qpos address matches joint id for 1-dof joints
        # Need strict qpos address.
        qpos_adr = model.jnt_qposadr[i]
        init_val = self.sim_widget.get_state().get("q", [])[qpos_adr]

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
        self.joint_controls[qpos_adr] = (slider, spin, range_min, range_max)

    def _val_to_slider(self, val: float, min_v: float, max_v: float) -> int:
        """Convert float value to slider integer position."""
        if val is None:
            raise ValueError("val must be provided")
        ratio = (val - min_v) / (max_v - min_v) if max_v > min_v else 0.5
        return int(ratio * 1000)

    def _slider_to_val(self, slider_val: int, min_v: float, max_v: float) -> float:
        """Convert slider integer position to float value."""
        if slider_val is None:
            raise ValueError("slider_val must be provided")
        ratio = slider_val / 1000.0
        return min_v + ratio * (max_v - min_v)

    def _update_joint(self, q_idx: int, val: float) -> None:
        """Update joint value in simulation."""
        if q_idx is None:
            raise ValueError("q_idx must be provided")
        if self.sim_widget.model is None or self.sim_widget.data is None:
            return
        state = self.sim_widget.get_state()
        state["q"][q_idx] = val
        self.sim_widget.set_state_and_forward(state)
        mujoco.mj_forward(self.sim_widget.model, self.sim_widget.data)
        self.sim_widget.render()

    def _update_joints(self, updates: dict[int, float]) -> None:
        """Update multiple joints in simulation and UI in a single pass.

        Args:
            updates: Dict mapping qpos_adr to target float value.
        """
        assert self.sim_widget.model is not None, "Model must be loaded"
        assert self.sim_widget.data is not None, "Data must be loaded"

        state = self.sim_widget.get_state()
        qpos = state.get("q", [])
        for q_idx, val in updates.items():
            if 0 <= q_idx < len(qpos):
                qpos[q_idx] = val

        self.sim_widget.set_state_and_forward(state)
        mujoco.mj_forward(self.sim_widget.model, self.sim_widget.data)
        self.sim_widget.render()

        for q_idx, val in updates.items():
            if q_idx in self.joint_controls:
                slider, spin, min_v, max_v = self.joint_controls[q_idx]
                slider.blockSignals(True)
                spin.blockSignals(True)
                slider.setValue(self._val_to_slider(val, min_v, max_v))
                spin.setValue(val)
                slider.blockSignals(False)
                spin.blockSignals(False)

    def _find_qpos_adr_by_name(self, name: str) -> int | None:
        """Find the qpos address of a joint by its name."""
        if self.sim_widget.model is None:
            return None
        model = self.sim_widget.model
        try:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id != -1:
                return int(model.jnt_qposadr[jnt_id])
        except (RuntimeError, ValueError):
            pass
        return None

    def add_synergy_slider(self, synergy: Synergy) -> None:
        """Add a synergy slider to the UI.

        Args:
            synergy: The Synergy mapping configuration.
        """
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        lbl = QtWidgets.QLabel(synergy.name)
        lbl.setFixedWidth(180)
        row_layout.addWidget(lbl)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, 1000)
        slider.setValue(0)
        slider.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        def _on_synergy_changed(v: int, syn: Synergy = synergy) -> None:
            t = v / 1000.0
            updates = {}
            for binding in syn.bindings:
                val = binding.min_val + t * (binding.max_val - binding.min_val)
                updates[binding.qpos_adr] = val
            self._update_joints(updates)

        slider.valueChanged.connect(_on_synergy_changed)
        row_layout.addWidget(slider)

        self.synergy_layout.addWidget(row)
        self.synergy_sliders.append(slider)

    def rebuild_synergy_controls(self) -> None:
        """Rebuild the synergy controls based on the loaded hand model."""
        while self.synergy_layout.count():
            item = self.synergy_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget:
                    widget.deleteLater()

        self.synergy_sliders.clear()

        if self.sim_widget.model is None:
            return

        model_name = self.combo_hand.currentText().lower()
        is_shadow = "shadow" in model_name
        is_allegro = "allegro" in model_name

        defaults: list[Synergy] = []
        prefixes: list[str] = []

        if "both" in model_name:
            prefixes = ["rh", "lh"]
        elif "right" in model_name:
            prefixes = ["rh" if is_shadow else "right"]
        elif "left" in model_name:
            prefixes = ["lh" if is_shadow else "left"]

        if is_shadow:
            fist_bindings = []
            for p in prefixes:
                for f in ["FF", "MF", "RF", "LF"]:
                    for j in [3, 2, 1]:
                        jnt_name = f"{p}_{f}J{j}"
                        q_adr = self._find_qpos_adr_by_name(jnt_name)
                        if q_adr is not None:
                            fist_bindings.append(SynergyJointBinding(q_adr, 0.0, 1.4))
            if fist_bindings:
                defaults.append(Synergy("Fist Curl", fist_bindings))

            index_bindings = []
            for p in prefixes:
                for j in [3, 2, 1]:
                    jnt_name = f"{p}_FFJ{j}"
                    q_adr = self._find_qpos_adr_by_name(jnt_name)
                    if q_adr is not None:
                        index_bindings.append(SynergyJointBinding(q_adr, 0.0, 1.4))
            if index_bindings:
                defaults.append(Synergy("Index Curl", index_bindings))

            pinch_bindings = []
            for p in prefixes:
                for j in [3, 2, 1]:
                    q_adr = self._find_qpos_adr_by_name(f"{p}_FFJ{j}")
                    if q_adr is not None:
                        pinch_bindings.append(SynergyJointBinding(q_adr, 0.0, 1.0))
                for j in [4, 3, 2, 1]:
                    q_adr = self._find_qpos_adr_by_name(f"{p}_THJ{j}")
                    if q_adr is not None:
                        pinch_bindings.append(SynergyJointBinding(q_adr, 0.0, 0.8))
            if pinch_bindings:
                defaults.append(Synergy("Pinch Grip", pinch_bindings))

        elif is_allegro:
            all_joints = []
            model = self.sim_widget.model
            for i in range(model.njnt):
                jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if jnt_name:
                    all_joints.append(jnt_name)

            fist_bindings = []
            for jnt_name in all_joints:
                target_joints = [
                    "ffj1",
                    "ffj2",
                    "ffj3",
                    "mfj1",
                    "mfj2",
                    "mfj3",
                    "rfj1",
                    "rfj2",
                    "rfj3",
                ]
                if any(x in jnt_name.lower() for x in target_joints):
                    q_adr = self._find_qpos_adr_by_name(jnt_name)
                    if q_adr is not None:
                        fist_bindings.append(SynergyJointBinding(q_adr, 0.0, 1.5))
            if fist_bindings:
                defaults.append(Synergy("Fist Curl", fist_bindings))

            index_bindings = []
            for jnt_name in all_joints:
                if any(x in jnt_name.lower() for x in ["ffj1", "ffj2", "ffj3"]):
                    q_adr = self._find_qpos_adr_by_name(jnt_name)
                    if q_adr is not None:
                        index_bindings.append(SynergyJointBinding(q_adr, 0.0, 1.5))
            if index_bindings:
                defaults.append(Synergy("Index Curl", index_bindings))

            pinch_bindings = []
            for jnt_name in all_joints:
                if any(x in jnt_name.lower() for x in ["ffj1", "ffj2", "ffj3"]):
                    q_adr = self._find_qpos_adr_by_name(jnt_name)
                    if q_adr is not None:
                        pinch_bindings.append(SynergyJointBinding(q_adr, 0.0, 1.0))
                if any(x in jnt_name.lower() for x in ["thj1", "thj2", "thj3"]):
                    q_adr = self._find_qpos_adr_by_name(jnt_name)
                    if q_adr is not None:
                        pinch_bindings.append(SynergyJointBinding(q_adr, 0.0, 1.0))
            if pinch_bindings:
                defaults.append(Synergy("Pinch Grip", pinch_bindings))

        for syn in defaults:
            self.add_synergy_slider(syn)

    def _on_add_custom_synergy(self) -> None:
        """Open the dialog to define a custom synergy slider."""
        if self.sim_widget.model is None:
            return

        model = self.sim_widget.model
        joints = []
        for i in range(model.njnt):
            jnt_type = model.jnt_type[i]
            if jnt_type in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
                continue

            raw_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if not raw_name:
                raw_name = f"Joint {i}"

            desc_name = get_descriptive_joint_name(raw_name)
            qpos_adr = model.jnt_qposadr[i]
            min_l, max_l = self._get_joint_range(i, model)

            joints.append((qpos_adr, desc_name, min_l, max_l))

        dialog = AddSynergyDialog(joints, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            synergy = dialog.get_synergy()
            if synergy:
                self.add_synergy_slider(synergy)

    def _on_slider(  # noqa: PLR0913
        self,
        val_int: int,
        spin: QtWidgets.QDoubleSpinBox,
        min_v: float,
        max_v: float,
        q_idx: int,
    ) -> None:
        """Handle slider value change."""
        if val_int is None:
            raise ValueError("val_int must be provided")
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
        if val is None:
            raise ValueError("val must be provided")
        slider_val = self._val_to_slider(val, min_v, max_v)
        slider.blockSignals(True)  # noqa: FBT003
        slider.setValue(slider_val)
        slider.blockSignals(False)  # noqa: FBT003
        self._update_joint(q_idx, val)

    def _get_joint_range(self, i: int, model: mujoco.MjModel) -> tuple[float, float]:
        """Get valid joint range, providing defaults if undefined."""
        if i is None:
            raise ValueError("i must be provided")
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

    def _extract_hand_contacts(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> tuple[list, list, list, list, list]:
        if model is None:
            raise ValueError("model must be provided")
        positions = []
        normals = []
        forces = []
        velocities = []
        body_names = []

        for i in range(data.ncon):
            contact = data.contact[i]
            pos = contact.pos.copy()
            normal = contact.frame[:3].copy()

            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            contact_force = force[:3]

            vel = np.zeros(3)

            geom1 = contact.geom1
            geom2 = contact.geom2
            body1_id = model.geom_bodyid[geom1]
            body2_id = model.geom_bodyid[geom2]
            body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

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

        return positions, normals, forces, velocities, body_names

    def _update_contact_visualizations(
        self, positions_arr: np.ndarray, state: Any
    ) -> None:
        if positions_arr is None:
            raise ValueError("positions_arr must be provided")
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

        margins = self.grip_contact_model.check_slip_margin()
        equilibrium = self.grip_contact_model.check_static_equilibrium(3.0)

        self.metrics_widget.update_metrics(
            normal_force=state.total_normal_force,
            tangent_force=float(np.linalg.norm(state.total_tangent_force)),
            num_contacts=len(state.contacts),
            num_slipping=state.num_slipping,
            slip_margin=margins["min_margin"],
            equilibrium=bool(equilibrium.get("equilibrium", False)),
        )

    def _update_contact_data(self) -> None:
        """Update contact data from MuJoCo simulation.

        Extracts contact information from MuJoCo and updates visualizations.
        """
        if self.sim_widget.model is None or self.sim_widget.data is None:
            return

        model = self.sim_widget.model
        data = self.sim_widget.data

        if data.ncon == 0:
            self.pressure_widget.clear()
            self.metrics_widget.update_metrics(0, 0, 0, 0, 0.0, False)
            return

        positions, normals, forces, velocities, body_names = (
            self._extract_hand_contacts(model, data)
        )  # noqa: E501

        if not positions:
            self.pressure_widget.clear()
            self.metrics_widget.update_metrics(0, 0, 0, 0, 0.0, False)
            return

        positions_arr = np.array(positions)
        normals_arr = np.array(normals)
        forces_arr = np.array(forces)
        velocities_arr = np.array(velocities)

        state = self.grip_contact_model.update_from_mujoco(
            positions_arr,
            normals_arr,
            forces_arr,
            velocities_arr,
            body_names,
            data.time,
        )

        self.contact_exporter.capture_timestep()
        self._update_contact_visualizations(positions_arr, state)

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

                data = self.contact_exporter.export_to_dict()  # type: ignore[assignment]
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
            )  # noqa: E501


__all__ = [
    "ContactMetricsWidget",
    "GripModellingTab",
    "PressureVisualizationWidget",
]
