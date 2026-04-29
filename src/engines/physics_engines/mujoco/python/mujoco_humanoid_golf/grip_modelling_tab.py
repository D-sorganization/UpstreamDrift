"""Grip Modelling Tab for Advanced Hand Models.

Issue #757: Contact-based hand-grip model in MuJoCo with pressure visualization.

Sub-components (extracted in issue #3060):
  - grip_plot_panel.py  — PressureVisualizationWidget, ContactMetricsWidget
  - grip_xml_builder.py — GripSceneXmlBuilder (scene XML preparation)
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

from .grip_plot_panel import ContactMetricsWidget, PressureVisualizationWidget
from .grip_xml_builder import GripSceneXmlBuilder
from .sim_widget import MuJoCoSimWidget

logger = get_logger(__name__)

# Backward-compat re-exports so existing importers keep working without changes.
__all__ = [
    "GripModellingTab",
    "PressureVisualizationWidget",
    "ContactMetricsWidget",
    "GripSceneXmlBuilder",
]


class GripModellingTab(QtWidgets.QWidget):
    """Tab for manipulating advanced hand models (Shadow, Allegro)."""

    def connect_sim_widget(self, sim_widget: MuJoCoSimWidget) -> None:
        """Connect to an external simulation widget.

        Args:
           sim_widget: The main simulation widget to connect to.
        """
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
        self.control_panel.setFixedWidth(300)
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
        )
        self.chk_contact_monitor.setChecked(False)
        self.chk_contact_monitor.toggled.connect(self._on_contact_monitor_toggled)
        self.control_layout.addWidget(self.chk_contact_monitor)

        self.control_layout.addSpacing(10)
        self.control_layout.addWidget(QtWidgets.QLabel("<b>Joint Controls</b>"))

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

        self.grip_contact_model = GripContactModel(GripParameters())
        self.contact_exporter = GripContactExporter(self.grip_contact_model)
        self.contact_timer: QtCore.QTimer | None = None
        self._xml_builder = GripSceneXmlBuilder()

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

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
            xml_content = self._xml_builder.prepare_scene_xml(
                scene_path, folder_path, is_both
            )
        except (RuntimeError, ValueError, OSError):
            logger.exception("Failed to prepare XML model from %s", scene_path)
            return

        try:
            current_dir = os.getcwd()
            os.chdir(scene_path.parent)
            try:
                self.sim_widget.load_model_from_xml(xml_content)
            finally:
                os.chdir(current_dir)
        except (RuntimeError, ValueError, OSError):
            logger.exception("Failed to load XML model")
            return

        self.rebuild_joint_controls()
        self._on_kinematic_toggled(self.chk_kinematic.isChecked())

    # -------------------------------------------------------------------------
    # Joint controls
    # -------------------------------------------------------------------------

    def rebuild_joint_controls(self) -> None:
        """Rebuild the joint control widgets for the current model."""
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

        model = self.sim_widget.model
        for i in range(model.njnt):
            self._add_joint_control_row(i, model)

    def _add_joint_control_row(
        self, i: int, model: mujoco.MjModel
    ) -> None:  # noqa: PLR0915
        """Create a control row for a single joint."""
        if i is None:
            raise ValueError("i must be provided")
        if self.sim_widget.data is None:
            return

        jnt_type = model.jnt_type[i]
        if jnt_type in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            return

        if self.sim_widget.data is None:
            return

        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not name:
            name = f"Joint {i}"

        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        label = QtWidgets.QLabel(name)
        label.setFixedWidth(120)
        row_layout.addWidget(label)

        range_min, range_max = self._get_joint_range(i, model)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, 1000)

        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(range_min, range_max)
        spin.setSingleStep(0.01)

        qpos_adr = model.jnt_qposadr[i]
        init_val = self.sim_widget.get_state()[0][qpos_adr]

        slider.setValue(self._val_to_slider(init_val, range_min, range_max))
        spin.setValue(init_val)

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
        state[0][q_idx] = val
        self.sim_widget.set_state_and_forward(*state)
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
        )

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
