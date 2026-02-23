"""Simulation and model management mixin for Pinocchio GUI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pinocchio as pin
from PyQt6 import QtCore, QtWidgets

from src.shared.python.data_io.common_utils import get_shared_urdf_path
from src.shared.python.ui.widgets import SignalBlocker

from .induced_acceleration import InducedAccelerationAnalyzer
from .manipulability import PinocchioManipulabilityAnalyzer

if TYPE_CHECKING:
    from .main_window import PinocchioGUI

logger = logging.getLogger(__name__)

# Constants (mirrored)
SLIDER_RANGE_RAD = 10.0
SLIDER_SCALE = 100.0


class SimulationMixin:
    """Mixin containing simulation control, model loading, and kinematic management."""

    def _scan_urdf_models(self: PinocchioGUI) -> None:
        """Scan shared/urdf for models."""
        try:
            urdf_dir = get_shared_urdf_path()
            if urdf_dir is not None and urdf_dir.exists():
                for urdf_file in urdf_dir.glob("*.urdf"):
                    name = urdf_file.stem.replace("_", " ").title()
                    self.available_models.append(
                        {"name": f"URDF: {name}", "path": str(urdf_file)}
                    )
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Failed to scan URDF models: {e}")

    def load_urdf(self: PinocchioGUI, fname: str | None = None) -> None:
        """Load a URDF model and initialize the viewer."""
        from . import MESHCAT_AVAILABLE, MeshcatVisualizer

        if not fname:
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select URDF File", "", "URDF Files (*.urdf *.xml)"
            )

        if not fname:
            return

        try:
            # Build models
            self.model = pin.buildModelFromUrdf(fname)
            if self.model is None:
                self.log_write("Error loading URDF: Failed to build model")
                return

            try:
                self.visual_model = pin.buildGeomFromUrdf(
                    self.model, fname, pin.GeometryType.VISUAL
                )
                self.collision_model = pin.buildGeomFromUrdf(
                    self.model, fname, pin.GeometryType.COLLISION
                )
            except (RuntimeError, ValueError, OSError) as e:
                self.log_write(f"Warning: Failed to load geometries: {e}")
                self.visual_model = None
                self.collision_model = None

            self.data = self.model.createData()
            self.q = pin.neutral(self.model)
            self.v = np.zeros(self.model.nv)
            self.sim_time = 0.0

            # Init Analyzer
            self.analyzer = InducedAccelerationAnalyzer(self.model, self.data)

            # Init Manipulability Analyzer
            self.manip_analyzer = PinocchioManipulabilityAnalyzer(self.model, self.data)
            self._populate_manipulability_checkboxes()

            # Reset recorder
            self.recorder.reset()
            self.lbl_rec_status.setText("Frames: 0")
            if self.btn_record.isChecked():
                self.btn_record.setChecked(False)
                self.btn_record.setText("Record")

            # Initialize Pinocchio MeshcatVisualizer
            if MESHCAT_AVAILABLE and self.viewer is not None:
                try:
                    self.viewer["robot"].delete()
                    self.viewer["overlays"].delete()

                    self.viz = MeshcatVisualizer(
                        self.model, self.collision_model, self.visual_model
                    )
                    self.viz.initViewer(viewer=self.viewer, open=False)
                    self.viz.loadViewerModel()
                except (RuntimeError, ValueError, OSError) as e:
                    self.log_write(f"Warning: Visualizer init failed: {e}")
                    self.viz = None
            else:
                self.log_write("Model loaded without 3D visualization.")
                self.viz = None

            self.log_write(f"Successfully loaded URDF: {fname}")
            self.log_write(f"NQ: {self.model.nq}, NV: {self.model.nv}")

            # Rebuild Kinematic Controls
            self._build_kinematic_controls()
            self._sync_kinematic_controls()

            # Init state display
            self._update_viewer()

            # Restore overlays for new model if checkboxes are active
            if self.chk_frames.isChecked():
                self._toggle_frames(checked=True)
            if self.chk_coms.isChecked():
                self._toggle_coms(checked=True)

            if not self.timer.isActive():
                self.timer.start(int(getattr(self, "dt", 0.01) * 1000))

            if hasattr(self, "live_plot"):
                self.live_plot.set_joint_names(self.get_joint_names())

        except (ValueError, RuntimeError) as e:
            self.log_write(f"Error loading URDF (Pinocchio): {e}")
        except (PermissionError, OSError) as e:
            self.log_write(f"Unexpected error loading URDF: {e}")
            logger.exception("Unexpected error loading URDF")

    def _build_kinematic_controls(self: PinocchioGUI) -> None:
        """Create sliders for all joints in the kinematic tab."""
        if self.model is None:
            return

        # Clear layout
        while self.slider_layout.count():
            item = self.slider_layout.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.joint_sliders = []
        self.joint_spinboxes = []
        self.joint_names = list(self.model.names)[1:]

        # Update joint selection combo for analysis
        if hasattr(self, "joint_select_combo"):
            self.joint_select_combo.clear()
            self.joint_select_combo.addItems(self.joint_names)

        # Iterate joints (skip universe)
        for i in range(1, self.model.njoints):
            self._add_joint_control_widget(i)

    def _add_joint_control_widget(self: PinocchioGUI, i: int) -> None:
        """Add a single joint control row."""
        if self.model is None:
            return

        joint_name = self.model.names[i]
        nq_joint = self.model.joints[i].nq

        if nq_joint != 1:
            msg = (
                f"Skipping joint '{joint_name}' (index {i}): "
                f"{nq_joint} DOFs not supported in kinematic controls."
            )
            self.log_write(msg)
            return

        row = QtWidgets.QWidget()
        r_layout = QtWidgets.QHBoxLayout(row)
        r_layout.setContentsMargins(0, 0, 0, 0)
        r_layout.addWidget(QtWidgets.QLabel(f"{joint_name}:"))

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider_min = int(-SLIDER_RANGE_RAD * SLIDER_SCALE)
        slider_max = int(SLIDER_RANGE_RAD * SLIDER_SCALE)
        slider.setRange(slider_min, slider_max)
        slider.setValue(0)

        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-SLIDER_RANGE_RAD, SLIDER_RANGE_RAD)
        spin.setSingleStep(0.1)

        idx_q = self.model.joints[i].idx_q
        idx = int(idx_q)

        slider.valueChanged.connect(
            lambda val, s=spin, k=idx: self._on_slider(val, s, k)
        )
        spin.valueChanged.connect(lambda val, s=slider, k=idx: self._on_spin(val, s, k))

        r_layout.addWidget(slider)
        r_layout.addWidget(spin)
        self.slider_layout.addWidget(row)

        self.joint_sliders.append(slider)
        self.joint_spinboxes.append(spin)

    def _sync_kinematic_controls(self: PinocchioGUI) -> None:
        """Synchronize sliders/spinboxes with current model state q."""
        if self.model is None or self.q is None:
            return

        slider_idx = 0
        for i in range(1, self.model.njoints):
            if self.model.joints[i].nq != 1:
                continue

            idx_q = self.model.joints[i].idx_q
            val = self.q[idx_q]

            if slider_idx < len(self.joint_sliders):
                slider = self.joint_sliders[slider_idx]
                spin = self.joint_spinboxes[slider_idx]

                with SignalBlocker(slider, spin):
                    slider.setValue(int(val * SLIDER_SCALE))
                    spin.setValue(val)

                slider_idx += 1

    def _on_slider(
        self: PinocchioGUI, val: int, spin: QtWidgets.QDoubleSpinBox, idx: int
    ) -> None:
        angle = val / SLIDER_SCALE
        with SignalBlocker(spin):
            spin.setValue(angle)
        self._update_q(idx, angle)

    def _on_spin(
        self: PinocchioGUI, val: float, slider: QtWidgets.QSlider, idx: int
    ) -> None:
        with SignalBlocker(slider):
            slider.setValue(int(val * SLIDER_SCALE))
        self._update_q(idx, val)

    def _update_q(self: PinocchioGUI, idx: int, val: float) -> None:
        if self.operating_mode != "kinematic":
            return
        if self.q is not None:
            self.q[idx] = val
            self._update_viewer()

    def _on_mode_changed(self: PinocchioGUI, mode_text: str) -> None:
        """Handle operating mode change."""
        if "Dynamic" in mode_text:
            self.operating_mode = "dynamic"
            self.controls_stack.setCurrentIndex(0)
        else:
            self.operating_mode = "kinematic"
            self.controls_stack.setCurrentIndex(1)
            self.is_running = False
            self.btn_run.setText("Run Simulation")
            self.btn_run.setChecked(False)
            self._sync_kinematic_controls()

    def step_simulation(self: PinocchioGUI) -> None:
        """Advance the physics simulation by one time step."""
        if self.model is None or self.data is None or self.q is None or self.v is None:
            return

        tau = np.zeros(self.model.nv)
        a = pin.aba(self.model, self.data, self.q, self.v, tau)
        self.v += a * self.dt
        self.q = pin.integrate(self.model, self.q, self.v * self.dt)
        self.sim_time += self.dt

        if self.recorder.is_recording:
            self._record_frame()

    def _record_frame(self: PinocchioGUI) -> None:
        """Capture and record a single frame of simulation state."""
        assert self.model is not None
        assert self.data is not None
        assert self.q is not None
        assert self.v is not None
        tau = np.zeros(self.model.nv)

        pin.computeKineticEnergy(self.model, self.data, self.q, self.v)
        pin.computePotentialEnergy(self.model, self.data, self.q)

        club_head_pos, club_head_vel = self._find_club_head_state()
        induced, counterfactuals = self._compute_live_analysis(tau)

        self.recorder.record_frame(
            time=self.sim_time,
            q=self.q.copy(),
            v=self.v.copy(),
            tau=tau,
            kinetic_energy=self.data.kinetic_energy,
            potential_energy=self.data.potential_energy,
            club_head_position=club_head_pos,
            club_head_velocity=club_head_vel,
            induced_accelerations=induced,
            counterfactuals=counterfactuals,
        )
        self.lbl_rec_status.setText(f"Frames: {self.recorder.get_num_frames()}")

    def _find_club_head_state(
        self: PinocchioGUI,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Resolve club head frame and return its pose/velocity."""
        assert self.model is not None
        assert self.data is not None

        club_id = -1
        for fid in range(self.model.nframes):
            name = self.model.frames[fid].name.lower()
            if "club" in name or "head" in name:
                club_id = fid
                break

        if club_id < 0:
            return None, None

        frame = self.data.oMf[club_id]
        pos = frame.translation.copy()
        v_frame = pin.getFrameVelocity(
            self.model, self.data, club_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        vel = v_frame.linear.copy()
        return pos, vel

    def _compute_live_analysis(
        self: PinocchioGUI, tau: np.ndarray
    ) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
        """Run real-time induced/counterfactual analysis if enabled."""
        if not self.chk_live_analysis.isChecked():
            return None, None

        if self.analyzer and self.q is not None and self.v is not None:
            induced = self.analyzer.compute_components(self.q, self.v, tau)
            self.latest_induced = induced

            # Simplified analyzer call for mixin
            if hasattr(self.analyzer, "compute_counterfactuals"):
                cf = self.analyzer.compute_counterfactuals(self.q, self.v)
                self.latest_cf = cf
                return induced, cf
            return induced, None
        return None, None

    def _populate_manipulability_checkboxes(self: PinocchioGUI) -> None:
        """Populate manipulability target selection grid."""
        if self.manip_analyzer is None:
            return

        while self.manip_body_layout.count():
            item = self.manip_body_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        self.manip_checkboxes.clear()
        bodies = self.manip_analyzer.find_potential_bodies()

        cols = 3
        for i, name in enumerate(bodies):
            chk = QtWidgets.QCheckBox(name)
            chk.toggled.connect(self._update_viewer)
            self.manip_checkboxes[name] = chk
            self.manip_body_layout.addWidget(chk, i // cols, i % cols)
            if any(x in name.lower() for x in ["club", "hand", "wrist"]):
                chk.setChecked(True)
