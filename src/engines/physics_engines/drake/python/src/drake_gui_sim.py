"""Drake GUI Simulation Control Mixin.

Contains simulation loop, recording, mode switching, and joint control methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.engine_core.engine_availability import (
    PYQT6_AVAILABLE,
)
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.theme.style_constants import Styles

HAS_QT = PYQT6_AVAILABLE

if HAS_QT:
    from PyQt6 import QtCore, QtWidgets

# Drake imports
if TYPE_CHECKING or HAS_QT:
    try:
        from pydrake.all import (
            BodyIndex,
            JointIndex,
            PrismaticJoint,
            RevoluteJoint,
        )
    except ImportError:
        BodyIndex = None  # type: ignore[misc, assignment]
        JointIndex = None  # type: ignore[misc, assignment]
        PrismaticJoint = None  # type: ignore[misc, assignment]
        RevoluteJoint = None  # type: ignore[misc, assignment]

# Constants
TIME_STEP_S = 0.001
MS_PER_SECOND = 1000
SLIDER_TO_RADIAN = 0.01
STYLE_BUTTON_RUN = Styles.BTN_RUN
STYLE_BUTTON_STOP = Styles.BTN_STOP

LOGGER = get_logger(__name__)


class SimulationMixin:
    """Mixin providing simulation control methods for DrakeSimApp."""

    def _on_mode_changed(self, text: str) -> None:
        if "Kinematic" in text:
            self.operating_mode = "kinematic"  # type: ignore[attr-defined]
            self.controls_stack.setCurrentIndex(1)  # type: ignore[attr-defined]
            self.is_running = False  # type: ignore[attr-defined]
            self.btn_run.setChecked(False)  # type: ignore[attr-defined]
            self.btn_run.setStyleSheet(STYLE_BUTTON_RUN)  # type: ignore[attr-defined]
            self.btn_run.setText("▶ Run Simulation")  # type: ignore[attr-defined]
            self._update_status("Mode: Kinematic Control")  # type: ignore[attr-defined]
            self._sync_kinematic_sliders()
            # Stop physics, allow manual
        else:
            self.operating_mode = "dynamic"  # type: ignore[attr-defined]
            self.controls_stack.setCurrentIndex(0)  # type: ignore[attr-defined]
            self._update_status("Mode: Dynamic Simulation")  # type: ignore[attr-defined]
            # Ensure simulation resumes or is stopped
            if self.is_running:  # type: ignore[attr-defined]
                self.btn_run.setText("■ Stop Simulation")  # type: ignore[attr-defined]
                self.btn_run.setChecked(True)  # type: ignore[attr-defined]
                self.btn_run.setStyleSheet(STYLE_BUTTON_STOP)  # type: ignore[attr-defined]
            else:
                self.btn_run.setText("▶ Run Simulation")  # type: ignore[attr-defined]
                self.btn_run.setChecked(False)  # type: ignore[attr-defined]
                self.btn_run.setStyleSheet(STYLE_BUTTON_RUN)  # type: ignore[attr-defined]

    def _toggle_run(self, checked: bool) -> None:  # noqa: FBT001
        self.is_running = checked  # type: ignore[attr-defined]
        if checked:
            self.btn_run.setText("■ Stop Simulation")  # type: ignore[attr-defined]
            self.btn_run.setStyleSheet(STYLE_BUTTON_STOP)  # type: ignore[attr-defined]
            self._update_status("Simulation Running...")  # type: ignore[attr-defined]
        else:
            self.btn_run.setText("▶ Run Simulation")  # type: ignore[attr-defined]
            self.btn_run.setStyleSheet(STYLE_BUTTON_RUN)  # type: ignore[attr-defined]
            self._update_status("Simulation Stopped.")  # type: ignore[attr-defined]

    def _reset_simulation(self) -> None:
        self.is_running = False  # type: ignore[attr-defined]
        self.btn_run.setChecked(False)  # type: ignore[attr-defined]
        self.btn_run.setText("▶ Run Simulation")  # type: ignore[attr-defined]
        self.btn_run.setStyleSheet(STYLE_BUTTON_RUN)  # type: ignore[attr-defined]
        self._update_status("Simulation Reset.")  # type: ignore[attr-defined]
        self._reset_state()  # type: ignore[attr-defined]

    def _game_loop(self) -> None:
        simulator = self.simulator  # type: ignore[attr-defined]
        context = self.context  # type: ignore[attr-defined]

        if not simulator or not context:
            return

        # Always update Live Plot (even if paused, to redraw last frame/resize)
        if hasattr(self, "live_plot"):
            self.live_plot.update_plot()

        if self.operating_mode == "dynamic" and self.is_running:  # type: ignore[attr-defined]
            self._advance_physics(simulator, context)

        # Visualization Update
        self._update_visualization()  # type: ignore[attr-defined]

    def _advance_physics(self, simulator: Any, context: Any) -> None:
        """Advance the physics simulation by one time step and record data."""
        t = context.get_time()
        simulator.AdvanceTo(t + self.time_step)  # type: ignore[attr-defined]

        if self.recorder.is_recording and self.plant:  # type: ignore[attr-defined]
            self._record_frame(context)

    def _record_frame(self, context: Any) -> None:
        """Record a single frame of simulation data."""
        assert self.plant is not None  # type: ignore[attr-defined]  # guaranteed by caller check
        plant_context = self.plant.GetMyContextFromRoot(context)  # type: ignore[attr-defined]
        q = self.plant.GetPositions(plant_context)  # type: ignore[attr-defined]
        v = self.plant.GetVelocities(plant_context)  # type: ignore[attr-defined]

        club_pos = self._find_club_head_position(plant_context)

        # Live Analysis if enabled OR if LivePlotWidget requested it via config
        if self._is_analysis_enabled() and self.eval_context:  # type: ignore[attr-defined]
            self._compute_live_analysis(q, v)  # type: ignore[attr-defined]

        # Calculate CoM and Angular Momentum for recording
        com_pos = None
        angular_momentum = None
        if self.plant:  # type: ignore[attr-defined]
            plant_context = self.plant.GetMyContextFromRoot(context)  # type: ignore[attr-defined]
            com_pos = self.plant.CalcCenterOfMassPositionInWorld(plant_context)  # type: ignore[attr-defined]
            angular_momentum = self.plant.CalcSpatialMomentumInWorldAboutPoint(  # type: ignore[attr-defined]
                plant_context, com_pos
            ).rotational()

        self.recorder.record(  # type: ignore[attr-defined]
            context.get_time(),
            q,
            v,
            club_pos,
            com_pos=com_pos,
            angular_momentum=angular_momentum,
        )
        self.lbl_rec_status.setText(f"Frames: {len(self.recorder.times)}")  # type: ignore[attr-defined]

    def _find_club_head_position(self, plant_context: Any) -> np.ndarray:
        """Find and return the club head position in world coordinates."""
        assert self.plant is not None  # type: ignore[attr-defined]  # guaranteed by caller
        body_names = ["clubhead", "club_body", "wrist", "hand", "link_7"]
        for name in body_names:
            if self.plant.HasBodyNamed(name):  # type: ignore[attr-defined]
                body = self.plant.GetBodyByName(name)  # type: ignore[attr-defined]
                X_WB = self.plant.EvalBodyPoseInWorld(plant_context, body)  # type: ignore[attr-defined]
                return X_WB.translation()

        # Fallback to last body
        body = self.plant.get_body(BodyIndex(self.plant.num_bodies() - 1))  # type: ignore[attr-defined]
        X_WB = self.plant.EvalBodyPoseInWorld(plant_context, body)  # type: ignore[attr-defined]
        return X_WB.translation()

    def _on_slider_change(  # type: ignore[no-any-unimported]
        self, val: int, spin: QtWidgets.QDoubleSpinBox, joint_idx: int
    ) -> None:
        radian = val * SLIDER_TO_RADIAN
        with QtCore.QSignalBlocker(spin):
            spin.setValue(radian)
        self._update_joint_pos(joint_idx, radian)

    def _on_spin_change(  # type: ignore[no-any-unimported]
        self, val: float, slider: QtWidgets.QSlider, joint_idx: int
    ) -> None:
        with QtCore.QSignalBlocker(slider):
            slider.setValue(int(val / SLIDER_TO_RADIAN))
        self._update_joint_pos(joint_idx, val)

    def _update_joint_pos(self, joint_idx: int, angle: float) -> None:
        """Update joint position in plant context."""
        if self.operating_mode != "kinematic":  # type: ignore[attr-defined]
            return

        plant = self.plant  # type: ignore[attr-defined]
        context = self.context  # type: ignore[attr-defined]
        diagram = self.diagram  # type: ignore[attr-defined]

        if not plant or not context or not diagram:
            return

        plant_context = plant.GetMyContextFromRoot(context)

        joint = plant.get_joint(JointIndex(joint_idx))

        # Assuming single DOF revolute/prismatic for now
        if joint.num_positions() == 1:
            if isinstance(joint, RevoluteJoint):
                joint.set_angle(plant_context, angle)
            elif isinstance(joint, PrismaticJoint):
                joint.set_translation(plant_context, angle)

        diagram.ForcedPublish(context)

        # Update overlays
        if self.visualizer:  # type: ignore[attr-defined]
            self.visualizer.update_frame_transforms(context)  # type: ignore[attr-defined]
            self.visualizer.update_com_transforms(context)  # type: ignore[attr-defined]

        self._update_visualization()  # type: ignore[attr-defined]

    def _sync_kinematic_sliders(self) -> None:
        """Read current plant state and update sliders."""
        plant = self.plant  # type: ignore[attr-defined]
        context = self.context  # type: ignore[attr-defined]
        if not plant or not context:
            return

        plant_context = plant.GetMyContextFromRoot(context)

        for j_idx, spin in self.spinboxes.items():  # type: ignore[attr-defined]
            joint = plant.get_joint(JointIndex(j_idx))
            if joint.num_positions() == 1:
                val = joint.GetOnePosition(plant_context)
                spin.setValue(val)

    def _toggle_recording(self, checked: bool) -> None:  # type: ignore[override]  # noqa: FBT001
        if checked:
            self.recorder.start()  # type: ignore[attr-defined]
            self.btn_record.setText("Stop Recording")  # type: ignore[attr-defined]
            self._update_status("Recording started...")  # type: ignore[attr-defined]
        else:
            self.recorder.stop()  # type: ignore[attr-defined]
            self.btn_record.setText("Record")  # type: ignore[attr-defined]
            self._update_status(  # type: ignore[attr-defined]
                f"Recording stopped. Total Frames: {len(self.recorder.times)}"  # type: ignore[attr-defined]
            )

    def _export_data(self) -> None:
        """Export recorded data to multiple formats."""
        if not self.recorder.times:  # type: ignore[attr-defined]
            QtWidgets.QMessageBox.warning(self, "No Data", "No data to export.")  # type: ignore[arg-type]
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Data", "drake_sim_data", "All Files (*)"  # type: ignore[arg-type]
        )
        if not filename:
            return

        try:
            from shared.python.data_io.export import export_recording_all_formats

            data_dict = self.recorder.export_to_dict()  # type: ignore[attr-defined]
            results = export_recording_all_formats(filename, data_dict)

            msg = "Export Results:\n"
            for fmt, success in results.items():
                msg += f"{fmt}: {'Success' if success else 'Failed'}\n"

            QtWidgets.QMessageBox.information(self, "Export Complete", msg)  # type: ignore[arg-type]
            self._update_status(f"Data exported to {filename}")  # type: ignore[attr-defined]

        except ImportError as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))  # type: ignore[arg-type]
            LOGGER.error(f"Export failed: {e}")
