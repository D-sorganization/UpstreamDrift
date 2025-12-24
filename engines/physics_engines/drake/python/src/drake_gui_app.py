"""Drake Golf GUI Application using PyQt6."""

import logging
import os
import sys
import typing
import webbrowser

import numpy as np
from pydrake.all import (
    BodyIndex,
    Context,
    Diagram,
    JacobianWrtVariable,
    JointIndex,
    Meshcat,
    MeshcatParams,
    MultibodyPlant,
    PrismaticJoint,
    RevoluteJoint,
    RigidTransform,
    Simulator,
)
from PyQt6 import QtCore, QtGui, QtWidgets

from .drake_golf_model import GolfModelParams, build_golf_swing_diagram
from .drake_visualizer import DrakeVisualizer
from .logger_utils import setup_logging

LOGGER = logging.getLogger(__name__)

SLIDER_TO_RADIAN: typing.Final[float] = (
    0.01  # [rad/slider_unit] Conversion factor from slider integer values to radians
)
JOINT_ANGLE_MIN_RAD: typing.Final[float] = (
    -10.0
)  # [rad] Minimum joint angle for UI controls (Safety limit)
JOINT_ANGLE_MAX_RAD: typing.Final[float] = (
    10.0  # [rad] Maximum joint angle for UI controls (Safety limit)
)

SLIDER_RANGE_MIN: typing.Final[int] = int(
    JOINT_ANGLE_MIN_RAD / SLIDER_TO_RADIAN
)  # [slider_units] Computed from JOINT_ANGLE_MIN_RAD / SLIDER_TO_RADIAN
SLIDER_RANGE_MAX: typing.Final[int] = int(
    JOINT_ANGLE_MAX_RAD / SLIDER_TO_RADIAN
)  # [slider_units] Computed from JOINT_ANGLE_MAX_RAD / SLIDER_TO_RADIAN
TIME_STEP_S: typing.Final[float] = 0.01  # [s] 100Hz update rate
MS_PER_SECOND: typing.Final[int] = 1000  # [ms/s]
INITIAL_PELVIS_HEIGHT_M: typing.Final[float] = 1.0  # [m] Standing height
SPINBOX_STEP_RAD: typing.Final[float] = 0.1  # [rad] Step size for UI

# UI Styles
STYLE_BUTTON_RUN: typing.Final[str] = "background-color: #ccffcc;"  # Light Green
STYLE_BUTTON_STOP: typing.Final[str] = "background-color: #ffcccc;"  # Light Red


class DrakeSimApp(QtWidgets.QMainWindow):  # type: ignore[misc, no-any-unimported]
    """Main GUI Window for Drake Golf Simulation."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Drake Golf Swing Analysis")
        self.resize(500, 800)

        # Simulation State
        self.simulator: Simulator | None = None  # type: ignore[no-any-unimported]
        self.diagram: Diagram | None = None  # type: ignore[no-any-unimported]
        self.plant: MultibodyPlant | None = None  # type: ignore[no-any-unimported]
        self.context: Context | None = None  # type: ignore[no-any-unimported]
        self.meshcat: Meshcat | None = None  # type: ignore[no-any-unimported]
        self.visualizer: DrakeVisualizer | None = None

        self.operating_mode = "dynamic"  # "dynamic" or "kinematic"
        self.is_running = False
        self.time_step = TIME_STEP_S
        self.sliders: dict[int, QtWidgets.QSlider] = {}  # type: ignore[no-any-unimported]
        self.spinboxes: dict[int, QtWidgets.QDoubleSpinBox] = {}  # type: ignore[no-any-unimported]

        # Initialize Simulation
        self._init_simulation()

        # UI Setup
        self._setup_ui()

        # Sync initial state to UI
        self._sync_kinematic_sliders()

        # Timer for loop
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._game_loop)
        self.timer.start(int(self.time_step * MS_PER_SECOND))

    def _init_simulation(self) -> None:
        """Initialize Drake simulation and Meshcat."""
        try:
            meshcat_params = MeshcatParams()
            # "0.0.0.0" is required for Docker port forwarding
            meshcat_params.host = os.environ.get("MESHCAT_HOST", "localhost")
            self.meshcat = Meshcat(meshcat_params)
            LOGGER.info("Meshcat available at: %s", self.meshcat.web_url())

            # Open browser automatically only if running locally (not in Docker)
            # In Docker, the launcher handles opening the browser on the host.
            if self.meshcat:
                if "MESHCAT_HOST" not in os.environ:
                    webbrowser.open(self.meshcat.web_url())
                else:
                    LOGGER.info(
                        "Running in Docker/Headless mode; "
                        "skipping auto-browser open inside container."
                    )
                    LOGGER.info(
                        "Please access Meshcat from your host browser (e.g., http://localhost:7000)."
                    )

        except Exception as e:
            LOGGER.exception("Failed to start Meshcat")
            LOGGER.error(  # noqa: TRY400 - Manual logging preferred over re-raising.
                "Failed to start Meshcat for Drake visualization.\n"
                "Common causes:\n"
                "  - Another Meshcat server is already running on the same port "
                "(default: 7000).\n"
                "  - Network issues or firewall blocking localhost.\n"
                "  - Meshcat or its dependencies are not installed correctly.\n"
                "Troubleshooting steps:\n"
                "  1. Check if another Meshcat process is running and terminate it "
                "if necessary.\n"
                "  2. Verify that your firewall allows connections to "
                "localhost:7000.\n"
                "  3. Ensure all required Python packages are installed "
                "(see project README).\n"
                "Original exception: %s",
                e,
            )
            # We don't return here anymore, allowing simulation to run
            # without Meshcat if needed
            # But let's keep visualizer optional
            self.meshcat = None

        params = GolfModelParams()
        self.diagram, self.plant, _ = build_golf_swing_diagram(
            params, meshcat=self.meshcat
        )

        self.simulator = Simulator(self.diagram)
        self.simulator.set_target_realtime_rate(1.0)
        self.simulator.Initialize()

        self.context = self.simulator.get_mutable_context()
        if self.plant is None:
            msg = "Plant initialization failed"
            raise RuntimeError(msg)

        # Only initialize visualizer if Meshcat is available
        if self.meshcat is not None:
            self.visualizer = DrakeVisualizer(self.meshcat, self.plant)
        else:
            LOGGER.warning("Visualizer disabled due to Meshcat initialization failure.")

        # Initial State
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset simulation state."""
        plant = self.plant
        context = self.context
        diagram = self.diagram

        if not plant or not context or not diagram:
            return

        context.SetTime(0.0)
        plant_context = plant.GetMyContextFromRoot(context)

        # Set default pose (standing)
        # We need to find the specific joints or bodies.
        # Just creating a generic reset for now.
        pelvis = plant.GetBodyByName("pelvis")
        # In newer Drake, SetFreeBodyPose takes RigidTransform

        plant.SetFreeBodyPose(
            plant_context, pelvis, RigidTransform([0, 0, INITIAL_PELVIS_HEIGHT_M])
        )

        # Zero out velocities
        from numpy import zeros

        plant.SetVelocities(plant_context, zeros(plant.num_velocities()))

        if self.simulator:
            self.simulator.Initialize()

        diagram.ForcedPublish(context)

        # Sync generic UI controls if needed
        self._sync_kinematic_sliders()

    def _setup_ui(self) -> None:  # noqa: PLR0915
        """Build the PyQt Interface."""
        # ... (implementation same as before, no state access needed here mostly) ...
        # But wait, _build_kinematic_controls uses state.

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # 1. Mode Selector
        mode_group = QtWidgets.QGroupBox("Operating Mode")
        mode_layout = QtWidgets.QHBoxLayout()
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Dynamic (Physics)", "Kinematic (Pose)"])
        self.mode_combo.setToolTip(
            "Select between physics simulation or manual pose control"
        )
        self.mode_combo.setStatusTip(
            "Select between physics simulation or manual pose control"
        )
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(QtWidgets.QLabel("Mode:"))
        mode_layout.addWidget(self.mode_combo)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # 2. Controls Area (Stack)
        self.controls_stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.controls_stack)

        # -- Page 1: Dynamic Controls
        dynamic_page = QtWidgets.QWidget()
        dyn_layout = QtWidgets.QVBoxLayout(dynamic_page)

        self.btn_run = QtWidgets.QPushButton("▶ Run Simulation")
        self.btn_run.setCheckable(True)
        self.btn_run.setToolTip("Start or stop the physics simulation (Space)")
        self.btn_run.setStatusTip("Start or stop the physics simulation (Space)")
        self.btn_run.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Space))
        self.btn_run.setStyleSheet(STYLE_BUTTON_RUN)
        self.btn_run.clicked.connect(self._toggle_run)
        dyn_layout.addWidget(self.btn_run)

        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.setToolTip("Reset the simulation to the initial state (Ctrl+R)")
        self.btn_reset.setStatusTip(
            "Reset the simulation to the initial state (Ctrl+R)"
        )
        self.btn_reset.setShortcut(QtGui.QKeySequence("Ctrl+R"))
        self.btn_reset.clicked.connect(self._reset_simulation)
        dyn_layout.addWidget(self.btn_reset)

        dyn_layout.addStretch()
        self.controls_stack.addWidget(dynamic_page)

        # -- Page 2: Kinematic Controls
        kinematic_page = QtWidgets.QWidget()
        kin_layout = QtWidgets.QVBoxLayout(kinematic_page)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.kinematic_content = QtWidgets.QWidget()
        self.kinematic_layout = QtWidgets.QVBoxLayout(self.kinematic_content)
        scroll.setWidget(self.kinematic_content)

        kin_layout.addWidget(scroll)
        self.controls_stack.addWidget(kinematic_page)

        # 3. Visualization Toggles
        vis_group = QtWidgets.QGroupBox("Visualization")
        vis_layout = QtWidgets.QVBoxLayout()

        self.btn_overlays = QtWidgets.QPushButton("Manage Body Overlays")
        self.btn_overlays.setToolTip(
            "Toggle visibility of reference frames and centers of mass"
        )
        self.btn_overlays.setStatusTip(
            "Toggle visibility of reference frames and centers of mass"
        )
        self.btn_overlays.clicked.connect(self._show_overlay_dialog)
        vis_layout.addWidget(self.btn_overlays)

        # Ellipsoid Toggles
        self.chk_mobility = QtWidgets.QCheckBox("Show Mobility Ellipsoid (Green)")
        self.chk_mobility.toggled.connect(self._on_visualization_changed)
        vis_layout.addWidget(self.chk_mobility)

        self.chk_force_ellip = QtWidgets.QCheckBox("Show Force Ellipsoid (Red)")
        self.chk_force_ellip.toggled.connect(self._on_visualization_changed)
        vis_layout.addWidget(self.chk_force_ellip)

        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)

        # Matrix Analysis
        matrix_group = QtWidgets.QGroupBox("Matrix Analysis")
        matrix_layout = QtWidgets.QFormLayout(matrix_group)
        self.lbl_cond = QtWidgets.QLabel("--")
        self.lbl_rank = QtWidgets.QLabel("--")
        matrix_layout.addRow("Jacobian Cond:", self.lbl_cond)
        matrix_layout.addRow("Constraint Rank:", self.lbl_rank)
        layout.addWidget(matrix_group)

        # Status Bar
        self._update_status("Ready")

        # Populate Kinematic Sliders
        self._build_kinematic_controls()

    def _build_kinematic_controls(self) -> None:  # noqa: PLR0915
        """Create sliders for all joints."""
        plant = self.plant
        if not plant:
            return

        self.sliders.clear()
        self.spinboxes.clear()

        # Iterate over joints
        for i in range(plant.num_joints()):
            # Safe way to get joint index in PyDrake?
            # Index is typically just 'i' if iterating, but let's be careful.
            # Using Plant.get_joint(JointIndex(i))

            joint = plant.get_joint(JointIndex(i))

            # Skip welds (0 DOF) and multi-DOF joints (not yet supported)
            if joint.num_positions() != 1:
                continue

            # Create control row
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            label = QtWidgets.QLabel(f"{joint.name()}:")
            label.setMinimumWidth(120)
            row_layout.addWidget(label)

            # Slider
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            label.setBuddy(slider)
            slider.setRange(SLIDER_RANGE_MIN, SLIDER_RANGE_MAX)
            slider.setValue(0)

            # Determine joint limits for tooltip: prefer physical, else UI
            try:
                # For single-DOF joints, these are length-1 arrays
                # We need to access limits from the plant or joint model
                joint_min = float(joint.position_lower_limits()[0])
                joint_max = float(joint.position_upper_limits()[0])
            except Exception:  # noqa: BLE001
                # Fallback to UI limits if joint does not provide limits
                joint_min = JOINT_ANGLE_MIN_RAD
                joint_max = JOINT_ANGLE_MAX_RAD

            slider.setToolTip(
                f"Adjust angle for {joint.name()} (radians, "
                f"{joint_min:.2f} to {joint_max:.2f})"
            )

            # Spinbox
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(JOINT_ANGLE_MIN_RAD, JOINT_ANGLE_MAX_RAD)
            spin.setSingleStep(SPINBOX_STEP_RAD)
            spin.setDecimals(3)

            # Connect
            # Use closure or partial to capture joint index
            # Drake's joint.index() returns a JointIndex wrapper, which is
            # always convertible to int.
            # By Drake API contract, JointIndex is a non-negative integer less
            # than plant.num_joints().
            j_idx = int(joint.index())
            if not (0 <= j_idx < plant.num_joints()):
                msg = (
                    f"Joint index {j_idx} out of bounds for plant with "
                    f"{plant.num_joints()} joints."
                )
                raise ValueError(msg)

            slider.valueChanged.connect(
                lambda val, s=spin, idx=j_idx: self._on_slider_change(val, s, idx)
            )
            spin.valueChanged.connect(
                lambda val, s=slider, idx=j_idx: self._on_spin_change(val, s, idx)
            )

            row_layout.addWidget(slider)
            row_layout.addWidget(spin)

            self.kinematic_layout.addWidget(row)

            self.sliders[j_idx] = slider
            self.spinboxes[j_idx] = spin

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
        if self.operating_mode != "kinematic":
            return

        plant = self.plant
        context = self.context
        diagram = self.diagram

        if not plant or not context or not diagram:
            return

        plant_context = plant.GetMyContextFromRoot(context)

        joint = plant.get_joint(JointIndex(joint_idx))

        # Assuming single DOF revolute/prismatic for now
        if joint.num_positions() == 1:
            # Generic way:
            # joint.index() returns JointIndex wrapper, cast to int
            # joint.set_angle(plant_context, angle) is for RevoluteJoint
            if isinstance(joint, RevoluteJoint):
                joint.set_angle(plant_context, angle)
            elif isinstance(joint, PrismaticJoint):
                joint.set_translation(plant_context, angle)

        diagram.ForcedPublish(context)

        # Update overlays
        if self.visualizer:
            self.visualizer.update_frame_transforms(context)
            self.visualizer.update_com_transforms(context)
            self._update_ellipsoids()

    def _sync_kinematic_sliders(self) -> None:
        """Read current plant state and update sliders."""
        plant = self.plant
        context = self.context
        if not plant or not context:
            return

        plant_context = plant.GetMyContextFromRoot(context)

        for j_idx, spin in self.spinboxes.items():
            joint = plant.get_joint(JointIndex(j_idx))
            if joint.num_positions() == 1:
                val = joint.GetOnePosition(plant_context)
                spin.setValue(val)

    def _update_status(self, message: str) -> None:
        """Update status bar message safely."""
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage(message)

    def _on_mode_changed(self, text: str) -> None:
        if "Kinematic" in text:
            self.operating_mode = "kinematic"
            self.controls_stack.setCurrentIndex(1)
            self.is_running = False
            self.btn_run.setChecked(False)
            self.btn_run.setStyleSheet(STYLE_BUTTON_RUN)
            self.btn_run.setText("▶ Run Simulation")
            self._update_status("Mode: Kinematic Control")
            self._sync_kinematic_sliders()
            # Stop physics, allow manual
        else:
            self.operating_mode = "dynamic"
            self.controls_stack.setCurrentIndex(0)
            self._update_status("Mode: Dynamic Simulation")
            # Ensure simulation resumes or is stopped
            if self.is_running:
                self.btn_run.setText("■ Stop Simulation")
                self.btn_run.setChecked(True)
                self.btn_run.setStyleSheet(STYLE_BUTTON_STOP)
            else:
                self.btn_run.setText("▶ Run Simulation")
                self.btn_run.setChecked(False)
                self.btn_run.setStyleSheet(STYLE_BUTTON_RUN)

    def _toggle_run(self, checked: bool) -> None:  # noqa: FBT001
        self.is_running = checked
        if checked:
            self.btn_run.setText("■ Stop Simulation")
            self.btn_run.setStyleSheet(STYLE_BUTTON_STOP)
            self._update_status("Simulation Running...")
        else:
            self.btn_run.setText("▶ Run Simulation")
            self.btn_run.setStyleSheet(STYLE_BUTTON_RUN)
            self._update_status("Simulation Stopped.")

    def _reset_simulation(self) -> None:
        self.is_running = False
        self.btn_run.setChecked(False)
        self.btn_run.setText("▶ Run Simulation")
        self.btn_run.setStyleSheet(STYLE_BUTTON_RUN)
        self._update_status("Simulation Reset.")
        self._reset_state()

    def _game_loop(self) -> None:
        simulator = self.simulator
        context = self.context

        if not simulator or not context:
            return

        if self.operating_mode == "dynamic" and self.is_running:
            t = context.get_time()
            simulator.AdvanceTo(t + self.time_step)

            # Visual update
            if self.visualizer:
                self.visualizer.update_frame_transforms(context)
                self.visualizer.update_com_transforms(context)
                self._update_ellipsoids()

    def _on_visualization_changed(self) -> None:
        """Handle toggling of visualization options."""
        if self.visualizer:
            if not (self.chk_mobility.isChecked() or self.chk_force_ellip.isChecked()):
                self.visualizer.clear_ellipsoids()
            else:
                self._update_ellipsoids()

    def _update_ellipsoids(self) -> None:
        """Compute and draw ellipsoids."""
        if not (self.chk_mobility.isChecked() or self.chk_force_ellip.isChecked()):
            return

        if not self.plant or not self.context or not self.visualizer:
            return

        plant_context = self.plant.GetMyContextFromRoot(self.context)

        # Use end effector (last body?)
        # For golf, look for a body named "clubhead", "club_body", or just last body.
        # Fallback to last body if specific ones not found.
        body_names = ["clubhead", "club_body", "wrist", "hand", "link_7"]
        target_body = None
        for name in body_names:
            if self.plant.HasBodyNamed(name):
                target_body = self.plant.GetBodyByName(name)
                break

        if target_body is None:
            # Last body
            target_body = self.plant.get_body(BodyIndex(self.plant.num_bodies() - 1))

        if target_body.name() == "world":
            return

        # Jacobian
        # We need Jacobian with respect to velocities (v)
        frame_W = self.plant.world_frame()
        frame_B = target_body.body_frame()

        J_spatial = self.plant.CalcJacobianSpatialVelocity(
            plant_context, JacobianWrtVariable.kV, frame_B, [0, 0, 0], frame_W, frame_W
        )
        # 6 x nv matrix. Top 3 rotational, bottom 3 translational.
        # Use Translational part for visualization
        J = J_spatial[3:, :]

        # Mass Matrix
        M = self.plant.CalcMassMatrix(plant_context)

        try:
            # Condition Number
            s = np.linalg.svd(J, compute_uv=False)
            cond = s[0] / s[-1] if s[-1] > 1e-9 else float("inf")
            self.lbl_cond.setText(f"{cond:.2f}")

            # Constraint Rank (if any constraints?)
            # Drake handles constraints differently.
            # We can show Mass Matrix rank or check if M is singular.
            rank = np.linalg.matrix_rank(M)
            self.lbl_rank.setText(f"{rank} / {self.plant.num_velocities()}")

            # Ellipsoid
            Minv = np.linalg.inv(M)
            Lambda_inv = J @ Minv @ J.T

            eigvals, eigvecs = np.linalg.eigh(Lambda_inv)

            X_WB = self.plant.EvalBodyPoseInWorld(plant_context, target_body)
            pos = X_WB.translation()

            if self.chk_mobility.isChecked():
                radii = np.sqrt(np.maximum(eigvals, 1e-6))
                self.visualizer.draw_ellipsoid(
                    "mobility", eigvecs, radii, pos, (0, 1, 0, 0.3)
                )

            if self.chk_force_ellip.isChecked():
                radii_f = 1.0 / np.sqrt(np.maximum(eigvals, 1e-6))
                radii_f = np.clip(radii_f, 0.01, 5.0)
                self.visualizer.draw_ellipsoid(
                    "force", eigvecs, radii_f, pos, (1, 0, 0, 0.3)
                )

        except Exception as e:
            LOGGER.warning(f"Ellipsoid calc error: {e}")

    def _show_overlay_dialog(self) -> None:  # noqa: PLR0915
        """Show dialog to toggle overlays for specific bodies."""
        plant = self.plant
        diagram = self.diagram
        context = self.context
        visualizer = self.visualizer

        if not plant or not diagram or not context or not visualizer:
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Manage Overlays")
        layout = QtWidgets.QVBoxLayout(dialog)

        scroll = QtWidgets.QScrollArea()
        content = QtWidgets.QWidget()
        c_layout = QtWidgets.QVBoxLayout(content)

        # List all bodies
        for i in range(plant.num_bodies()):
            body = plant.get_body(BodyIndex(i))
            name = body.name()
            if name == "world":
                continue

            b_row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(name)

            chk_frame = QtWidgets.QCheckBox("Frame")
            is_vis_f = name in visualizer.visible_frames
            chk_frame.setChecked(is_vis_f)
            chk_frame.toggled.connect(lambda c, n=name: visualizer.toggle_frame(n, c))

            chk_com = QtWidgets.QCheckBox("COM")
            is_vis_c = name in visualizer.visible_coms
            chk_com.setChecked(is_vis_c)
            chk_com.toggled.connect(lambda c, n=name: visualizer.toggle_com(n, c))

            b_row.addWidget(lbl)
            b_row.addWidget(chk_frame)
            b_row.addWidget(chk_com)
            c_layout.addLayout(b_row)

        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)

        dialog.exec()

        # Refresh kinematics to update frame positions immediately
        if self.operating_mode == "kinematic":
            diagram.ForcedPublish(context)
            if self.visualizer:
                self.visualizer.update_frame_transforms(context)
                self.visualizer.update_com_transforms(context)


def main() -> None:
    setup_logging()
    app = QtWidgets.QApplication(sys.argv)
    window = DrakeSimApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    from pydrake.all import JacobianWrtVariable  # Import here for use in method

    main()
