"""Drake Golf Swing Analysis GUI Application."""

from __future__ import annotations

import logging
import os
import sys
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Qt imports
try:
    from PyQt6 import QtCore, QtGui, QtWidgets

    HAS_QT = True
except ImportError:
    HAS_QT = False
    QtCore = None  # type: ignore[misc, assignment]
    QtGui = None  # type: ignore[misc, assignment]
    QtWidgets = None  # type: ignore[misc, assignment]

# Matplotlib imports
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore[misc, assignment]

# Drake imports
if TYPE_CHECKING or HAS_QT:
    try:
        from pydrake.all import (
            AddMultibodyPlantSceneGraph,
            BodyIndex,
            Context,
            Diagram,
            DiagramBuilder,
            DrakeVisualizer,
            JacobianWrtVariable,
            JointIndex,
            Meshcat,
            MeshcatParams,
            MeshcatVisualizer,
            MultibodyPlant,
            Parser,
            PrismaticJoint,
            RevoluteJoint,
            RigidTransform,
            Simulator,
        )
    except ImportError:
        # Fallback for when Drake is not available
        AddMultibodyPlantSceneGraph = None  # type: ignore[misc, assignment]
        BodyIndex = None  # type: ignore[misc, assignment]
        Context = None  # type: ignore[misc, assignment]
        Diagram = None  # type: ignore[misc, assignment]
        DiagramBuilder = None  # type: ignore[misc, assignment]
        DrakeVisualizer = None  # type: ignore[misc, assignment]
        JointIndex = None  # type: ignore[misc, assignment]
        JacobianWrtVariable = None  # type: ignore[misc, assignment]
        Meshcat = None  # type: ignore[misc, assignment]
        MeshcatParams = None  # type: ignore[misc, assignment]
        MeshcatVisualizer = None  # type: ignore[misc, assignment]
        MultibodyPlant = None  # type: ignore[misc, assignment]
        Parser = None  # type: ignore[misc, assignment]
        PrismaticJoint = None  # type: ignore[misc, assignment]
        RevoluteJoint = None  # type: ignore[misc, assignment]
        RigidTransform = None  # type: ignore[misc, assignment]
        Simulator = None  # type: ignore[misc, assignment]

# Shared imports
try:
    from shared.python.plotting import GolfSwingPlotter
    from shared.python.statistical_analysis import StatisticalAnalyzer
except ImportError:
    GolfSwingPlotter = None  # type: ignore[misc, assignment]
    StatisticalAnalyzer = None  # type: ignore[misc, assignment]

# Try to import golf model components
try:
    from engines.physics_engines.drake.python.src.drake_golf_model import (
        GolfModelParams,
        build_golf_swing_diagram,
    )
except ImportError:
    # Fallback classes
    class GolfModelParams:  # type: ignore[no-redef]
        """Placeholder for golf model parameters."""

        pass

    def build_golf_swing_diagram(*args, **kwargs):  # type: ignore[no-redef, misc]
        """Placeholder for golf swing diagram builder."""
        return None, None, None


# Constants
TIME_STEP_S = 0.001
MS_PER_SECOND = 1000
JOINT_ANGLE_MIN_RAD = -3.14159
JOINT_ANGLE_MAX_RAD = 3.14159
SPINBOX_STEP_RAD = 0.01
SLIDER_TO_RADIAN = 0.01
SLIDER_RANGE_MIN = -314
SLIDER_RANGE_MAX = 314
INITIAL_PELVIS_HEIGHT_M = 1.0

# Styles
STYLE_BUTTON_RUN = "QPushButton { background-color: #4CAF50; color: white; }"
STYLE_BUTTON_STOP = "QPushButton { background-color: #f44336; color: white; }"

# Logger
LOGGER = logging.getLogger(__name__)


# Placeholder for missing classes
class DrakeInducedAccelerationAnalyzer:
    """Induced acceleration analyzer for Drake."""

    def __init__(self, plant: MultibodyPlant | None) -> None:
        self.plant = plant

    def compute_components(self, context: Context) -> dict[str, np.ndarray]:
        """Compute induced acceleration components.

        Args:
            context: The plant context (with q, v set)

        Returns:
            Dict with 'gravity', 'velocity', 'total' (passive)
        """
        if self.plant is None:
            return {
                "gravity": np.array([]),
                "velocity": np.array([]),
                "total": np.array([]),
            }

        # 1. Calc Mass Matrix
        M = self.plant.CalcMassMatrix(context)

        # 2. Calc Gravity Torque
        tau_g = self.plant.CalcGravityGeneralizedForces(context)

        # 3. Calc Bias Term (Cv - tau_g)
        bias = self.plant.CalcBiasTerm(context)

        # Invert M
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            Minv = np.linalg.pinv(M)

        a_g = Minv @ tau_g

        # Force due to velocity = -Cv = -(bias + tau_g)
        a_v = Minv @ (-(bias + tau_g))

        return {"gravity": a_g, "velocity": a_v, "total": a_g + a_v}

    def compute_counterfactuals(self, context: Context) -> dict[str, np.ndarray]:
        """Compute ZTCF and ZVCF."""
        if self.plant is None:
            return {}

        # ZTCF (Zero Torque Accel): a = -M^-1 (C + G).
        # We assume zero torque applied.
        # This is essentially the passive dynamics accel.
        # We already computed this as 'total' in compute_components if we sum a_g + a_v.
        # Or specifically: M a + Cv - tau_g = 0 => M a = tau_g - Cv = -bias.
        # So a_ztcf = -M^-1 * bias.

        M = self.plant.CalcMassMatrix(context)
        bias = self.plant.CalcBiasTerm(context)

        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            Minv = np.linalg.pinv(M)

        ztcf_accel = Minv @ (-bias)

        # ZVCF (Zero Velocity Torque):
        # M a + G = tau. If v=0, C=0.
        # If we hold position (a=0, v=0), tau = G.
        # tau_g = CalcGravityGeneralizedForces.
        # Wait, equation is M vdot + Cv - tau_g = tau.
        # If v=0 => Cv=0. If a=0 => M vdot = 0.
        # So -tau_g = tau => tau = -tau_g?
        # Drake defines tau_g as forces on RHS.
        # So tau_holding = -tau_g.

        tau_g = self.plant.CalcGravityGeneralizedForces(context)
        zvcf_torque = -tau_g

        return {"ztcf_accel": ztcf_accel, "zvcf_torque": zvcf_torque}

    def compute_specific_control(self, context: Context, tau: np.ndarray) -> np.ndarray:
        """Compute induced acceleration for a specific control vector.

        Note:
            This method calculates the acceleration induced solely by the provided
            torque vector `tau`.
            It solves M * a = tau. If `tau` represents a unit torque
            (e.g., [0, 1, 0]), the result is the sensitivity of acceleration
            to that specific actuator.
        """
        if self.plant is None:
            return np.array([])

        # M * a = tau
        M = self.plant.CalcMassMatrix(context)
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            Minv = np.linalg.pinv(M)

        return Minv @ tau


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO)


class DrakeRecorder:
    """Records simulation data for analysis."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.times: list[float] = []
        self.q_history: list[np.ndarray] = []
        self.v_history: list[np.ndarray] = []
        self.club_head_pos_history: list[np.ndarray] = []
        # Store computed metrics
        self.induced_accelerations: dict[str, list[np.ndarray]] = {}
        self.counterfactuals: dict[str, list[np.ndarray]] = {}
        self.is_recording = False

    def start(self) -> None:
        self.reset()
        self.is_recording = True

    def stop(self) -> None:
        self.is_recording = False

    def record(
        self,
        t: float,
        q: np.ndarray,
        v: np.ndarray,
        club_pos: np.ndarray | None = None,
    ) -> None:
        if not self.is_recording:
            return
        self.times.append(t)
        self.q_history.append(q.copy())
        self.v_history.append(v.copy())
        if club_pos is not None:
            self.club_head_pos_history.append(club_pos.copy())
        else:
            self.club_head_pos_history.append(np.zeros(3))

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray | list]:
        """Implement RecorderInterface."""
        times = np.array(self.times)
        if field_name == "club_head_position":
            return times, np.array(self.club_head_pos_history)
        if field_name == "joint_positions":
            return times, np.array(self.q_history)
        if field_name == "joint_velocities":
            return times, np.array(self.v_history)

        # Fallback
        return times, []

    def get_induced_acceleration_series(
        self, source_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get induced accelerations."""
        if source_name not in self.induced_accelerations:
            return np.array([]), np.array([])

        times = np.array(self.times)
        # Ensure alignment
        vals = self.induced_accelerations[source_name]
        if len(vals) != len(times):
            # Truncate to match
            min_len = min(len(vals), len(times))
            return times[:min_len], np.array(vals[:min_len])

        return times, np.array(vals)

    def get_counterfactual_series(self, cf_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get counterfactual data."""
        if cf_name not in self.counterfactuals:
            return np.array([]), np.array([])

        times = np.array(self.times)
        vals = self.counterfactuals[cf_name]

        if len(vals) != len(times):
            min_len = min(len(vals), len(times))
            return times[:min_len], np.array(vals[:min_len])

        return times, np.array(vals)


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
        self.visualizer: DrakeVisualizer | None = None  # type: ignore[no-any-unimported]
        self.operating_mode = "dynamic"  # "dynamic" or "kinematic"
        self.is_running = False
        self.time_step = TIME_STEP_S
        self.sliders: dict[int, QtWidgets.QSlider] = {}  # type: ignore[no-any-unimported]
        self.spinboxes: dict[int, QtWidgets.QDoubleSpinBox] = {}  # type: ignore[no-any-unimported]

        self.recorder = DrakeRecorder()
        self.eval_context: Context | None = None  # type: ignore[no-any-unimported]

        # Model Management
        self.current_urdf_path: str | None = None
        self.available_models: list[dict] = [
            {"name": "Default Golf Model", "path": None}
        ]
        self._scan_urdf_models()

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

    def _scan_urdf_models(self) -> None:
        """Scan shared/urdf for models."""
        try:
            # Calculate path to shared directory relative to this file
            # engines/physics_engines/drake/python/src/drake_gui_app.py
            current_file = Path(__file__)

            # Check for Docker environment mount first
            docker_shared = Path("/shared/urdf")
            if docker_shared.exists():
                urdf_dir = docker_shared
                LOGGER.info(f"Found Docker shared URDF directory: {urdf_dir}")
            else:
                # Fallback to local relative path
                # Up 5 levels: src->python->drake->physics_engines->engines->root
                try:
                    project_root = current_file.parents[5]
                    urdf_dir = project_root / "shared" / "urdf"
                except IndexError:
                     # Fallback for when path depth is insufficient
                     urdf_dir = Path("non_existent")

            if urdf_dir.exists():
                for urdf_file in urdf_dir.glob("*.urdf"):
                    name = urdf_file.stem.replace("_", " ").title()
                    self.available_models.append(
                        {"name": f"URDF: {name}", "path": str(urdf_file)}
                    )
        except Exception as e:
            LOGGER.error(f"Failed to scan URDF models: {e}")

    def _init_simulation(self) -> None:
        """Initialize Drake simulation and Meshcat."""
        if self.meshcat is None:
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

        # Build Diagram
        if self.current_urdf_path:
            # Load custom URDF
            self._build_custom_urdf_diagram(self.current_urdf_path)
        else:
            # Load default golf model
            params = GolfModelParams()
            self.diagram, self.plant, _ = build_golf_swing_diagram(
                params, meshcat=self.meshcat
            )

        if self.diagram is None:
            # Create a simple placeholder diagram if build failed or returned None
            builder = DiagramBuilder()
            plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
            plant.Finalize()
            self.plant = plant
            self.diagram = builder.Build()

        self.simulator = Simulator(self.diagram)
        self.simulator.set_target_realtime_rate(1.0)
        self.simulator.Initialize()

        self.context = self.simulator.get_mutable_context()
        if self.plant is None:
            msg = "Plant initialization failed"
            raise RuntimeError(msg)

        # Only initialize visualizer if Meshcat is available
        if self.meshcat is not None:
            # self.visualizer = DrakeVisualizer(self.meshcat, self.plant)
            # Use of pure pydrake.geometry.DrakeVisualizer here is incorrect
            # as it expects LCM parameters. Meshcat visualization is inserted
            # into the diagram during build_golf_swing_diagram.
            self.visualizer = None
        else:
            LOGGER.warning("Visualizer disabled due to Meshcat initialization failure.")

        # Create evaluation context for analysis
        self.eval_context = self.plant.CreateDefaultContext()

        # Initial State
        self._reset_state()

    def _build_custom_urdf_diagram(self, urdf_path: str) -> None:
        """Build a simple diagram for a custom URDF."""
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
        parser = Parser(plant)
        parser.AddModels(urdf_path)
        plant.Finalize()

        if self.meshcat:
            MeshcatVisualizer.AddToBuilder(builder, scene_graph, self.meshcat)

        self.plant = plant
        self.diagram = builder.Build()

    def _reset_state(self) -> None:
        """Reset simulation state."""
        plant = self.plant
        context = self.context
        diagram = self.diagram

        if not plant or not context or not diagram:
            return

        context.SetTime(0.0)
        plant_context = plant.GetMyContextFromRoot(context)

        # Set default pose (standing) if 'pelvis' exists (Golf Model)
        if plant.HasBodyNamed("pelvis"):
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

    def _on_model_changed(self, index: int) -> None:
        """Handle model change."""
        model_data = self.available_models[index]
        new_path = model_data["path"]

        if new_path != self.current_urdf_path:
            self.current_urdf_path = new_path

            # Re-initialize simulation
            # We need to stop the timer temporarily to avoid thread issues
            self.timer.stop()
            try:
                self._init_simulation()
                self._build_kinematic_controls()
                self._sync_kinematic_sliders()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error Loading Model", str(e))
                # Revert selection? For now just log.
                LOGGER.error(f"Error loading model: {e}")
            finally:
                self.timer.start(int(self.time_step * MS_PER_SECOND))

    def _setup_ui(self) -> None:  # noqa: PLR0915
        """Build the PyQt Interface."""
        # ... (implementation same as before, no state access needed here mostly) ...
        # But wait, _build_kinematic_controls uses state.

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # 0. Model Selector
        model_group = QtWidgets.QGroupBox("Model Selection")
        model_layout = QtWidgets.QHBoxLayout()
        self.model_combo = QtWidgets.QComboBox()
        for model in self.available_models:
            self.model_combo.addItem(model["name"])
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_layout.addWidget(QtWidgets.QLabel("Model:"))
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

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

        # Recording & Analysis
        analysis_group = QtWidgets.QGroupBox("Analysis")
        analysis_layout = QtWidgets.QVBoxLayout()

        rec_row = QtWidgets.QHBoxLayout()
        self.btn_record = QtWidgets.QPushButton("Record")
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self._toggle_recording)
        self.lbl_rec_status = QtWidgets.QLabel("Frames: 0")
        rec_row.addWidget(self.btn_record)
        rec_row.addWidget(self.lbl_rec_status)
        analysis_layout.addLayout(rec_row)

        # Induced Accel
        ind_layout = QtWidgets.QHBoxLayout()
        self.btn_induced_acc = QtWidgets.QPushButton("Show Induced Acceleration")
        self.btn_induced_acc.setToolTip(
            "Analyze Gravity/Velocity/Control contributions to Acceleration"
        )
        self.btn_induced_acc.clicked.connect(self._show_induced_acceleration_plot)
        self.btn_induced_acc.setEnabled(HAS_MATPLOTLIB)
        ind_layout.addWidget(self.btn_induced_acc)

        self.txt_specific_actuator = QtWidgets.QLineEdit()
        self.txt_specific_actuator.setPlaceholderText("Specific Actuator (index)")
        self.txt_specific_actuator.setToolTip("Index of actuator to isolate (optional)")
        self.txt_specific_actuator.setMaximumWidth(150)
        ind_layout.addWidget(self.txt_specific_actuator)

        analysis_layout.addLayout(ind_layout)

        self.btn_counterfactuals = QtWidgets.QPushButton(
            "Show Counterfactuals (ZTCF/ZVCF)"
        )
        self.btn_counterfactuals.setToolTip(
            "Show Zero Torque (ZTCF) and Zero Velocity (ZVCF) analysis"
        )
        self.btn_counterfactuals.clicked.connect(self._show_counterfactuals_plot)
        self.btn_counterfactuals.setEnabled(HAS_MATPLOTLIB)
        analysis_layout.addWidget(self.btn_counterfactuals)

        self.btn_swing_plane = QtWidgets.QPushButton("Show Swing Plane Analysis")
        self.btn_swing_plane.setToolTip("Analyze the swing plane and deviation")
        self.btn_swing_plane.clicked.connect(self._show_swing_plane_analysis)
        self.btn_swing_plane.setEnabled(HAS_MATPLOTLIB)
        analysis_layout.addWidget(self.btn_swing_plane)

        self.btn_advanced_plots = QtWidgets.QPushButton("Show Advanced Plots")
        self.btn_advanced_plots.setToolTip(
            "Show Radar Chart, CoP Field, and Power Flow"
        )
        self.btn_advanced_plots.clicked.connect(self._show_advanced_plots)
        self.btn_advanced_plots.setEnabled(HAS_MATPLOTLIB)
        analysis_layout.addWidget(self.btn_advanced_plots)

        analysis_group.setLayout(analysis_layout)
        dyn_layout.addWidget(analysis_group)

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

            # Recording
            if self.recorder.is_recording and self.plant:
                plant_context = self.plant.GetMyContextFromRoot(context)
                q = self.plant.GetPositions(plant_context)
                v = self.plant.GetVelocities(plant_context)

                # Get club head position
                club_pos = None
                body_names = ["clubhead", "club_body", "wrist", "hand", "link_7"]
                for name in body_names:
                    if self.plant.HasBodyNamed(name):
                        body = self.plant.GetBodyByName(name)
                        X_WB = self.plant.EvalBodyPoseInWorld(plant_context, body)
                        club_pos = X_WB.translation()
                        break

                if club_pos is None:
                    # Fallback to last body
                    body = self.plant.get_body(BodyIndex(self.plant.num_bodies() - 1))
                    X_WB = self.plant.EvalBodyPoseInWorld(plant_context, body)
                    club_pos = X_WB.translation()

                self.recorder.record(context.get_time(), q, v, club_pos)
                self.lbl_rec_status.setText(f"Frames: {len(self.recorder.times)}")

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
            cond = s[0] / s[-1] if s[-1] > 1e-9 else float("inf")  # Fix line length
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

    def _toggle_recording(self, checked: bool) -> None:  # noqa: FBT001
        if checked:
            self.recorder.start()
            self.btn_record.setText("Stop Recording")
            self._update_status("Recording started...")
        else:
            self.recorder.stop()
            self.btn_record.setText("Record")
            self._update_status(
                f"Recording stopped. Total Frames: {len(self.recorder.times)}"
            )

    def _show_induced_acceleration_plot(self) -> None:
        """Calculate and plot induced accelerations."""
        if not HAS_MATPLOTLIB:
            QtWidgets.QMessageBox.warning(self, "Error", "Matplotlib not found.")
            return

        if not self.recorder.times:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No recording available. Please Record a simulation first.",
            )
            return

        if not self.plant or not self.eval_context:
            return

        # Specific Actuator Input
        spec_act_idx = -1
        txt = self.txt_specific_actuator.text().strip()
        if txt:
            try:
                spec_act_idx = int(txt)
                if spec_act_idx < 0 or spec_act_idx >= self.plant.num_actuators():
                    # Wait, num_actuators vs num_joints.
                    # We usually control joints.
                    pass
            except ValueError:
                pass

        # Computation
        times = self.recorder.times
        g_induced = []
        c_induced = []
        spec_induced = []
        # Control is assumed 0 for now as we don't capture u

        analyzer = DrakeInducedAccelerationAnalyzer(self.plant)

        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

            for _i, (q, v) in enumerate(
                zip(self.recorder.q_history, self.recorder.v_history, strict=False)
            ):
                self.plant.SetPositions(self.eval_context, q)
                self.plant.SetVelocities(self.eval_context, v)

                # Use Analyzer (updated API)
                res = analyzer.compute_components(self.eval_context)

                g_induced.append(res["gravity"])
                c_induced.append(res["velocity"])

                if spec_act_idx >= 0:
                    # Construct torque vector
                    # Note: We need the ACTUAL torque applied.
                    # We don't record 'u' in recorder yet.
                    # So we can only simulate "Unit Torque" induced acceleration?
                    # Or "What acceleration would 1 Nm cause?"
                    # The prompt asks for "induced accelerations
                    # (for a specified joint torque source)".
                    # This usually means contribution of that source to motion.
                    # Without recording 'u', we assume 1.0 or 0.0?
                    # Let's compute Unit Torque response: Accel if tau[i]=1, others=0.

                    tau = np.zeros(self.plant.num_velocities())  # Generalized forces
                    # Map actuator index to generalized force index?
                    # If fully actuated, nu = nv.
                    if spec_act_idx < len(tau):
                        tau[spec_act_idx] = 1.0  # Unit torque

                    spec = analyzer.compute_specific_control(self.eval_context, tau)
                    spec_induced.append(spec)

        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Analysis Error", str(e))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        g_induced_arr = np.array(g_induced)
        c_induced_arr = np.array(c_induced)
        spec_induced_arr = np.array(spec_induced) if spec_induced else None

        # Total passive
        total_arr = g_induced_arr + c_induced_arr

        # Plotting - Select a joint (e.g., joint 0)
        # We can pick the Joint with largest movement or just the first.
        joint_idx = 0

        # Try to find a meaningful joint (e.g. Spine Twist) if available
        # But joint indices are internal.
        # Let's just pick index 2 (Hips/Spine?) or 0.
        if g_induced_arr.shape[1] > 2:
            joint_idx = 2

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            times,
            g_induced_arr[:, joint_idx],
            label="Gravity Induced",
            linestyle="--",
        )
        ax.plot(
            times,
            c_induced_arr[:, joint_idx],
            label="Velocity Induced",
            linestyle="-.",
        )
        ax.plot(times, total_arr[:, joint_idx], label="Total (Passive)", color="k")

        if spec_induced_arr is not None and spec_induced_arr.shape[0] > 0:
            ax.plot(
                times,
                spec_induced_arr[:, joint_idx],
                label=f"Induced by Act {spec_act_idx} (Unit Torque)",
                color="m",
                linewidth=2,
            )

        ax.set_title(f"Induced Acceleration Analysis (Joint Index {joint_idx})")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Acceleration [rad/s^2]")
        ax.legend()
        ax.grid(True)

        plt.show()

    def _show_counterfactuals_plot(self) -> None:
        """Calculate and plot Counterfactuals (ZTCF/ZVCF)."""
        if not HAS_MATPLOTLIB:
            QtWidgets.QMessageBox.warning(self, "Error", "Matplotlib not found.")
            return

        if not self.recorder.times:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No recording available. Please Record a simulation first.",
            )
            return

        if not self.plant or not self.eval_context:
            return

        analyzer = DrakeInducedAccelerationAnalyzer(self.plant)
        times = self.recorder.times
        ztcf_list = []
        zvcf_list = []

        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

            for q, v in zip(
                self.recorder.q_history, self.recorder.v_history, strict=False
            ):
                self.plant.SetPositions(self.eval_context, q)
                self.plant.SetVelocities(self.eval_context, v)

                res = analyzer.compute_counterfactuals(self.eval_context)
                ztcf_list.append(res["ztcf_accel"])
                zvcf_list.append(res["zvcf_torque"])

        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Analysis Error", str(e))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        ztcf_arr = np.array(ztcf_list)
        zvcf_arr = np.array(zvcf_list)

        # Plotting - Joint Index
        joint_idx = 0
        if ztcf_arr.shape[1] > 2:
            joint_idx = 2

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # ZTCF (Accel) on Left Axis
        color = "tab:blue"
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("ZTCF Acceleration [rad/s^2]", color=color)
        ax1.plot(times, ztcf_arr[:, joint_idx], color=color, label="ZTCF Accel")
        ax1.tick_params(axis="y", labelcolor=color)

        # ZVCF (Torque) on Right Axis
        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("ZVCF Torque [Nm]", color=color)
        ax2.plot(
            times,
            zvcf_arr[:, joint_idx],
            color=color,
            linestyle="--",
            label="ZVCF Torque",
        )
        ax2.tick_params(axis="y", labelcolor=color)

        plt.title(f"Counterfactual Analysis (Joint Index {joint_idx})")
        fig.tight_layout()
        plt.show()

    def _show_swing_plane_analysis(self) -> None:
        """Show swing plane analysis using shared plotter."""
        if not HAS_MATPLOTLIB:
            QtWidgets.QMessageBox.warning(self, "Error", "Matplotlib not found.")
            return

        # Import here to avoid hard dependency on matplotlib
        from shared.python.plotting import GolfSwingPlotter

        if not self.recorder.times:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No recording available. Please Record a simulation first.",
            )
            return

        plotter = GolfSwingPlotter(self.recorder)
        fig = plt.figure(figsize=(10, 8))
        plotter.plot_swing_plane(fig)
        plt.show()

    def _show_advanced_plots(self) -> None:
        """Show advanced analysis plots."""
        if not HAS_MATPLOTLIB:
            QtWidgets.QMessageBox.warning(self, "Error", "Matplotlib not found.")
            return

        from shared.python.plotting import GolfSwingPlotter
        from shared.python.statistical_analysis import StatisticalAnalyzer

        if not self.recorder.times:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No recording available. Please Record a simulation first.",
            )
            return

        # Create Plotter and Analyzer
        plotter = GolfSwingPlotter(self.recorder)

        # We need to extract data for StatisticalAnalyzer
        times = np.array(self.recorder.times)
        q_history = np.array(self.recorder.q_history)
        v_history = np.array(self.recorder.v_history)
        # Assuming torques not recorded in DrakeRecorder yet, need to update
        # if we want torque analysis
        # For now pass zeros for torques
        tau_history = np.zeros((len(times), v_history.shape[1]))

        _, club_pos = self.recorder.get_time_series("club_head_position")

        analyzer = StatisticalAnalyzer(
            times, q_history, v_history, tau_history, club_head_position=club_pos
        )
        report = analyzer.generate_comprehensive_report()

        # Prepare metrics for Radar Chart
        metrics = {
            "Club Speed": 0.0,
            "Swing Efficiency": 0.0,
            "Tempo": 0.0,
            "Consistency": 0.8,  # Placeholder
            "Power Transfer": 0.0,
        }

        if "club_head_speed" in report:
            # Normalize reasonably (e.g. max speed 50 m/s)
            peak_speed = report["club_head_speed"]["peak_value"]
            metrics["Club Speed"] = min(peak_speed / 50.0, 1.0)

        if "tempo" in report:
            ratio = report["tempo"]["ratio"]
            # Ideal 3:1 => 3.0. Normalize error from 3.0
            error = abs(ratio - 3.0)
            metrics["Tempo"] = max(0, 1.0 - error)

        # Create Figure with tabs or subplots
        # For simplicity, just use subplots in one figure
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)

        # 1. Radar Chart
        plotter.plot_radar_chart(fig, metrics)

        # 2. CoP Vector Field (Drake doesn't record CoP yet, so skip or mock)
        # If we had CoP data, we would call plotter.plot_cop_vector_field(fig)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(0.5, 0.5, "CoP Data Not Available in Drake", ha="center", va="center")

        # 3. Power Flow
        # Requires actuator powers. DrakeRecorder needs to record powers.
        # For now placeholder
        ax3 = fig.add_subplot(gs[1, :])
        ax3.text(
            0.5, 0.5, "Power Data Not Available in Drake", ha="center", va="center"
        )

        plt.tight_layout()
        plt.show()


def main() -> None:
    setup_logging()
    app = QtWidgets.QApplication(sys.argv)
    window = DrakeSimApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    from pydrake.all import JacobianWrtVariable  # Import here for use in method

    main()
