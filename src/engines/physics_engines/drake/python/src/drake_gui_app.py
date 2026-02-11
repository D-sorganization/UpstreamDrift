"""Drake Golf Swing Analysis GUI Application.

Refactored: DrakeRecorder and DrakeInducedAccelerationAnalyzer are in
``drake_recorder.py``.  UI construction is in ``drake_ui_mixin.py``.
Visualization/analysis is in ``drake_visualization_mixin.py``.
"""

from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add project root to path for src imports when run as standalone script
# Path: src/engines/physics_engines/drake/python/src/drake_gui_app.py -> need 7 parents
_project_root = (
    Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent
)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np  # noqa: E402

from src.shared.python.engine_availability import (  # noqa: E402
    PYQT6_AVAILABLE,
)
from src.shared.python.logging_config import get_logger  # noqa: E402
from src.shared.python.ui.simulation_gui_base import SimulationGUIBase  # noqa: E402

# Mixin and helper imports
from .drake_recorder import (  # noqa: E402
    DrakeInducedAccelerationAnalyzer,
    DrakeRecorder,
    setup_logging,
)
from .drake_ui_mixin import DrakeUIMixin  # noqa: E402
from .drake_visualization_mixin import DrakeVisualizationMixin  # noqa: E402

# Use centralized availability flags
HAS_QT = PYQT6_AVAILABLE

# Qt imports
if HAS_QT:
    from PyQt6 import QtCore, QtWidgets
else:
    QtCore = None  # type: ignore[misc, assignment]
    QtWidgets = None  # type: ignore[misc, assignment]

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
            RigidTransform,
            Simulator,
        )
    except ImportError:
        AddMultibodyPlantSceneGraph = None  # type: ignore[misc, assignment]
        BodyIndex = None  # type: ignore[misc, assignment]
        Context = None  # type: ignore[misc, assignment]
        Diagram = None  # type: ignore[misc, assignment]
        DiagramBuilder = None  # type: ignore[misc, assignment]
        DrakeVisualizer = None  # type: ignore[misc, assignment]
        JacobianWrtVariable = None  # type: ignore[misc, assignment]
        JointIndex = None  # type: ignore[misc, assignment]
        Meshcat = None  # type: ignore[misc, assignment]
        MeshcatParams = None  # type: ignore[misc, assignment]
        MeshcatVisualizer = None  # type: ignore[misc, assignment]
        MultibodyPlant = None  # type: ignore[misc, assignment]
        Parser = None  # type: ignore[misc, assignment]
        RigidTransform = None  # type: ignore[misc, assignment]
        Simulator = None  # type: ignore[misc, assignment]

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

    def build_golf_swing_diagram(
        params: GolfModelParams | None = None,
        urdf_path: str | None = None,
        meshcat: Any | None = None,
    ) -> tuple[Any, Any, Any]:
        """Placeholder for golf swing diagram builder."""
        return None, None, None


# Manipulability Import
try:
    from .manipulability import DrakeManipulabilityAnalyzer
except ImportError:
    DrakeManipulabilityAnalyzer = None  # type: ignore

# Constants
TIME_STEP_S = 0.001
MS_PER_SECOND = 1000
INITIAL_PELVIS_HEIGHT_M = 1.0

# Logger
LOGGER = get_logger(__name__)


class DrakeSimApp(  # type: ignore[misc, no-any-unimported]
    DrakeUIMixin,
    DrakeVisualizationMixin,
    SimulationGUIBase,
):
    """Main GUI Window for Drake Golf Simulation."""

    WINDOW_TITLE = "Drake Golf Swing Analysis"
    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 800

    def __init__(self) -> None:
        # Pre-init state needed before super().__init__() triggers _build_base_ui
        self._drake_pre_init_done = False
        super().__init__()

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

        # Pass self as engine to recorder so it can call get_joint_names
        self.recorder = DrakeRecorder(engine=self)
        self.eval_context: Context | None = None  # type: ignore[no-any-unimported]

        # Manipulability
        self.manip_analyzer: DrakeManipulabilityAnalyzer | None = None
        self.manip_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        self.manip_body_layout: QtWidgets.QGridLayout | None = None

        # Model Management
        self.current_urdf_path: str | None = None
        self.available_models: list[dict] = [
            {"name": "Default Golf Model", "path": None}
        ]
        self._scan_urdf_models()

        # Initialize Simulation
        self._init_simulation()

        # UI Setup (from DrakeUIMixin)
        self._setup_ui()

        # Sync initial state to UI
        self._sync_kinematic_sliders()

        # Timer for loop
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._game_loop)
        self.timer.start(int(self.time_step * MS_PER_SECOND))

    def get_joint_names(self) -> list[str]:
        """Return joint names for LivePlotWidget."""
        if not self.plant:
            return []

        names = []
        for i in range(self.plant.num_joints()):
            joint = self.plant.get_joint(JointIndex(i))
            # Only include 1-DOF joints for simplicity in plotting mapping
            if joint.num_velocities() == 1:
                names.append(joint.name())
        return names

    def _scan_urdf_models(self) -> None:
        """Scan shared/urdf for models."""
        try:
            current_file = Path(__file__)

            # Check for Docker environment mount first
            docker_shared = Path("/shared/urdf")
            if docker_shared.exists():
                urdf_dir = docker_shared
                LOGGER.info("Found Docker shared URDF directory: %s", urdf_dir)
            else:
                # Fallback to local relative path
                try:
                    project_root = current_file.parents[5]
                    urdf_dir = project_root / "shared" / "urdf"
                except IndexError:
                    urdf_dir = Path("non_existent")

            if urdf_dir.exists():
                for urdf_file in urdf_dir.glob("*.urdf"):
                    name = urdf_file.stem.replace("_", " ").title()
                    self.available_models.append(
                        {"name": f"URDF: {name}", "path": str(urdf_file)}
                    )
        except (FileNotFoundError, OSError) as e:
            LOGGER.error("Failed to scan URDF models: %s", e)

    def _init_simulation(self) -> None:
        """Initialize Drake simulation and Meshcat."""
        if self.meshcat is None:
            try:
                meshcat_params = MeshcatParams()
                # "0.0.0.0" is required for Docker port forwarding
                meshcat_params.host = os.environ.get("MESHCAT_HOST", "localhost")
                self.meshcat = Meshcat(meshcat_params)
                LOGGER.info("Meshcat available at: %s", self.meshcat.web_url())

                if self.meshcat:
                    if "MESHCAT_HOST" not in os.environ:
                        webbrowser.open(self.meshcat.web_url())
                    else:
                        LOGGER.info(
                            "Running in Docker/Headless mode; "
                            "skipping auto-browser open inside container."
                        )

            except (FileNotFoundError, PermissionError, OSError):
                LOGGER.exception("Failed to start Meshcat")
                self.meshcat = None

        # Build Diagram
        if self.current_urdf_path:
            self._build_custom_urdf_diagram(self.current_urdf_path)
        else:
            params = GolfModelParams()
            self.diagram, self.plant, _ = build_golf_swing_diagram(
                params, meshcat=self.meshcat
            )

        if self.diagram is None:
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

        if self.meshcat is not None:
            self.visualizer = None
        else:
            LOGGER.warning("Visualizer disabled due to Meshcat initialization failure.")

        # Create evaluation context for analysis
        self.eval_context = self.plant.CreateDefaultContext()

        # Init Manipulability
        if self.plant and DrakeManipulabilityAnalyzer is not None:
            self.manip_analyzer = DrakeManipulabilityAnalyzer(self.plant)
            self._populate_manip_checkboxes()

        # Initial State
        self._reset_state()

        # Refresh Recorder Engine ref if plant changed
        if hasattr(self, "recorder"):
            self.recorder.engine = self

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

        # Clear visualizations
        if self.meshcat:
            self.meshcat.Delete("overlays")

        # Reset recorder
        if hasattr(self, "recorder"):
            self.recorder.reset()
            if hasattr(self, "lbl_rec_status"):
                self.lbl_rec_status.setText("Frames: 0")
            if hasattr(self, "btn_record") and self.btn_record.isChecked():
                self.btn_record.setChecked(False)
                self.btn_record.setText("Record")

    def _on_model_changed(self, index: int) -> None:
        """Handle model change."""
        model_data = self.available_models[index]
        new_path = model_data["path"]

        if new_path != self.current_urdf_path:
            self.current_urdf_path = new_path

            # Re-initialize simulation
            self.timer.stop()
            try:
                self._init_simulation()
                self._build_kinematic_controls()
                self._sync_kinematic_sliders()
            except (RuntimeError, ValueError, OSError) as e:
                QtWidgets.QMessageBox.critical(self, "Error Loading Model", str(e))
                LOGGER.error("Error loading model: %s", e)
            finally:
                self.timer.start(int(self.time_step * MS_PER_SECOND))

    def _game_loop(self) -> None:
        simulator = self.simulator
        context = self.context

        if not simulator or not context:
            return

        # Always update Live Plot (even if paused, to redraw last frame/resize)
        if hasattr(self, "live_plot"):
            self.live_plot.update_plot()

        if self.operating_mode == "dynamic" and self.is_running:
            t = context.get_time()
            simulator.AdvanceTo(t + self.time_step)

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
                    body = self.plant.get_body(BodyIndex(self.plant.num_bodies() - 1))
                    X_WB = self.plant.EvalBodyPoseInWorld(plant_context, body)
                    club_pos = X_WB.translation()

                # Live Analysis if enabled OR if LivePlotWidget requested
                config_requests_analysis = False
                if hasattr(self.recorder, "analysis_config") and isinstance(
                    self.recorder.analysis_config, dict
                ):
                    cfg = self.recorder.analysis_config
                    if (
                        cfg.get("ztcf")
                        or cfg.get("zvcf")
                        or cfg.get("track_drift")
                        or cfg.get("track_total_control")
                        or cfg.get("induced_accel_sources")
                    ):
                        config_requests_analysis = True

                analysis_enabled = (
                    self.chk_live_analysis.isChecked() or config_requests_analysis
                )

                if analysis_enabled and self.eval_context:
                    self.plant.SetPositions(self.eval_context, q)
                    self.plant.SetVelocities(self.eval_context, v)

                    analyzer = DrakeInducedAccelerationAnalyzer(self.plant)

                    res = analyzer.compute_components(self.eval_context)

                    sources_to_compute = []

                    if self.chk_induced_vec.isChecked():
                        sources_to_compute.append(
                            self.combo_induced_source.currentText()
                        )

                    if hasattr(self.recorder, "analysis_config") and isinstance(
                        self.recorder.analysis_config, dict
                    ):
                        sources = self.recorder.analysis_config.get(
                            "induced_accel_sources", []
                        )
                        if isinstance(sources, list):
                            sources_to_compute.extend(sources)

                    unique_sources = set()
                    for src in sources_to_compute:
                        if src:
                            unique_sources.add(str(src))

                    for source in unique_sources:
                        if source in ["gravity", "velocity", "total"]:
                            continue

                        try:
                            act_idx = -1
                            try:
                                act_idx = int(source)
                            except ValueError:
                                if self.plant.HasJointNamed(source):
                                    joint = self.plant.GetJointByName(source)
                                    if joint.num_velocities() == 1:
                                        act_idx = joint.velocity_start()

                            if act_idx >= 0:
                                tau_vec = np.zeros(self.plant.num_velocities())
                                if 0 <= act_idx < len(tau_vec):
                                    tau_vec[act_idx] = 1.0
                                    accels = analyzer.compute_specific_control(
                                        self.eval_context, tau_vec
                                    )
                                    res[source] = accels
                        except (ValueError, TypeError, RuntimeError):
                            pass

                    for k, val in res.items():
                        if k not in self.recorder.induced_accelerations:
                            self.recorder.induced_accelerations[k] = []
                        self.recorder.induced_accelerations[k].append(val)

                    cf_res = analyzer.compute_counterfactuals(self.eval_context)
                    for k, val in cf_res.items():
                        if k not in self.recorder.counterfactuals:
                            self.recorder.counterfactuals[k] = []
                        self.recorder.counterfactuals[k].append(val)

                # Calculate CoM and Angular Momentum for recording
                com_pos = None
                angular_momentum = None
                if self.plant:
                    plant_context = self.plant.GetMyContextFromRoot(context)
                    com_pos = self.plant.CalcCenterOfMassPositionInWorld(plant_context)
                    angular_momentum = self.plant.CalcSpatialMomentumInWorldAboutPoint(
                        plant_context, com_pos
                    ).rotational()

                self.recorder.record(
                    context.get_time(),
                    q,
                    v,
                    club_pos,
                    com_pos=com_pos,
                    angular_momentum=angular_momentum,
                )
                self.lbl_rec_status.setText(f"Frames: {len(self.recorder.times)}")

        # Visualization Update (from DrakeVisualizationMixin)
        self._update_visualization()

    # ==================================================================
    # SimulationGUIBase overrides
    # ==================================================================

    def _build_base_ui(self) -> None:
        """Override base UI construction.

        Drake builds its own comprehensive UI in ``_setup_ui``,
        so we skip the generic skeleton.
        """
        # No-op: Drake builds its own UI entirely

    def step_simulation(self) -> None:
        """Advance the Drake simulation by one time step."""
        if self.simulator and self.context:
            t = self.context.get_time()
            self.simulator.AdvanceTo(t + self.time_step)

    def reset_simulation(self) -> None:
        """Reset the Drake simulation state."""
        self._reset_state()

    def update_visualization(self) -> None:
        """Refresh all Drake visualizations."""
        self._update_visualization()

    def load_model(self, index: int) -> None:
        """Load a model at the given index."""
        self._on_model_changed(index)

    def sync_kinematic_controls(self) -> None:
        """Synchronize kinematic slider values with model state."""
        self._sync_kinematic_sliders()

    def start_recording(self) -> None:
        """Start recording simulation data."""
        self.recorder.start()

    def stop_recording(self) -> None:
        """Stop recording simulation data."""
        self.recorder.stop()

    def get_recording_frame_count(self) -> int:
        """Return the number of recorded frames."""
        return len(self.recorder.times)

    def export_data(self, filename: str) -> None:
        """Export recorded data to the given filename."""
        self._export_data()


def main() -> None:
    setup_logging()
    app = QtWidgets.QApplication(sys.argv)
    window = DrakeSimApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
