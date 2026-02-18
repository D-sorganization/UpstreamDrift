"""Pinocchio GUI Wrapper (PyQt6 + meshcat).

The GUI functionality is decomposed into focused mixins:
- ``UISetupMixin`` (gui_ui_setup.py): Widget/layout construction, kinematic controls
- ``SimulationMixin`` (gui_simulation.py): Physics loop, recording, live analysis
- ``PinocchioAnalysisMixin`` (pinocchio_analysis_mixin.py): Plotting and data export
- ``PinocchioVisualizationMixin`` (pinocchio_visualization_mixin.py): Meshcat overlays

``PinocchioGUI`` inherits from all four mixins and ``SimulationGUIBase``,
acting as the coordinator class.
"""

import sys
from pathlib import Path

import numpy as np
import pinocchio as pin  # type: ignore
from PyQt6 import QtCore, QtWidgets

from src.shared.python.data_io.common_utils import get_shared_urdf_path
from src.shared.python.logging_pkg.logging_config import (
    configure_gui_logging,
    get_logger,
)
from src.shared.python.ui.simulation_gui_base import SimulationGUIBase
from src.shared.python.ui.widgets import LogPanel

# Mixin imports
from .gui_simulation import SimulationMixin
from .gui_ui_setup import UISetupMixin
from .manipulability import PinocchioManipulabilityAnalyzer
from .pinocchio_analysis_mixin import PinocchioAnalysisMixin
from .pinocchio_recorder import PinocchioRecorder
from .pinocchio_visualization_mixin import PinocchioVisualizationMixin

# Check meshcat availability
try:
    import meshcat.visualizer as viz

    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False
    viz = None  # type: ignore

if MESHCAT_AVAILABLE:
    from pinocchio.visualize import MeshcatVisualizer
else:
    MeshcatVisualizer = object  # Dummy class if missing

try:
    from .induced_acceleration import InducedAccelerationAnalyzer
except ImportError:
    # Fallback for when script is run directly
    from induced_acceleration import (  # type: ignore[no-redef]
        InducedAccelerationAnalyzer,
    )

# Set up logging using centralized module
configure_gui_logging()
logger = get_logger(__name__)


# Constants
DT_DEFAULT = (
    0.01  # [s] Physics time step. 10ms is standard for real-time visualization.
)
SLIDER_RANGE_RAD = 10.0  # [rad] Range for joint sliders provided in UI
SLIDER_SCALE = 100.0  # Scale factor for QSlider (int) -> rad (float)
COM_SPHERE_RADIUS = 0.02  # [m] Radius for Center of Mass visualization spheres
COM_COLOR = 0xFFFF00  # Yellow color for COMs


class PinocchioGUI(
    UISetupMixin,
    SimulationMixin,
    PinocchioAnalysisMixin,
    PinocchioVisualizationMixin,
    SimulationGUIBase,
):
    """Main GUI widget for Pinocchio robot visualization and computation."""

    WINDOW_TITLE = "Pinocchio Golf Model (Dynamics & Kinematics)"
    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 900

    def _init_internal_state(self) -> None:
        self.model: pin.Model | None = None
        self.data: pin.Data | None = None
        self.visual_model: pin.VisualModel | None = None
        self.collision_model: pin.CollisionModel | None = None
        self.viz: MeshcatVisualizer | None = None
        self.q: np.ndarray | None = None
        self.v: np.ndarray | None = None

        self.analyzer: InducedAccelerationAnalyzer | None = None
        self.latest_induced: dict[str, np.ndarray] | None = None
        self.latest_cf: dict[str, np.ndarray] | None = None

        self.manip_analyzer: PinocchioManipulabilityAnalyzer | None = None
        self.manip_checkboxes: dict[str, QtWidgets.QCheckBox] = {}

        self.recorder = PinocchioRecorder(engine=self)
        self.sim_time = 0.0

        self.joint_sliders: list[QtWidgets.QSlider] = []
        self.joint_spinboxes: list[QtWidgets.QDoubleSpinBox] = []
        self.joint_names: list[str] = []

        self.operating_mode = "dynamic"
        self.is_running = False
        self.dt = DT_DEFAULT

    def _init_meshcat_viewer(self) -> None:
        self.viewer: viz.Visualizer | None = None
        if not MESHCAT_AVAILABLE:
            self.log_write("Warning: Meshcat not available. Visualization disabled.")
            logger.warning("Meshcat module not found.")
            return

        try:
            try:
                self.viewer = viz.Visualizer(server_args=["--port", "7000"])
            except TypeError:
                logger.warning(
                    "Meshcat Visualizer: server_args not supported. Using default."
                )
                self.viewer = viz.Visualizer()

            url = self.viewer.url() if callable(self.viewer.url) else self.viewer.url
            logger.info("Internal Meshcat URL: %s", url)

            self._log_meshcat_url(url)
        except (ConnectionError, OSError, RuntimeError) as exc:
            logger.error(f"Failed to initialize Meshcat viewer: {exc}")
            self.log_write(f"Error: Failed to initialize Meshcat viewer: {exc}")
            self.log_write("Please ensure meshcat-server is running or try again.")

    def _log_meshcat_url(self, url: str) -> None:
        try:
            port = url.split(":")[-1].split("/")[0]
            host_url = f"http://127.0.0.1:{port}/static/"
            logger.info(f"Host Access URL: {host_url}")
            self.log_write("=" * 40)
            self.log_write("VISUALIZER READY")
            self.log_write("Open this URL in your browser:")
            self.log_write(f"{host_url}")
            self.log_write("=" * 40)
        except (PermissionError, OSError):
            logger.info("Could not determine host URL from: %s", url)

    def _load_default_model(self) -> None:
        default_urdf = (
            Path(__file__).parent / "../../models/generated/golfer.urdf"
        ).resolve()

        if default_urdf.exists():
            self.available_models.insert(
                0, {"name": "Default: Golfer", "path": str(default_urdf)}
            )
            self.load_urdf(str(default_urdf))
        else:
            self.available_models.insert(0, {"name": "Select Model...", "path": None})

    def __init__(self) -> None:
        """Initialize the Pinocchio GUI."""
        super().__init__()

        self._init_internal_state()

        pin_version = getattr(pin, "__version__", "unknown")
        logger.info(f"Pinocchio Version: {pin_version}")
        logger.info(f"Python Executable: {sys.executable}")

        self.log = LogPanel()

        self._init_meshcat_viewer()

        self.available_models: list[dict] = []
        self._scan_urdf_models()

        self._setup_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._game_loop)

        self._load_default_model()

    def get_joint_names(self) -> list[str]:
        """Return joint names for LivePlotWidget."""
        return self.joint_names

    def _scan_urdf_models(self) -> None:
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

    def log_write(self, text: str) -> None:
        """Append a message to the log panel and logger."""
        self.log.append(text)
        logger.info(text)

    def load_urdf(self, fname: str | None = None) -> None:
        """Load a URDF model and initialize the viewer."""
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
            # (Imports handled at module level)
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
                self.timer.start(int(self.dt * 1000))

            # Update Live Plot joint names if initialized
            if hasattr(self, "live_plot"):
                self.live_plot.set_joint_names(self.get_joint_names())

        except (ValueError, RuntimeError) as e:
            self.log_write(f"Error loading URDF (Pinocchio): {e}")
        except (PermissionError, OSError) as e:
            # Catch-all for unexpected errors
            self.log_write(f"Unexpected error loading URDF: {e}")
            logger.exception("Unexpected error loading URDF")

    # ==================================================================
    # SimulationGUIBase overrides
    # ==================================================================

    def _build_base_ui(self) -> None:
        """Override base UI construction.

        Pinocchio builds its own comprehensive UI in ``_setup_ui``,
        so we skip the generic skeleton.
        """
        # No-op: Pinocchio builds its own UI entirely

    def step_simulation(self) -> None:
        """Advance the Pinocchio simulation by one time step."""
        if (
            self.model is not None
            and self.data is not None
            and self.q is not None
            and self.v is not None
        ):
            tau = np.zeros(self.model.nv)
            a = pin.aba(self.model, self.data, self.q, self.v, tau)
            self.v += a * self.dt
            self.q = pin.integrate(self.model, self.q, self.v * self.dt)
            self.sim_time += self.dt

    def reset_simulation(self) -> None:
        """Reset the Pinocchio simulation state."""
        self._reset_simulation()

    def update_visualization(self) -> None:
        """Refresh the Pinocchio visualization."""
        self._update_viewer()

    def load_model(self, index: int) -> None:
        """Load a model at the given index."""
        self._on_model_combo_changed(index)

    def sync_kinematic_controls(self) -> None:
        """Synchronize kinematic slider values with model state."""
        self._sync_kinematic_controls()

    def start_recording(self) -> None:
        """Start recording simulation data."""
        self.recorder.start_recording()

    def stop_recording(self) -> None:
        """Stop recording simulation data."""
        self.recorder.stop_recording()

    def get_recording_frame_count(self) -> int:
        """Return the number of recorded frames."""
        return self.recorder.get_num_frames()

    def export_data(self, filename: str) -> None:
        """Export recorded data to the given filename."""
        self._export_statistics()


def main() -> None:
    """Main entry point for the GUI application."""
    app = QtWidgets.QApplication(sys.argv)
    window = PinocchioGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
