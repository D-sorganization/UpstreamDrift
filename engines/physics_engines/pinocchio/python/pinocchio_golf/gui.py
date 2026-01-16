"""Pinocchio GUI Wrapper (PyQt6 + meshcat)."""

import logging
import sys
import types
from pathlib import Path
from typing import Any

# Add suite root to sys.path to allow imports from shared.
# Instead of assuming a fixed directory depth, search upwards for a repository marker.
try:
    current_path = Path(__file__).resolve()
    suite_root: Path | None = None
    for parent in current_path.parents:
        if (parent / ".git").exists() or (parent / ".antigravityignore").exists():
            suite_root = parent
            break

    if suite_root and str(suite_root) not in sys.path:
        sys.path.insert(0, str(suite_root))
except Exception:
    # If detection fails, fall back to existing sys.path configuration or do nothing
    pass

import numpy as np
import pinocchio as pin  # type: ignore
from PyQt6 import QtCore, QtWidgets

from .manipulability import PinocchioManipulabilityAnalyzer

try:
    import meshcat.geometry as g
    import meshcat.visualizer as viz

    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False
    g = None  # type: ignore
    viz = None  # type: ignore

if MESHCAT_AVAILABLE:
    from pinocchio.visualize import MeshcatVisualizer
else:
    MeshcatVisualizer = object  # Dummy class if missing

from shared.python.biomechanics_data import BiomechanicalData
from shared.python.common_utils import get_shared_urdf_path
from shared.python.dashboard.widgets import LivePlotWidget
from shared.python.plotting import GolfSwingPlotter, MplCanvas
from shared.python.statistical_analysis import StatisticalAnalyzer

try:
    from .induced_acceleration import InducedAccelerationAnalyzer
except ImportError:
    # Fallback for when script is run directly
    from induced_acceleration import (  # type: ignore[no-redef]
        InducedAccelerationAnalyzer,
    )

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LogPanel(QtWidgets.QTextEdit):
    """Log panel widget for displaying messages."""

    def __init__(self) -> None:
        """Initialize the log panel."""
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet(
            "background:#111; color:#0F0; font-family:Consolas; font-size:12px;"
        )


# Constants
DT_DEFAULT = (
    0.01  # [s] Physics time step. 10ms is standard for real-time visualization.
)
SLIDER_RANGE_RAD = 10.0  # [rad] Range for joint sliders provided in UI
SLIDER_SCALE = 100.0  # Scale factor for QSlider (int) -> rad (float)
COM_SPHERE_RADIUS = 0.02  # [m] Radius for Center of Mass visualization spheres
COM_COLOR = 0xFFFF00  # Yellow color for COMs


class SignalBlocker:
    """Context manager to block signals for a set of widgets."""

    def __init__(self, *widgets: QtWidgets.QWidget) -> None:
        """Initialize with widgets to block."""
        self.widgets = widgets

    def __enter__(self) -> None:
        """Block signals for all widgets."""
        for w in self.widgets:
            w.blockSignals(True)  # noqa: FBT003

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """Restore signals for all widgets."""
        for w in self.widgets:
            w.blockSignals(False)  # noqa: FBT003


class PinocchioRecorder:
    """Records time-series data from Pinocchio simulation.

    Implements RecorderInterface for LivePlotWidget.
    """

    def __init__(self, engine: Any = None) -> None:
        """Initialize empty recorder."""
        self.reset()
        self.engine = engine  # Reference for joint names
        self.analysis_config: dict[str, Any] = {}

    def reset(self) -> None:
        """Clear all recorded data."""
        self.frames: list[BiomechanicalData] = []
        self.is_recording = False

    def start_recording(self) -> None:
        """Start recording data."""
        self.is_recording = True
        self.frames = []

    def stop_recording(self) -> None:
        """Stop recording data."""
        self.is_recording = False

    def get_num_frames(self) -> int:
        """Get number of recorded frames."""
        return len(self.frames)

    def record_frame(
        self,
        time: float,
        q: np.ndarray,
        v: np.ndarray,
        tau: np.ndarray | None = None,
        kinetic_energy: float = 0.0,
        potential_energy: float = 0.0,
        club_head_position: np.ndarray | None = None,
        club_head_velocity: np.ndarray | None = None,
        induced_accelerations: dict[str, np.ndarray] | None = None,
        counterfactuals: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Add a frame of data to the recording."""
        if self.is_recording:
            # Simple energy calculation if not provided (approximate)
            total_energy = kinetic_energy + potential_energy

            club_head_speed = 0.0
            if club_head_velocity is not None:
                club_head_speed = float(np.linalg.norm(club_head_velocity))

            frame = BiomechanicalData(
                time=float(time),
                joint_positions=q.copy(),
                joint_velocities=v.copy(),
                joint_torques=tau.copy() if tau is not None else np.zeros_like(v),
                kinetic_energy=kinetic_energy,
                potential_energy=potential_energy,
                total_energy=total_energy,
                club_head_position=club_head_position,
                club_head_velocity=club_head_velocity,
                club_head_speed=club_head_speed,
                induced_accelerations=induced_accelerations or {},
                counterfactuals=counterfactuals or {},
            )
            self.frames.append(frame)

    def set_analysis_config(self, config: dict[str, Any]) -> None:
        """Update analysis configuration."""
        self.analysis_config = config

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray | list]:
        """Extract time series for a specific field."""
        if not self.frames:
            return np.array([]), np.array([])

        times = np.array([f.time for f in self.frames])

        # Handle special counterfactual fields
        if field_name == "ztcf_accel":
            return self.get_counterfactual_series("ztcf_accel")
        if field_name == "zvcf_accel":
            # Assuming torque is stored here for now or adapt
            return self.get_counterfactual_series("zvcf_torque")

        values = [getattr(f, field_name, None) for f in self.frames]

        # Handle None values
        if all(v is None for v in values):
            return times, np.array([])

        # Filter out None values
        valid_indices = [i for i, v in enumerate(values) if v is not None]
        if not valid_indices:
            return times, np.array([])

        times = times[valid_indices]
        values = [values[i] for i in valid_indices]

        # Stack into array
        try:
            values_array = np.array(values)
        except (ValueError, TypeError):
            return times, values

        return times, values_array

    def get_induced_acceleration_series(
        self, source_name: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract time series for a specific induced acceleration source."""
        if not self.frames:
            return np.array([]), np.array([])

        times = []
        values = []

        # Map int index to key if possible (unlikely for Pinocchio basic)
        key = str(source_name)

        for f in self.frames:
            val = f.induced_accelerations.get(key)
            # Try int key if str fails
            if val is None and isinstance(source_name, int):
                # But dict keys are likely strings?
                # Let's check sim loop
                pass

            if val is not None:
                times.append(f.time)
                values.append(val)

        if not values:
            return np.array([]), np.array([])

        return np.array(times), np.array(values)

    def get_counterfactual_series(self, cf_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract time series for a specific counterfactual component."""
        if not self.frames:
            return np.array([]), np.array([])

        times = np.array([f.time for f in self.frames])

        valid_indices = [
            i
            for i, f in enumerate(self.frames)
            if hasattr(f, "counterfactuals") and cf_name in f.counterfactuals
        ]

        if not valid_indices:
            return np.array([]), np.array([])

        filtered_times = times[valid_indices]
        values = [self.frames[i].counterfactuals[cf_name] for i in valid_indices]

        return filtered_times, np.array(values)

    def export_to_dict(self) -> dict[str, Any]:
        """Export all recorded data to a dictionary."""
        if not self.frames:
            return {}

        export_data: dict[str, Any] = {}

        # Get basic time series
        times, positions = self.get_time_series("joint_positions")
        _, velocities = self.get_time_series("joint_velocities")
        _, torques = self.get_time_series("joint_torques")
        _, energies = self.get_time_series("total_energy")

        export_data["time"] = times
        export_data["joint_positions"] = positions
        export_data["joint_velocities"] = velocities
        export_data["joint_torques"] = torques
        export_data["total_energy"] = energies

        # Export Induced
        first_frame = self.frames[0]
        if first_frame.induced_accelerations:
            all_keys: set[str] = set()
            for f in self.frames:
                all_keys.update(f.induced_accelerations.keys())

            for key in all_keys:
                _, vals = self.get_induced_acceleration_series(key)
                if len(vals) > 0:
                    export_data[f"induced_{key}"] = vals

        # Export Counterfactuals
        if first_frame.counterfactuals:
            all_keys_cf: set[str] = set()
            for f in self.frames:
                all_keys_cf.update(f.counterfactuals.keys())

            for key in all_keys_cf:
                _, vals = self.get_counterfactual_series(key)
                if len(vals) > 0:
                    export_data[f"cf_{key}"] = vals

        return export_data


class PinocchioGUI(QtWidgets.QMainWindow):
    """Main GUI widget for Pinocchio robot visualization and computation."""

    def __init__(self) -> None:
        """Initialize the Pinocchio GUI."""
        super().__init__()
        self.setWindowTitle("Pinocchio Golf Model (Dynamics & Kinematics)")
        self.resize(1000, 900)  # Increased size for analysis tabs

        # Internal state
        self.model: pin.Model | None = None
        self.data: pin.Data | None = None
        self.visual_model: pin.VisualModel | None = None
        self.collision_model: pin.CollisionModel | None = None
        self.viz: MeshcatVisualizer | None = None
        self.q: np.ndarray | None = None
        self.v: np.ndarray | None = None

        # Analysis
        self.analyzer: InducedAccelerationAnalyzer | None = None
        self.latest_induced: dict[str, np.ndarray] | None = None
        self.latest_cf: dict[str, np.ndarray] | None = None

        # Manipulability
        self.manip_analyzer: PinocchioManipulabilityAnalyzer | None = None
        self.manip_checkboxes: dict[str, QtWidgets.QCheckBox] = {}

        # Recorder - pass self as engine
        self.recorder = PinocchioRecorder(engine=self)
        self.sim_time = 0.0

        self.joint_sliders: list[QtWidgets.QSlider] = []
        self.joint_spinboxes: list[QtWidgets.QDoubleSpinBox] = []
        self.joint_names: list[str] = []

        self.operating_mode = "dynamic"  # "dynamic", "kinematic"
        self.is_running = False
        self.dt = DT_DEFAULT

        # Diagnostics
        pin_version = getattr(pin, "__version__", "unknown")
        logger.info(f"Pinocchio Version: {pin_version}")
        logger.info(f"Python Executable: {sys.executable}")
        # Meshcat viewer
        self.viewer: viz.Visualizer | None = None
        if MESHCAT_AVAILABLE:
            try:
                # Force Meshcat to use port 7000 to match Docker exposure
                # and bind to 0.0.0.0 to allow external connections
                try:
                    self.viewer = viz.Visualizer(server_args=["--port", "7000"])
                except TypeError:
                    # Fallback for older meshcat versions that might not
                    # support server_args
                    logger.warning(
                        "Meshcat Visualizer: server_args not supported. Using default."
                    )
                    self.viewer = viz.Visualizer()

                if callable(self.viewer.url):
                    url = self.viewer.url()
                else:
                    url = self.viewer.url
                logger.info("Internal Meshcat URL: %s", url)

                # Explicitly log the external access URL for the user
                # We assume port 7000 based on our request (or fallback logic)
                try:
                    port = url.split(":")[-1].split("/")[0]
                    # Update to 7000 if we successfully requested it,
                    # or trust the return
                    host_url = f"http://127.0.0.1:{port}/static/"
                    logger.info(f"Host Access URL: {host_url}")
                    self.log_write("=" * 40)
                    self.log_write("VISUALIZER READY")
                    self.log_write("Open this URL in your browser:")
                    self.log_write(f"{host_url}")
                    self.log_write("=" * 40)
                except Exception:
                    logger.info("Could not determine host URL from: %s", url)
            except (ConnectionError, OSError, RuntimeError) as exc:
                logger.error(f"Failed to initialize Meshcat viewer: {exc}")
                self.log_write(f"Error: Failed to initialize Meshcat viewer: {exc}")
                self.log_write("Please ensure meshcat-server is running or try again.")
        else:
            self.log_write("Warning: Meshcat not available. Visualization disabled.")
            logger.warning("Meshcat module not found.")

        # Model Management
        self.available_models: list[dict] = []
        self._scan_urdf_models()

        # Setup UI
        self._setup_ui()

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._game_loop)

        # Try load default model
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
        except Exception as e:
            logger.error(f"Failed to scan URDF models: {e}")

    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # 1. Top Bar: Load & Mode
        top_layout = QtWidgets.QHBoxLayout()

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setMinimumWidth(200)
        self._populate_model_combo()
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        top_layout.addWidget(self.model_combo)

        self.load_btn = QtWidgets.QPushButton("Load File...")
        self.load_btn.clicked.connect(lambda: self.load_urdf())
        top_layout.addWidget(self.load_btn)

        top_layout.addStretch()

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Dynamic (Physics)", "Kinematic (Pose)"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        top_layout.addWidget(QtWidgets.QLabel("Mode:"))
        top_layout.addWidget(self.mode_combo)

        layout.addLayout(top_layout)

        # 2. Controls Stack (Main Tabs)
        self.main_tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.main_tabs)

        # Tab 1: Control & Simulation
        sim_tab = QtWidgets.QWidget()
        sim_layout = QtWidgets.QVBoxLayout(sim_tab)

        self.controls_stack = QtWidgets.QStackedWidget()
        sim_layout.addWidget(self.controls_stack)

        self._setup_dynamic_tab()
        self._setup_kinematic_tab()

        # Visuals & Logs in Sim Tab
        vis_group = QtWidgets.QGroupBox("Visualization")
        vis_layout = QtWidgets.QVBoxLayout()

        # Checkboxes row
        chk_layout = QtWidgets.QHBoxLayout()
        self.chk_frames = QtWidgets.QCheckBox("Show Frames")
        self.chk_frames.toggled.connect(self._toggle_frames)
        chk_layout.addWidget(self.chk_frames)

        self.chk_coms = QtWidgets.QCheckBox("Show COMs")
        self.chk_coms.toggled.connect(self._toggle_coms)
        chk_layout.addWidget(self.chk_coms)

        self.chk_forces = QtWidgets.QCheckBox("Show Forces")
        self.chk_forces.toggled.connect(self._toggle_forces)
        chk_layout.addWidget(self.chk_forces)

        self.chk_torques = QtWidgets.QCheckBox("Show Torques")
        self.chk_torques.toggled.connect(self._toggle_torques)
        chk_layout.addWidget(self.chk_torques)
        vis_layout.addLayout(chk_layout)

        # Ellipsoids & Body Selection
        ellip_group = QtWidgets.QGroupBox("Manipulability Analysis")
        ellip_layout = QtWidgets.QVBoxLayout()

        # Toggles
        toggles_layout = QtWidgets.QHBoxLayout()
        self.chk_mobility = QtWidgets.QCheckBox("Mobility (Green)")
        self.chk_mobility.toggled.connect(self._update_viewer)
        toggles_layout.addWidget(self.chk_mobility)

        self.chk_force_ellip = QtWidgets.QCheckBox("Force (Red)")
        self.chk_force_ellip.toggled.connect(self._update_viewer)
        toggles_layout.addWidget(self.chk_force_ellip)
        ellip_layout.addLayout(toggles_layout)

        # Body Grid
        self.manip_body_layout = QtWidgets.QGridLayout()
        body_container = QtWidgets.QWidget()
        body_container.setLayout(self.manip_body_layout)
        ellip_layout.addWidget(QtWidgets.QLabel("Points of Interest:"))
        ellip_layout.addWidget(body_container)

        ellip_group.setLayout(ellip_layout)
        vis_layout.addWidget(ellip_group)

        # Advanced Vectors
        adv_vec_layout = QtWidgets.QHBoxLayout()
        self.chk_induced = QtWidgets.QCheckBox("Induced Accel")
        self.chk_induced.toggled.connect(self._update_viewer)

        self.combo_induced = QtWidgets.QComboBox()
        self.combo_induced.setEditable(True)
        self.combo_induced.addItems(["gravity", "velocity", "total"])
        self.combo_induced.setToolTip(
            "Select source (e.g. gravity) or type "
            "specific torque vector in comma-sep form"
        )

        # Use lineEdit signal to avoid lag on keystrokes
        if line_edit := self.combo_induced.lineEdit():
            line_edit.editingFinished.connect(self._update_viewer)
        self.combo_induced.currentIndexChanged.connect(self._update_viewer)

        self.chk_cf = QtWidgets.QCheckBox("Counterfactuals")
        self.chk_cf.toggled.connect(self._update_viewer)

        self.combo_cf = QtWidgets.QComboBox()
        self.combo_cf.addItems(["ztcf_accel", "zvcf_torque"])
        self.combo_cf.currentTextChanged.connect(self._update_viewer)

        adv_vec_layout.addWidget(self.chk_induced)
        adv_vec_layout.addWidget(self.combo_induced)
        adv_vec_layout.addWidget(self.chk_cf)
        adv_vec_layout.addWidget(self.combo_cf)
        vis_layout.addLayout(adv_vec_layout)

        # Vector Scales
        scale_layout = QtWidgets.QHBoxLayout()
        self.spin_force_scale = QtWidgets.QDoubleSpinBox()
        self.spin_force_scale.setRange(0.01, 10.0)
        self.spin_force_scale.setSingleStep(0.05)
        self.spin_force_scale.setValue(0.1)
        self.spin_force_scale.setPrefix("Scale: ")
        self.spin_force_scale.valueChanged.connect(self._update_viewer)
        scale_layout.addWidget(self.spin_force_scale)

        self.spin_torque_scale = QtWidgets.QDoubleSpinBox()
        self.spin_torque_scale.setRange(0.01, 10.0)
        self.spin_torque_scale.setSingleStep(0.05)
        self.spin_torque_scale.setValue(0.1)
        self.spin_torque_scale.setPrefix("T Scale: ")
        self.spin_torque_scale.valueChanged.connect(self._update_viewer)
        scale_layout.addWidget(self.spin_torque_scale)
        vis_layout.addLayout(scale_layout)

        # Live Analysis Toggle
        self.chk_live_analysis = QtWidgets.QCheckBox("Live Analysis (Induced/CF)")
        self.chk_live_analysis.setToolTip(
            "Compute Induced Accelerations and Counterfactuals in real-time "
            "(Can slow down sim)"
        )
        self.chk_live_analysis.toggled.connect(self._on_live_analysis_toggled)
        vis_layout.addWidget(self.chk_live_analysis)

        vis_group.setLayout(vis_layout)
        sim_layout.addWidget(vis_group)

        # Matrix Analysis Panel
        matrix_group = QtWidgets.QGroupBox("Matrix Analysis")
        matrix_layout = QtWidgets.QFormLayout(matrix_group)
        self.lbl_cond = QtWidgets.QLabel("--")
        self.lbl_rank = QtWidgets.QLabel("--")
        matrix_layout.addRow("Jacobian Cond:", self.lbl_cond)
        matrix_layout.addRow("Mass Matrix Rank:", self.lbl_rank)
        sim_layout.addWidget(matrix_group)

        self.log = LogPanel()
        sim_layout.addWidget(self.log)

        self.main_tabs.addTab(sim_tab, "Simulation")

        # Tab 2: Live Analysis (LivePlotWidget)
        if LivePlotWidget is not None:
            self.live_tab = QtWidgets.QWidget()
            live_layout = QtWidgets.QVBoxLayout(self.live_tab)
            self.live_plot = LivePlotWidget(self.recorder)
            live_layout.addWidget(self.live_plot)
            self.main_tabs.addTab(self.live_tab, "Live Analysis")

        # Tab 3: Post-Hoc Analysis & Plotting
        self._setup_analysis_tab()

    def _on_live_analysis_toggled(self, checked: bool) -> None:
        """Handle live analysis toggle."""
        if checked:
            self.log_write("Live Analysis Enabled")
        else:
            self.log_write("Live Analysis Disabled")
            self.latest_induced = None
            self.latest_cf = None
            self._update_viewer()

    def _setup_analysis_tab(self) -> None:
        """Setup the analysis and plotting tab."""
        analysis_page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(analysis_page)

        # Controls
        controls = QtWidgets.QHBoxLayout()

        self.plot_combo = QtWidgets.QComboBox()
        self.plot_combo.addItems(
            [
                "Dashboard",
                "Joint Angles",
                "Joint Velocities",
                "Joint Torques",
                "Energy Analysis",
                "Kinematic Sequence",
                "Phase Diagram",
                "Frequency Analysis (PSD)",
                "Correlation Matrix",
                "Induced Accelerations",
                "Counterfactuals (ZTCF/ZVCF)",
                "Swing Profile (Radar)",
                "Power Flow",
            ]
        )

        self.joint_select_combo = QtWidgets.QComboBox()
        # Will be populated when model loads
        self.joint_select_combo.setMinimumWidth(120)

        # We reuse combo_induced logic, no separate edit box needed here for live vis

        controls.addWidget(QtWidgets.QLabel("Joint:"))
        controls.addWidget(self.joint_select_combo)
        controls.addWidget(QtWidgets.QLabel("Plot Type:"))
        controls.addWidget(self.plot_combo)

        self.btn_plot = QtWidgets.QPushButton("Generate Plot")
        self.btn_plot.clicked.connect(self._generate_plot)
        controls.addWidget(self.btn_plot)

        self.btn_export_csv = QtWidgets.QPushButton("Export CSV")
        self.btn_export_csv.clicked.connect(self._export_statistics)
        controls.addWidget(self.btn_export_csv)

        controls.addStretch()
        layout.addLayout(controls)

        # Canvas
        try:
            self.canvas = MplCanvas(width=5, height=4, dpi=100)
            layout.addWidget(self.canvas)
        except RuntimeError:
            self.canvas = None  # type: ignore[assignment]
            layout.addWidget(QtWidgets.QLabel("Plotting requires GUI environment"))

        self.main_tabs.addTab(analysis_page, "Post-Hoc Analysis")

    def _generate_plot(self) -> None:
        """Generate the selected plot."""
        if self.canvas is None:
            return

        if self.recorder.get_num_frames() == 0:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "No simulation data recorded yet."
            )
            return

        self.canvas.fig.clear()

        # Initialize plotter
        plotter = GolfSwingPlotter(self.recorder, self.joint_names)

        plot_type = self.plot_combo.currentText()

        if plot_type == "Dashboard":
            plotter.plot_summary_dashboard(self.canvas.fig)
        elif plot_type == "Joint Angles":
            plotter.plot_joint_angles(self.canvas.fig)
        elif plot_type == "Joint Velocities":
            plotter.plot_joint_velocities(self.canvas.fig)
        elif plot_type == "Joint Torques":
            plotter.plot_joint_torques(self.canvas.fig)
        elif plot_type == "Energy Analysis":
            plotter.plot_energy_analysis(self.canvas.fig)
        elif plot_type == "Kinematic Sequence":
            # Map all joints for now
            segments = {name: i for i, name in enumerate(self.joint_names)}
            plotter.plot_kinematic_sequence(self.canvas.fig, segments)
        elif plot_type == "Phase Diagram":
            plotter.plot_phase_diagram(
                self.canvas.fig, joint_idx=0
            )  # First joint by default
        elif plot_type == "Frequency Analysis (PSD)":
            plotter.plot_frequency_analysis(self.canvas.fig, joint_idx=0)
        elif plot_type == "Correlation Matrix":
            plotter.plot_correlation_matrix(self.canvas.fig)
        elif plot_type == "Induced Accelerations":
            self._plot_induced_accelerations()
        elif plot_type == "Counterfactuals (ZTCF/ZVCF)":
            self._plot_counterfactuals()
        elif plot_type == "Swing Profile (Radar)":
            self._plot_swing_profile(plotter)
        elif plot_type == "Power Flow":
            # Requires power data in recorder
            if any(f.actuator_powers.size > 0 for f in self.recorder.frames):
                plotter.plot_power_flow(self.canvas.fig)
            else:
                ax = self.canvas.fig.add_subplot(111)
                ax.text(0.5, 0.5, "No actuator power data", ha="center", va="center")

        self.canvas.draw()

    def _plot_swing_profile(self, plotter: GolfSwingPlotter) -> None:
        """Plot the Swing Profile radar chart."""
        # Calculate metrics using StatisticalAnalyzer
        times, positions = self.recorder.get_time_series("joint_positions")
        _, velocities = self.recorder.get_time_series("joint_velocities")
        _, torques = self.recorder.get_time_series("joint_torques")
        _, club_speed = self.recorder.get_time_series("club_head_speed")

        # Need to ensure types are correct
        positions = np.asarray(positions)
        velocities = np.asarray(velocities)
        torques = np.asarray(torques)
        club_speed = np.asarray(club_speed)

        analyzer = StatisticalAnalyzer(
            times, positions, velocities, torques, club_head_speed=club_speed
        )
        report = analyzer.generate_comprehensive_report()

        metrics = {
            "Speed": 0.0,
            "Efficiency": 0.0,
            "Tempo": 0.0,
            "Stability": 0.0,  # Placeholder
            "Power": 0.0,
        }

        # Populate real metrics where possible
        if "club_head_speed" in report:
            peak = report["club_head_speed"]["peak_value"]
            # Normalize: assume 50 m/s is pro level
            metrics["Speed"] = min(peak / 50.0, 1.0)

        if "tempo" in report:
            ratio = report["tempo"]["ratio"]
            # Ideal 3.0.
            err = abs(ratio - 3.0)
            metrics["Tempo"] = max(0.0, 1.0 - (err / 2.0))

        if "energy_efficiency" in report:
            metrics["Efficiency"] = report["energy_efficiency"] / 100.0

        plotter.plot_radar_chart(self.canvas.fig, metrics)

    def _ensure_analyzer_initialized(self) -> None:
        """Ensure the InducedAccelerationAnalyzer is initialized."""
        if self.analyzer is None and self.model is not None:
            self.analyzer = InducedAccelerationAnalyzer(self.model, self.data)

    def _plot_induced_accelerations(self) -> None:
        """Calculate and plot induced accelerations for selected joint."""
        # Use updated GolfSwingPlotter logic
        if not self.recorder.frames:
            return

        # Get selected joint
        joint_name = self.joint_select_combo.currentText()
        if not joint_name:
            if self.joint_names:
                joint_name = self.joint_names[0]
            else:
                return

        # Get velocity index for plotting
        try:
            if self.model is None:
                return
            joint_idx = list(self.model.names).index(joint_name)
            v_idx = self.model.joints[joint_idx].idx_v
        except ValueError:
            return

        # Populate recorder with post-hoc data if live analysis was off.
        # This can be expensive on large datasets, so only do it once per
        # GUI instance unless explicitly reset elsewhere.
        if not getattr(self, "_analysis_data_populated", False):
            self._ensure_analysis_data_populated()
            self._analysis_data_populated = True

        # Check if 'specific_control' is already in frames (from live recording)
        has_specific = any(
            "specific_control" in f.induced_accelerations for f in self.recorder.frames
        )

        # If not in frames, but we have text in combo box, compute it post-hoc
        txt = self.combo_induced.currentText()
        if (
            not has_specific
            and txt
            and txt not in ["gravity", "velocity", "total"]
            and self.analyzer
        ):
            try:
                parts = [float(x) for x in txt.split(",")]
                if len(parts) == self.model.nv:
                    spec_tau = np.array(parts)
                    # Compute for all frames
                    QtWidgets.QApplication.setOverrideCursor(
                        QtCore.Qt.CursorShape.WaitCursor
                    )
                    for frame in self.recorder.frames:
                        if frame.joint_positions is not None:
                            a_spec = self.analyzer.compute_specific_control(
                                frame.joint_positions, spec_tau
                            )
                            frame.induced_accelerations["specific_control"] = a_spec
                    QtWidgets.QApplication.restoreOverrideCursor()
                    has_specific = True
            except ValueError:
                pass

        plotter = GolfSwingPlotter(self.recorder, self.joint_names)

        plotter.plot_induced_acceleration(
            self.canvas.fig, "breakdown", joint_idx=v_idx, breakdown_mode=True
        )

        # Manually add specific control trace if it exists
        if has_specific:
            times, spec_vals = self.recorder.get_induced_acceleration_series(
                "specific_control"
            )
            if len(times) > 0 and spec_vals.size > 0:
                ax = self.canvas.fig.axes[0]
                if v_idx < spec_vals.shape[1]:
                    ax.plot(
                        times,
                        spec_vals[:, v_idx],
                        label="Specific Source",
                        color="magenta",
                        linewidth=2,
                        linestyle=":",
                    )
                    ax.legend()

    def _plot_counterfactuals(self) -> None:
        """Plot ZTCF (Zero Torque Accel) and ZVCF (Zero Velocity Torque)."""
        if not self.recorder.frames:
            return

        joint_name = self.joint_select_combo.currentText()
        if not joint_name:
            if self.joint_names:
                joint_name = self.joint_names[0]
            else:
                return

        try:
            if self.model is None:
                return
            joint_idx = list(self.model.names).index(joint_name)
            v_idx = self.model.joints[joint_idx].idx_v
        except ValueError:
            return

        self._ensure_analysis_data_populated()

        plotter = GolfSwingPlotter(self.recorder, self.joint_names)
        plotter.plot_counterfactual_comparison(
            self.canvas.fig, "dual", metric_idx=v_idx
        )

    def _ensure_analysis_data_populated(self) -> None:
        """Populate recorder frames with analysis data if missing."""
        if not self.recorder.frames:
            return

        # Check first frame
        if (
            self.recorder.frames[0].induced_accelerations
            and self.recorder.frames[0].counterfactuals
        ):
            return  # Already populated

        self._ensure_analyzer_initialized()
        if self.analyzer is None:
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            for frame in self.recorder.frames:
                if not frame.induced_accelerations:
                    frame.induced_accelerations = self.analyzer.compute_components(
                        frame.joint_positions,
                        frame.joint_velocities,
                        frame.joint_torques,
                    )
                if not frame.counterfactuals:
                    if hasattr(self.analyzer, "compute_counterfactuals"):
                        frame.counterfactuals = self.analyzer.compute_counterfactuals(
                            frame.joint_positions, frame.joint_velocities
                        )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _export_statistics(self) -> None:
        """Export recorded data to multiple formats."""
        if self.recorder.get_num_frames() == 0:
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Data", "pinocchio_data", "All Files (*)"
        )
        if not filename:
            return

        # Ensure advanced metrics are computed
        self._ensure_analysis_data_populated()

        try:
            from shared.python.export import export_recording_all_formats

            data_dict = self.recorder.export_to_dict()
            results = export_recording_all_formats(filename, data_dict)

            msg = "Export Results:\n"
            for fmt, success in results.items():
                msg += f"{fmt}: {'Success' if success else 'Failed'}\n"

            QtWidgets.QMessageBox.information(self, "Export Complete", msg)
            self.log_write(f"Data exported to {filename}")

        except Exception as e:
            self.log_write(f"Error exporting data: {e}")
            logger.exception("Export failed")

    def _populate_model_combo(self) -> None:
        """Populate the model dropdown."""
        self.model_combo.clear()
        for model in self.available_models:
            self.model_combo.addItem(model["name"])

    def _on_model_combo_changed(self, index: int) -> None:
        """Handle model selection."""
        if index < 0 or index >= len(self.available_models):
            return

        path = self.available_models[index]["path"]
        if path:
            self.load_urdf(path)

    def log_write(self, text: str) -> None:
        self.log.append(text)
        logger.info(text)

    def _setup_dynamic_tab(self) -> None:
        dyn_page = QtWidgets.QWidget()
        dyn_layout = QtWidgets.QVBoxLayout(dyn_page)

        # Run Controls
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run Simulation")
        self.btn_run.setCheckable(True)
        self.btn_run.clicked.connect(self._toggle_run)
        btn_layout.addWidget(self.btn_run)

        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset_simulation)
        btn_layout.addWidget(self.btn_reset)
        dyn_layout.addLayout(btn_layout)

        # Recording Controls
        rec_layout = QtWidgets.QHBoxLayout()
        self.btn_record = QtWidgets.QPushButton("Record")
        self.btn_record.setCheckable(True)
        self.btn_record.setStyleSheet(
            "QPushButton:checked { background-color: #ffcccc; }"
        )
        self.btn_record.clicked.connect(self._toggle_recording)
        rec_layout.addWidget(self.btn_record)

        self.lbl_rec_status = QtWidgets.QLabel("Frames: 0")
        rec_layout.addWidget(self.lbl_rec_status)
        dyn_layout.addLayout(rec_layout)

        dyn_layout.addStretch()
        self.controls_stack.addWidget(dyn_page)

    def _toggle_recording(self) -> None:
        """Toggle recording state."""
        if self.btn_record.isChecked():
            self.recorder.start_recording()
            self.log_write("Recording started.")
            self.btn_record.setText("Stop Recording")
        else:
            self.recorder.stop_recording()
            self.log_write(
                f"Recording stopped. Frames: {self.recorder.get_num_frames()}"
            )
            self.btn_record.setText("Record")

    def _setup_kinematic_tab(self) -> None:
        kin_page = QtWidgets.QWidget()
        kin_layout = QtWidgets.QVBoxLayout(kin_page)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.slider_container = QtWidgets.QWidget()
        self.slider_layout = QtWidgets.QVBoxLayout(self.slider_container)
        scroll.setWidget(self.slider_container)

        kin_layout.addWidget(scroll)
        self.controls_stack.addWidget(kin_page)

    def load_urdf(self, fname: str | None = None) -> None:
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
            except Exception as e:
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
            self._populate_manip_checkboxes()

            # Reset recorder
            self.recorder.reset()
            self.lbl_rec_status.setText("Frames: 0")
            if self.btn_record.isChecked():
                self.btn_record.setChecked(False)
                self.btn_record.setText("Record")

            # Initialize Pinocchio MeshcatVisualizer
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
                except Exception as e:
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
        except Exception as e:
            # Catch-all for unexpected errors
            self.log_write(f"Unexpected error loading URDF: {e}")
            logger.exception("Unexpected error loading URDF")

    def _build_kinematic_controls(self) -> None:
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
        self.joint_names = []

        # Populate joint_names for joints 1..N (excluding Universe at index 0).
        self.joint_names = list(self.model.names)[1:]

        # Update joint selection combo for analysis
        self.joint_select_combo.clear()
        self.joint_select_combo.addItems(self.joint_names)

        # Iterate joints (skip universe)
        for i in range(1, self.model.njoints):
            self._add_joint_control_widget(i)

    def _add_joint_control_widget(self, i: int) -> None:
        if self.model is None:
            return

        joint_name = self.model.names[i]
        # Simple assumption: 1 DOF per joint for sliders.
        nq_joint = self.model.joints[i].nq

        if nq_joint != 1:
            msg = (
                f"Skipping joint '{joint_name}' (index {i}): "
                f"{nq_joint} DOFs not supported in kinematic controls."
            )
            self.log_write(msg)
            return

        # joint_names is pre-populated above; widgets are only created for supported
        # 1-DOF joints
        row = QtWidgets.QWidget()
        r_layout = QtWidgets.QHBoxLayout(row)
        r_layout.setContentsMargins(0, 0, 0, 0)

        r_layout.addWidget(QtWidgets.QLabel(f"{joint_name}:"))

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        # Range +/- SLIDER_RANGE_RAD
        slider_min = int(-SLIDER_RANGE_RAD * SLIDER_SCALE)
        slider_max = int(SLIDER_RANGE_RAD * SLIDER_SCALE)
        slider.setRange(slider_min, slider_max)
        slider.setValue(0)

        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-SLIDER_RANGE_RAD, SLIDER_RANGE_RAD)
        spin.setSingleStep(0.1)

        # Connect
        idx_q = self.model.joints[i].idx_q
        idx = int(idx_q)  # Capture index into q vector

        slider.valueChanged.connect(
            lambda val, s=spin, k=idx: self._on_slider(val, s, k)
        )
        spin.valueChanged.connect(lambda val, s=slider, k=idx: self._on_spin(val, s, k))

        r_layout.addWidget(slider)
        r_layout.addWidget(spin)
        self.slider_layout.addWidget(row)

        self.joint_sliders.append(slider)
        self.joint_spinboxes.append(spin)

    def _sync_kinematic_controls(self) -> None:
        """Synchronize sliders/spinboxes with current model state q."""
        if self.model is None or self.q is None:
            return

        slider_idx = 0
        for i in range(1, self.model.njoints):
            # Must match the filtering in _build_kinematic_controls
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

    def _on_slider(self, val: int, spin: QtWidgets.QDoubleSpinBox, idx: int) -> None:
        angle = val / SLIDER_SCALE
        with SignalBlocker(spin):
            spin.setValue(angle)
        self._update_q(idx, angle)

    def _on_spin(self, val: float, slider: QtWidgets.QSlider, idx: int) -> None:
        with SignalBlocker(slider):
            slider.setValue(int(val * SLIDER_SCALE))
        self._update_q(idx, val)

    def _update_q(self, idx: int, val: float) -> None:
        if self.operating_mode != "kinematic":
            return
        if self.q is not None:
            self.q[idx] = val
            self._update_viewer()

    def _on_mode_changed(self, mode_text: str) -> None:
        if "Dynamic" in mode_text:
            self.operating_mode = "dynamic"
            self.controls_stack.setCurrentIndex(0)
        else:
            self.operating_mode = "kinematic"
            self.controls_stack.setCurrentIndex(1)
            # Stop simulation when entering kinematic mode
            self.is_running = False
            self.btn_run.setText("Run Simulation")
            self.btn_run.setChecked(False)
            self._sync_kinematic_controls()

    def _toggle_run(self) -> None:
        self.is_running = not self.is_running
        self.btn_run.setText(
            "Pause Simulation" if self.is_running else "Run Simulation"
        )
        self.btn_run.setChecked(self.is_running)

    def _reset_simulation(self) -> None:
        if self.model is None:
            return
        self.q = pin.neutral(self.model)
        self.v = np.zeros(self.model.nv)
        self.is_running = False
        self.sim_time = 0.0
        self.btn_run.setText("Run Simulation")
        self.btn_run.setChecked(False)
        self._update_viewer()
        self._sync_kinematic_controls()

        # Reset recording
        self.recorder.reset()
        self.lbl_rec_status.setText("Frames: 0")
        if self.btn_record.isChecked():
            self.btn_record.setChecked(False)
            self.btn_record.setText("Record")

    def _game_loop(self) -> None:
        if self.model is None or self.data is None or self.q is None or self.v is None:
            return

        # Always update Live Plot (even if paused, to redraw last frame/resize)
        if hasattr(self, "live_plot"):
            self.live_plot.update_plot()

        if self.operating_mode == "dynamic" and self.is_running:
            # --- Physics integration loop ---
            tau = np.zeros(self.model.nv)
            a = pin.aba(self.model, self.data, self.q, self.v, tau)
            self.v += a * self.dt
            self.q = pin.integrate(self.model, self.q, self.v * self.dt)
            self.sim_time += self.dt

            # Recording
            if self.recorder.is_recording:
                # Compute energies for recording
                pin.computeKineticEnergy(self.model, self.data, self.q, self.v)
                pin.computePotentialEnergy(self.model, self.data, self.q)

                # Capture club head data if available
                club_head_pos = None
                club_head_vel = None

                club_id = -1
                for fid in range(self.model.nframes):
                    name = self.model.frames[fid].name.lower()
                    if "club" in name or "head" in name:
                        club_id = fid
                        break

                if club_id == -1 and self.model.nframes > 0:
                    club_id = self.model.nframes - 1

                if club_id >= 0:
                    pin.forwardKinematics(self.model, self.data, self.q, self.v)
                    pin.updateFramePlacements(self.model, self.data)

                    frame = self.data.oMf[club_id]
                    club_head_pos = frame.translation.copy()

                    v_frame = pin.getFrameVelocity(
                        self.model,
                        self.data,
                        club_id,
                        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                    )
                    club_head_vel = v_frame.linear.copy()

                q_for_recording = self.q if self.q is not None else np.array([])

                # Induced / Counterfactuals
                induced = None
                counterfactuals = None

                # Check "Live Analysis" toggle
                if self.chk_live_analysis.isChecked():
                    if self.analyzer and self.q is not None and self.v is not None:
                        induced = self.analyzer.compute_components(self.q, self.v, tau)
                        self.latest_induced = induced

                        # Check for specific torque override from UI or Config
                        sources_to_compute = []
                        txt = self.combo_induced.currentText()
                        if txt:
                            sources_to_compute.append(txt)

                        # From config
                        if hasattr(self.recorder, 'analysis_config') and isinstance(self.recorder.analysis_config, dict):
                            sources = self.recorder.analysis_config.get('induced_accel_sources', [])
                            if isinstance(sources, list):
                                sources_to_compute.extend(sources)

                        unique_sources = set()
                        for s in sources_to_compute:
                            if s: unique_sources.add(str(s))

                        for src in unique_sources:
                            if src in ["gravity", "velocity", "total"]: continue

                            # Attempt to parse as comma-separated vector OR Joint Name
                            spec_tau = np.zeros(self.model.nv)
                            found_joint = False

                            # Check if it's a joint name
                            if self.model.existJointName(src):
                                j_id = self.model.getJointId(src)
                                joint = self.model.joints[j_id]
                                if joint.nv == 1:
                                    spec_tau[joint.idx_v] = 1.0
                                    found_joint = True

                            # Check if it's an int index
                            if not found_joint:
                                try:
                                    act_idx = int(src)
                                    if 0 <= act_idx < self.model.nv:
                                        spec_tau[act_idx] = 1.0
                                        found_joint = True
                                except ValueError:
                                    pass

                            if not found_joint:
                                try:
                                    parts = [float(x) for x in src.split(",")]
                                    if len(parts) == self.model.nv:
                                        spec_tau = np.array(parts)
                                        found_joint = True
                                except ValueError:
                                    pass

                            if found_joint:
                                spec_acc = self.analyzer.compute_specific_control(
                                    self.q, spec_tau
                                )
                                # Store result using source string as key
                                induced[src] = spec_acc

                        if hasattr(self.analyzer, "compute_counterfactuals"):
                            counterfactuals = self.analyzer.compute_counterfactuals(
                                self.q, self.v
                            )
                            self.latest_cf = counterfactuals

                self.recorder.record_frame(
                    time=self.sim_time,
                    q=q_for_recording,
                    v=self.v,
                    tau=tau,
                    kinetic_energy=self.data.kinetic_energy,
                    potential_energy=self.data.potential_energy,
                    club_head_position=club_head_pos,
                    club_head_velocity=club_head_vel,
                    induced_accelerations=induced,
                    counterfactuals=counterfactuals,
                )
                self.lbl_rec_status.setText(f"Frames: {self.recorder.get_num_frames()}")

            self._update_viewer()

    def _update_viewer(self) -> None:
        if (
            self.model is None
            or self.data is None
            or self.q is None
            or self.viz is None
        ):
            return

        # Update Visuals via Pinocchio Visualizer
        self.viz.display(self.q)

        # Kinematics Logic for frames (needed for custom overlays)
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

        # Calculate matrices for analysis
        self._compute_analysis()

        # Overlays
        if self.chk_frames.isChecked():
            self._draw_frames()
        if self.chk_coms.isChecked():
            self._draw_coms()
        if self.chk_forces.isChecked() or self.chk_torques.isChecked():
            self._draw_vectors()

        if self.chk_induced.isChecked():
            self._draw_induced_vectors()
        if self.chk_cf.isChecked():
            self._draw_cf_vectors()

        if self.chk_mobility.isChecked() or self.chk_force_ellip.isChecked():
            self._draw_ellipsoids()
        else:
            if self.viewer:
                self.viewer["overlays/ellipsoids"].delete()

    def _compute_analysis(self) -> None:
        """Compute Jacobian and Mass matrix analysis."""
        if self.model is None or self.data is None or self.q is None:
            return

        joint_id = self.model.njoints - 1
        pin.computeJointJacobians(self.model, self.data, self.q)
        J = pin.getJointJacobian(
            self.model, self.data, joint_id, pin.ReferenceFrame.LOCAL
        )

        try:
            s = np.linalg.svd(J, compute_uv=False)
            cond = s[0] / s[-1] if s[-1] > 1e-9 else float("inf")
            self.lbl_cond.setText(f"{cond:.2f}")
        except Exception:
            self.lbl_cond.setText("Error")

        M = pin.crba(self.model, self.data, self.q)
        try:
            rank = np.linalg.matrix_rank(M)
            self.lbl_rank.setText(f"{rank} / {self.model.nv}")
        except Exception:
            self.lbl_rank.setText("Error")

    def _draw_ellipsoids(self) -> None:
        """Draw mobility/force ellipsoids for selected bodies."""
        if (
            self.model is None
            or self.data is None
            or self.viewer is None
            or self.manip_analyzer is None
        ):
            return

        # Clear previous ellipsoids to prevent ghosting
        try:
            self.viewer["overlays/ellipsoids"].delete()
        except Exception:
            pass

        if self.chk_mobility.isChecked() or self.chk_force_ellip.isChecked():
            # Get selected bodies
            selected_bodies = [
                name for name, chk in self.manip_checkboxes.items() if chk.isChecked()
            ]

            # Compute for all selected
            if selected_bodies:
                # Iterate individually as compute_metrics takes a single body name
                for body_name in selected_bodies:
                    res = self.manip_analyzer.compute_metrics(body_name, self.q)
                    if not res:
                        continue

                    pos = res.velocity_ellipsoid.center

                    if (
                        self.chk_mobility.isChecked()
                        and res.mobility_matrix is not None
                    ):
                        path_name = f"{res.body_name}/mobility"

                        radii = res.velocity_ellipsoid.radii
                        # Scale down for viz (metrics are usually large)
                        self._draw_ellipsoid_meshcat(
                            path_name,
                            pos,
                            res.velocity_ellipsoid.axes,
                            radii * 0.5,
                            0x00FF00,
                        )

                    if (
                        self.chk_force_ellip.isChecked()
                        and res.force_matrix is not None
                    ):
                        path_name = f"{res.body_name}/force"

                        radii = res.force_ellipsoid.radii
                        # Scale for viz
                        self._draw_ellipsoid_meshcat(
                            path_name,
                            pos,
                            res.force_ellipsoid.axes,
                            radii * 0.2,
                            0xFF0000,
                        )

    def _draw_ellipsoid_meshcat(
        self,
        name: str,
        pos: np.ndarray,
        rot: np.ndarray,
        radii: np.ndarray,
        color: int,
    ) -> None:
        """Draw ellipsoid using Meshcat."""
        if self.viewer is None:
            return

        path = f"overlays/ellipsoids/{name}"

        self.viewer[path].set_object(
            g.Sphere(1.0),
            g.MeshLambertMaterial(color=color, opacity=0.5, transparent=True),
        )

        T = np.eye(4)
        T[:3, :3] = rot @ np.diag(radii)
        T[:3, 3] = pos

        self.viewer[path].set_transform(T)

    def _draw_vectors(self) -> None:
        """Draw force and torque vectors at joints."""
        if self.model is None or self.data is None or self.viewer is None:
            return

        v = self.v if self.v is not None else np.zeros(self.model.nv)
        a = pin.aba(self.model, self.data, self.q, v, np.zeros(self.model.nv))

        pin.rnea(self.model, self.data, self.q, v, a)

        force_scale = self.spin_force_scale.value()
        torque_scale = self.spin_torque_scale.value()

        for i in range(1, self.model.njoints):
            joint_placement = self.data.oMi[i]
            f_local = self.data.f[i]

            f_world = joint_placement.rotation @ f_local.linear
            t_world = joint_placement.rotation @ f_local.angular

            joint_name = self.model.names[i]

            if self.chk_forces.isChecked() and np.linalg.norm(f_world) > 1e-3:
                self._draw_arrow(
                    f"overlays/forces/{joint_name}",
                    joint_placement.translation,
                    f_world * force_scale,
                    0xFF0000,
                )

            if self.chk_torques.isChecked() and np.linalg.norm(t_world) > 1e-3:
                self._draw_arrow(
                    f"overlays/torques/{joint_name}",
                    joint_placement.translation,
                    t_world * torque_scale,
                    0x0000FF,
                )

    def _draw_induced_vectors(self) -> None:
        """Draw induced acceleration vectors."""
        if (
            self.model is None
            or self.data is None
            or self.viewer is None
            or self.latest_induced is None
        ):
            return

        source = self.combo_induced.currentText()
        accels = np.zeros(self.model.nv)

        if source in ["gravity", "velocity", "total"]:
            if source in self.latest_induced:
                accels = self.latest_induced[source]
        else:
            # Maybe it's a specific torque from combo box text?
            # We don't have it pre-calculated in 'latest_induced' from the loop
            # unless we add logic there.
            # But we can calculate it on fly for visualization if q is current.
            # (Loop updated to push to latest_induced, so this might be covered)
            if source in self.latest_induced:
                accels = self.latest_induced[source]
            else:
                txt = source
                if txt and self.analyzer and self.q is not None:
                    try:
                        parts = [float(x) for x in txt.split(",")]
                        tau = np.zeros(self.model.nv)
                        min_len = min(len(parts), len(tau))
                        tau[:min_len] = parts[:min_len]
                        accels = self.analyzer.compute_specific_control(self.q, tau)
                    except ValueError:
                        pass

        scale = self.spin_torque_scale.value()  # Use torque scale for now

        for i in range(1, self.model.njoints):
            joint = self.model.joints[i]
            idx_v = joint.idx_v
            nv = joint.nv
            if nv != 1:
                continue

            # Get acceleration scalar
            alpha = accels[idx_v]
            if abs(alpha) < 1e-3:
                continue

            # Get joint axis in world frame
            # Transform motion subspace to world
            oMi = self.data.oMi[i]
            # Spatial motion vector in local frame
            # For Revolute, it is angular around axis
            # S is 6x1.
            S = joint.S
            # Spatial accel in local frame
            a_local = S * alpha
            # Transform to world
            # Action matrix: [R 0; [p]xR R]
            # Actually se3 action.
            a_world = oMi.act(a_local)

            # Draw angular part (top 3)
            # Or linear part (bottom 3)
            # For revolute, angular is primary.
            vec = a_world.angular
            if np.linalg.norm(vec) < 1e-6:
                vec = a_world.linear

            self._draw_arrow(
                f"overlays/induced/{self.model.names[i]}",
                oMi.translation,
                vec * scale,
                0xFF00FF,  # Magenta
            )

    def _draw_cf_vectors(self) -> None:
        """Draw Counterfactual vectors."""
        if (
            self.model is None
            or self.data is None
            or self.viewer is None
            or self.latest_cf is None
        ):
            return

        cf_type = self.combo_cf.currentText()
        if cf_type not in self.latest_cf:
            return

        vals = self.latest_cf[cf_type]
        scale = self.spin_torque_scale.value()

        # ZTCF is acceleration (angular), ZVCF is torque (angular)
        # is_accel = "accel" in cf_type  # Unused, but concept is similar

        for i in range(1, self.model.njoints):
            joint = self.model.joints[i]
            idx_v = joint.idx_v
            nv = joint.nv
            if nv != 1:
                continue

            val = vals[idx_v]
            if abs(val) < 1e-3:
                continue

            oMi = self.data.oMi[i]
            S = joint.S
            # Transform to world spatial vector
            spatial_vec = oMi.act(S * val)

            # If accel, use angular. If torque, use angular.
            # Both are rotational usually for golf.
            vec = spatial_vec.angular
            if np.linalg.norm(vec) < 1e-6:
                vec = spatial_vec.linear

            self._draw_arrow(
                f"overlays/cf/{self.model.names[i]}",
                oMi.translation,
                vec * scale,
                0xFFFF00,  # Yellow
            )

    def _draw_arrow(
        self, path: str, start: np.ndarray, vector: np.ndarray, color: int
    ) -> None:
        """Helper to draw an arrow in Meshcat."""
        if self.viewer is None:
            return

        points = np.vstack([start, start + vector]).T.astype(np.float32)
        self.viewer[path].set_object(
            g.Line(g.PointsGeometry(points), g.LineBasicMaterial(color=color))
        )

    def _draw_frames(self) -> None:
        if self.model is None or self.data is None or self.viewer is None:
            return

        for i, frame in enumerate(self.model.frames):
            if frame.name == "universe":
                continue

            transform = self.data.oMf[i]
            homogeneous_matrix = transform.homogeneous
            self.viewer[f"overlays/frames/{frame.name}"].set_transform(
                homogeneous_matrix
            )

    def _draw_coms(self) -> None:
        if self.model is None or self.data is None or self.viewer is None:
            return

        for i in range(1, self.model.njoints):
            inertia = self.model.inertias[i]
            joint_transform = self.data.oMi[i]
            com_world = joint_transform.act(inertia.lever)

            self.viewer[f"overlays/coms/{self.model.names[i]}"].set_transform(
                pin.SE3(np.eye(3), com_world).homogeneous
            )

    # --- Vis Helpers ---
    def _toggle_frames(self, checked: bool) -> None:  # noqa: FBT001
        if self.viewer is None:
            return

        if not checked:
            self.viewer["overlays/frames"].delete()
        else:
            if self.model:
                for frame in self.model.frames:
                    if frame.name == "universe":
                        continue
                    self.viewer[f"overlays/frames/{frame.name}"].set_object(
                        g.triad(scale=0.1)
                    )
            self._update_viewer()

    def _toggle_coms(self, checked: bool) -> None:  # noqa: FBT001
        if self.viewer is None:
            return

        if not checked:
            self.viewer["overlays/coms"].delete()
        else:
            if self.model:
                for i in range(1, self.model.njoints):
                    self.viewer[f"overlays/coms/{self.model.names[i]}"].set_object(
                        g.Sphere(COM_SPHERE_RADIUS),
                        g.MeshLambertMaterial(color=COM_COLOR),
                    )
            self._update_viewer()

    def _toggle_forces(self, checked: bool) -> None:  # noqa: FBT001
        if self.viewer is None:
            return
        if not checked:
            self.viewer["overlays/forces"].delete()
        self._update_viewer()

    def _toggle_torques(self, checked: bool) -> None:  # noqa: FBT001
        if self.viewer is None:
            return
        if not checked:
            self.viewer["overlays/torques"].delete()
        self._update_viewer()


def main() -> None:
    """Main entry point for the GUI application."""
    app = QtWidgets.QApplication(sys.argv)
    window = PinocchioGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
