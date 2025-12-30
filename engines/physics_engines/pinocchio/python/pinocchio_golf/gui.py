"""Pinocchio GUI Wrapper (PyQt6 + meshcat)."""

import logging
import sys
import types
from pathlib import Path

try:
    import meshcat.geometry as g
    import meshcat.visualizer as viz
except ImportError:
    pass

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from PyQt6 import QtCore, QtWidgets
from shared.python.biomechanics_data import BiomechanicalData
from shared.python.common_utils import get_shared_urdf_path
from shared.python.plotting import GolfSwingPlotter, MplCanvas
from shared.python.statistical_analysis import StatisticalAnalyzer

from .induced_acceleration import InducedAccelerationAnalyzer

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
    """Records time-series data from Pinocchio simulation."""

    def __init__(self) -> None:
        """Initialize empty recorder."""
        self.reset()

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
        """Add a frame of data to the recording.

        Args:
            time: Current simulation time
            q: Joint positions
            v: Joint velocities
            tau: Joint torques (optional)
            kinetic_energy: System kinetic energy
            potential_energy: System potential energy
            club_head_position: Club head position (3,)
            club_head_velocity: Club head linear velocity (3,)
            induced_accelerations: Dict of induced accel components
            counterfactuals: Dict of counterfactual metrics
        """
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

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray | list]:
        """Extract time series for a specific field.

        Args:
            field_name: Name of the field in BiomechanicalData

        Returns:
            Tuple of (times, values)
        """
        if not self.frames:
            return np.array([]), np.array([])

        times = np.array([f.time for f in self.frames])
        values = [getattr(f, field_name) for f in self.frames]

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
        self, source_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract time series for a specific induced acceleration source.

        Args:
            source_name: Name of the force source

        Returns:
            Tuple of (times, acceleration_array)
        """
        if not self.frames:
            return np.array([]), np.array([])

        times = np.array([f.time for f in self.frames])

        # Check if frames have induced acceleration data
        first_frame = self.frames[0]
        if (
            hasattr(first_frame, "induced_accelerations")
            and source_name in first_frame.induced_accelerations
        ):
            values = [f.induced_accelerations[source_name] for f in self.frames]
            return times, np.array(values)

        # Return empty arrays if no data
        return np.array([]), np.array([])

    def get_counterfactual_series(self, cf_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract time series for a specific counterfactual component.

        Args:
            cf_name: Name of the counterfactual (e.g. 'ztcf', 'zvcf')

        Returns:
            Tuple of (times, data_array)
        """
        if not self.frames:
            return np.array([]), np.array([])

        times = np.array([f.time for f in self.frames])

        # Check if frames have counterfactual data
        first_frame = self.frames[0]
        if (
            hasattr(first_frame, "counterfactuals")
            and cf_name in first_frame.counterfactuals
        ):
            values = [f.counterfactuals[cf_name] for f in self.frames]
            return times, np.array(values)

        # Return empty arrays if no data
        return np.array([]), np.array([])


class PinocchioGUI(QtWidgets.QMainWindow):
    """Main GUI widget for Pinocchio robot visualization and computation."""

    def __init__(self) -> None:
        """Initialize the Pinocchio GUI."""
        super().__init__()
        self.setWindowTitle("Pinocchio Golf Model (Dynamics & Kinematics)")
        self.resize(800, 900)  # Increased size for analysis tabs

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

        # Recorder
        self.recorder = PinocchioRecorder()
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
        try:
            self.viewer = viz.Visualizer()
            url = self.viewer.url() if callable(self.viewer.url) else self.viewer.url
            logger.info("Internal Meshcat URL: %s", url)

            try:
                port = url.split(":")[-1].split("/")[0]
                host_url = f"http://127.0.0.1:{port}/static/"
                logger.info(f"Host Access URL: {host_url}")
                self.log_write(f"Visualizer available at: {host_url}")
            except Exception:
                logger.info("Could not determine host URL from: %s", url)
        except (ConnectionError, OSError, RuntimeError) as exc:
            logger.error(f"Failed to initialize Meshcat viewer: {exc}")
            self.log_write(f"Error: Failed to initialize Meshcat viewer: {exc}")
            self.log_write("Please ensure meshcat-server is running or try again.")

        # Setup UI
        self._setup_ui()

        # Model Management
        self.available_models: list[dict] = []
        self._scan_urdf_models()

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

        # Ellipsoids row
        ellip_layout = QtWidgets.QHBoxLayout()
        self.chk_mobility = QtWidgets.QCheckBox("Mobility Ellipsoid (Green)")
        self.chk_mobility.toggled.connect(self._update_viewer)
        ellip_layout.addWidget(self.chk_mobility)

        self.chk_force_ellip = QtWidgets.QCheckBox("Force Ellipsoid (Red)")
        self.chk_force_ellip.toggled.connect(self._update_viewer)
        ellip_layout.addWidget(self.chk_force_ellip)
        vis_layout.addLayout(ellip_layout)

        # Vector Scales
        scale_layout = QtWidgets.QHBoxLayout()
        self.spin_force_scale = QtWidgets.QDoubleSpinBox()
        self.spin_force_scale.setRange(0.01, 10.0)
        self.spin_force_scale.setSingleStep(0.05)
        self.spin_force_scale.setValue(0.1)
        self.spin_force_scale.setPrefix("F Scale: ")
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

        self.chk_forces = QtWidgets.QCheckBox("Show Forces")
        self.chk_forces.toggled.connect(self._toggle_forces)
        vis_layout.addWidget(self.chk_forces)

        self.chk_torques = QtWidgets.QCheckBox("Show Torques")
        self.chk_torques.toggled.connect(self._toggle_torques)
        vis_layout.addWidget(self.chk_torques)

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

        # Tab 2: Analysis & Plotting
        self._setup_analysis_tab()

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
                "Swing DNA (Radar)",
                "Power Flow",
            ]
        )

        self.joint_select_combo = QtWidgets.QComboBox()
        # Will be populated when model loads
        self.joint_select_combo.setMinimumWidth(120)

        self.induced_source_edit = QtWidgets.QLineEdit()
        self.induced_source_edit.setPlaceholderText("Induced Source (Torque Vector)")
        self.induced_source_edit.setToolTip(
            "Enter comma-separated torques or keep empty for standard analysis"
        )
        self.induced_source_edit.setMaximumWidth(150)

        controls.addWidget(QtWidgets.QLabel("Joint:"))
        controls.addWidget(self.joint_select_combo)
        controls.addWidget(QtWidgets.QLabel("Source:"))
        controls.addWidget(self.induced_source_edit)
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

        self.main_tabs.addTab(analysis_page, "Analysis")

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
        elif plot_type == "Swing DNA (Radar)":
            self._plot_swing_dna(plotter)
        elif plot_type == "Power Flow":
            # Requires power data in recorder
            if any(f.actuator_powers.size > 0 for f in self.recorder.frames):
                plotter.plot_power_flow(self.canvas.fig)
            else:
                ax = self.canvas.fig.add_subplot(111)
                ax.text(0.5, 0.5, "No actuator power data", ha="center", va="center")

        self.canvas.draw()

    def _plot_swing_dna(self, plotter: GolfSwingPlotter) -> None:
        """Plot the Swing DNA radar chart."""
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
        # Check for data presence
        if not self.recorder.frames:
            return

        # Check if we have induced data in frames
        # If we recorded it frame-by-frame, we can just pull it.
        # But we currently don't populate it in _game_loop to save perf
        # (unless I update _game_loop).
        # However, we can compute post-hoc if we saved Q/V/Tau.

        # Let's assume we use the Analyzer to recompute if missing,
        # or use saved if present.

        # Get selected joint
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

        # Prepare arrays
        times = []
        g_accs = []
        v_accs = []
        c_accs = []
        t_accs = []

        # Use existing analyzer or create new
        self._ensure_analyzer_initialized()

        for frame in self.recorder.frames:
            times.append(frame.time)

            # If we have pre-computed induced accels (future proofing)
            if frame.induced_accelerations and self.model is not None:
                # Assume dictionary structure
                # We need to map joint index to array index.
                # Usually array is size NV. v_idx points to start.
                # Assuming 1-DOF for plotting.
                g = frame.induced_accelerations.get("gravity", np.zeros(self.model.nv))
                v = frame.induced_accelerations.get("velocity", np.zeros(self.model.nv))
                c = frame.induced_accelerations.get("control", np.zeros(self.model.nv))
                t = frame.induced_accelerations.get("total", np.zeros(self.model.nv))

                g_accs.append(g[v_idx])
                v_accs.append(v[v_idx])
                c_accs.append(c[v_idx])
                t_accs.append(t[v_idx])

            else:
                # Recompute post-hoc
                if self.analyzer is not None:
                    res = self.analyzer.compute_components(
                        frame.joint_positions,
                        frame.joint_velocities,
                        frame.joint_torques,
                    )
                g_accs.append(res["gravity"][v_idx])
                v_accs.append(res["velocity"][v_idx])
                c_accs.append(res["control"][v_idx])
                t_accs.append(res["total"][v_idx])

        # Specific Control Input
        spec_tau = None
        txt = self.induced_source_edit.text().strip()
        if txt and self.model:
            try:
                # Parse comma separated values
                parts = [float(x) for x in txt.split(",")]
                if len(parts) == self.model.nv:
                    spec_tau = np.array(parts)
                else:
                    # Pad or truncate? Or just log warning.
                    # Let's try to map to size NV.
                    temp_tau: np.ndarray = np.zeros(self.model.nv)
                    for i, v in enumerate(parts):
                        if i < len(temp_tau):
                            temp_tau[i] = float(v)
                    spec_tau = temp_tau
            except ValueError:
                pass

        spec_accs = []
        if spec_tau is not None and self.analyzer:
            # Recompute for whole trajectory
            for frame in self.recorder.frames:
                # We need q.
                # Assuming compute_specific_control(q, tau)
                a_spec = self.analyzer.compute_specific_control(
                    frame.joint_positions, spec_tau
                )
                spec_accs.append(a_spec[v_idx])

        # Plot
        ax = self.canvas.fig.add_subplot(111)
        ax.plot(times, g_accs, label="Gravity", linestyle="--", alpha=0.8)
        ax.plot(times, v_accs, label="Velocity (Coriolis)", linestyle="-.", alpha=0.8)
        ax.plot(times, c_accs, label="Control (Torque)", linestyle=":", alpha=0.8)
        ax.plot(times, t_accs, label="Total", color="k", linewidth=1.5)

        if spec_accs:
            ax.plot(times, spec_accs, label="Specific Source", color="m", linewidth=2)

        ax.set_title(f"Induced Accelerations: {joint_name}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Acceleration [rad/s²]")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

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

        times = []
        ztcf_vals = []  # Acceleration
        zvcf_vals = []  # Torque/Force

        if self.analyzer is None and self.model is not None:
            self._ensure_analyzer_initialized()

        for frame in self.recorder.frames:
            times.append(frame.time)

            if frame.counterfactuals and self.model is not None:
                ztcf = frame.counterfactuals.get("ztcf_accel", np.zeros(self.model.nv))
                zvcf = frame.counterfactuals.get("zvcf_torque", np.zeros(self.model.nv))
                ztcf_vals.append(ztcf[v_idx])
                zvcf_vals.append(zvcf[v_idx])
            else:
                # Recompute
                if self.analyzer is not None and hasattr(
                    self.analyzer, "compute_counterfactuals"
                ):
                    res = self.analyzer.compute_counterfactuals(
                        frame.joint_positions, frame.joint_velocities
                    )
                ztcf_vals.append(res["ztcf_accel"][v_idx])
                zvcf_vals.append(res["zvcf_torque"][v_idx])

        # Plot with dual y-axis
        ax1 = self.canvas.fig.add_subplot(111)
        line1 = ax1.plot(times, ztcf_vals, "b-", label="ZTCF Accel (Zero Torque)")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Acceleration [rad/s²]", color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        ax2 = ax1.twinx()
        line2 = ax2.plot(times, zvcf_vals, "r--", label="ZVCF Torque (Zero Velocity)")
        ax2.set_ylabel("Torque [Nm]", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        ax1.set_title(f"Counterfactual Analysis: {joint_name}")

        # Legend
        lns = line1 + line2
        labs = [str(line.get_label()) for line in lns]
        ax1.legend(lns, labs, loc=0)

        ax1.grid(True)

    def _export_statistics(self) -> None:
        """Export statistical analysis to CSV."""
        if self.recorder.get_num_frames() == 0:
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Statistics", "stats.csv", "CSV Files (*.csv)"
        )
        if not filename:
            return

        times, positions = self.recorder.get_time_series("joint_positions")
        _, velocities = self.recorder.get_time_series("joint_velocities")
        _, torques = self.recorder.get_time_series("joint_torques")

        # Ensure arrays are numpy arrays for mypy
        positions_arr = np.asarray(positions)
        velocities_arr = np.asarray(velocities)
        torques_arr = np.asarray(torques)

        analyzer = StatisticalAnalyzer(
            times, positions_arr, velocities_arr, torques_arr
        )

        try:
            analyzer.export_statistics_csv(filename)
            self.log_write(f"Statistics exported to {filename}")
        except Exception as e:
            self.log_write(f"Error exporting statistics: {e}")

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

            # Reset recorder
            self.recorder.reset()
            self.lbl_rec_status.setText("Frames: 0")
            if self.btn_record.isChecked():
                self.btn_record.setChecked(False)
                self.btn_record.setText("Record")

            # Initialize Pinocchio MeshcatVisualizer
            if self.viewer is not None:
                self.viewer["robot"].delete()
                self.viewer["overlays"].delete()
            else:
                self.log_write("Error: Meshcat viewer not initialized")
                return

            self.viz = MeshcatVisualizer(
                self.model, self.collision_model, self.visual_model
            )
            self.viz.initViewer(viewer=self.viewer, open=False)
            self.viz.loadViewerModel()

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
                # We assume the last body/frame is the club head or end-effector
                club_head_pos = None
                club_head_vel = None

                # Find club head body
                # Heuristic: look for "club" or "head" or take last body
                club_id = -1
                for fid in range(self.model.nframes):
                    name = self.model.frames[fid].name.lower()
                    if "club" in name or "head" in name:
                        club_id = fid
                        break

                if club_id == -1 and self.model.nframes > 0:
                    club_id = self.model.nframes - 1

                if club_id >= 0:
                    # Get position and velocity
                    # Need to update frame placement first
                    # (done in update_viewer usually, but needed here)
                    pin.forwardKinematics(self.model, self.data, self.q, self.v)
                    pin.updateFramePlacements(self.model, self.data)

                    frame = self.data.oMf[club_id]
                    club_head_pos = frame.translation.copy()

                    # Velocity
                    v_frame = pin.getFrameVelocity(
                        self.model,
                        self.data,
                        club_id,
                        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                    )
                    club_head_vel = v_frame.linear.copy()

                # Ensure q is not None for recording
                q_for_recording = self.q if self.q is not None else np.array([])

                # Induced / Counterfactuals
                # Computing every frame inside loop is costly but most correct
                # for playback.
                # Let's compute them.
                induced = None
                counterfactuals = None

                if self.analyzer and self.q is not None and self.v is not None:
                    induced = self.analyzer.compute_components(self.q, self.v, tau)
                    if hasattr(self.analyzer, "compute_counterfactuals"):
                        counterfactuals = self.analyzer.compute_counterfactuals(
                            self.q, self.v
                        )

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

        if self.chk_mobility.isChecked() or self.chk_force_ellip.isChecked():
            self._draw_ellipsoids()
        else:
            if self.viewer:
                self.viewer["overlays/ellipsoids"].delete()

    def _compute_analysis(self) -> None:
        """Compute Jacobian and Mass matrix analysis."""
        if self.model is None or self.data is None or self.q is None:
            return

        # Compute Jacobian for end-effector (assumed last joint for now)
        joint_id = self.model.njoints - 1

        # We need to ensure Jacobians are computed.
        # pin.computeJointJacobians(self.model, self.data, self.q)
        # Done in update loop via FK? No.
        pin.computeJointJacobians(self.model, self.data, self.q)
        J = pin.getJointJacobian(
            self.model, self.data, joint_id, pin.ReferenceFrame.LOCAL
        )

        # Condition number
        try:
            s = np.linalg.svd(J, compute_uv=False)
            cond = s[0] / s[-1] if s[-1] > 1e-9 else float("inf")
            self.lbl_cond.setText(f"{cond:.2f}")
        except Exception:
            self.lbl_cond.setText("Error")

        # Mass Matrix Rank
        M = pin.crba(self.model, self.data, self.q)
        try:
            rank = np.linalg.matrix_rank(M)
            self.lbl_rank.setText(f"{rank} / {self.model.nv}")
        except Exception:
            self.lbl_rank.setText("Error")

    def _draw_ellipsoids(self) -> None:
        """Draw mobility/force ellipsoids at end effector."""
        if self.model is None or self.data is None or self.viewer is None:
            return

        # End effector joint
        joint_id = self.model.njoints - 1
        joint_pos = self.data.oMi[joint_id].translation

        # Get Jacobian (Translational only for 3D visualization)
        J_full = pin.getJointJacobian(
            self.model,
            self.data,
            joint_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        J = J_full[:3, :]  # Linear part

        # Mass Matrix
        M = pin.crba(self.model, self.data, self.q)
        # Ensure symmetric if using older pin version
        M_sym = np.triu(M) + np.triu(M, 1).T

        try:
            Minv = np.linalg.inv(M_sym)
            Lambda_inv = J @ Minv @ J.T

            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(Lambda_inv)

            if self.chk_mobility.isChecked():
                # Radii = sqrt(eigenvalues)
                radii = np.sqrt(np.maximum(eigvals, 1e-6))
                self._draw_ellipsoid_meshcat(
                    "mobility", joint_pos, eigvecs, radii, 0x00FF00
                )

            if self.chk_force_ellip.isChecked():
                # Radii = 1/sqrt(eigenvalues)
                radii_force = 1.0 / np.sqrt(np.maximum(eigvals, 1e-6))
                # Clip to reasonable visual size
                radii_force = np.clip(radii_force, 0.01, 5.0)
                self._draw_ellipsoid_meshcat(
                    "force", joint_pos, eigvecs, radii_force, 0xFF0000
                )

        except Exception as e:
            logger.error(f"Ellipsoid computation failed: {e}")

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

        # Meshcat Sphere scaled
        self.viewer[path].set_object(
            g.Sphere(1.0),
            g.MeshLambertMaterial(color=color, opacity=0.5, transparent=True),
        )

        # Transform: [Rot * Diag(radii) | Pos]
        # Meshcat applies transform to the object.
        # We need to construct 4x4 matrix
        T = np.eye(4)
        T[:3, :3] = rot @ np.diag(radii)
        T[:3, 3] = pos

        self.viewer[path].set_transform(T)

    def _draw_vectors(self) -> None:
        """Draw force and torque vectors at joints."""
        if self.model is None or self.data is None or self.viewer is None:
            return

        # We need accelerations for RNEA to get consistent joint forces.
        # If in kinematic mode, assume zero v/a.
        v = self.v if self.v is not None else np.zeros(self.model.nv)
        # Use ABA to get a if it hasn't been computed recently
        a = pin.aba(self.model, self.data, self.q, v, np.zeros(self.model.nv))

        # Compute reaction forces via RNEA
        # self.data.f will contain spatial forces at each joint
        pin.rnea(self.model, self.data, self.q, v, a)

        force_scale = self.spin_force_scale.value()
        torque_scale = self.spin_torque_scale.value()

        for i in range(1, self.model.njoints):
            joint_placement = self.data.oMi[i]
            # f is a spatial force (linear part = forces, angular part = torques)
            f_local = self.data.f[i]

            # Linear force in world frame
            f_world = joint_placement.rotation @ f_local.linear
            # Angular torque in world frame
            t_world = joint_placement.rotation @ f_local.angular

            joint_name = self.model.names[i]

            # Draw Force
            if self.chk_forces.isChecked() and np.linalg.norm(f_world) > 1e-3:
                self._draw_arrow(
                    f"overlays/forces/{joint_name}",
                    joint_placement.translation,
                    f_world * force_scale,
                    0xFF0000,  # Red for force
                )

            # Draw Torque
            if self.chk_torques.isChecked() and np.linalg.norm(t_world) > 1e-3:
                self._draw_arrow(
                    f"overlays/torques/{joint_name}",
                    joint_placement.translation,
                    t_world * torque_scale,
                    0x0000FF,  # Blue for torque
                )

    def _draw_arrow(
        self, path: str, start: np.ndarray, vector: np.ndarray, color: int
    ) -> None:
        """Helper to draw an arrow in Meshcat."""
        if self.viewer is None:
            return

        # Note: Meshcat Arrow might not exist in all versions, using a Line for now
        # as it is highly compatible. Arrows can be added with Triad or custom mesh.
        # Simplified: Draw a line from start to start + vector
        points = np.vstack([start, start + vector]).T.astype(np.float32)
        self.viewer[path].set_object(
            g.Line(g.PointsGeometry(points), g.LineBasicMaterial(color=color))
        )

    def _draw_frames(self) -> None:
        if self.model is None or self.data is None or self.viewer is None:
            return

        # Visualize joint frames (oMf)
        # To optimize performance, we create frame objects once and update their
        # transforms each frame. The Meshcat Python client caches objects, so
        # updating transforms is efficient for visualization.
        for i, frame in enumerate(self.model.frames):
            if frame.name == "universe":
                continue

            # Update transform
            transform = self.data.oMf[i]
            # Convert Pinocchio SE3 (transform) to a 4x4 homogeneous transformation
            # matrix.
            # Pinocchio's SE3.homogeneous property returns a 4x4 matrix representing
            # the pose in the world frame. Meshcat's set_transform expects a 4x4
            # column-major matrix (compatible with this layout).
            homogeneous_matrix = transform.homogeneous
            self.viewer[f"overlays/frames/{frame.name}"].set_transform(
                homogeneous_matrix
            )

    def _draw_coms(self) -> None:
        if self.model is None or self.data is None or self.viewer is None:
            return

        # Draw Center of Mass for each link
        for i in range(1, self.model.njoints):
            # Joint i. Associated body has inertia.
            inertia = self.model.inertias[i]
            joint_transform = self.data.oMi[i]
            # Compute world-frame COM for each body by transforming the local COM
            # (inertia.lever) through the joint placement (joint_transform).
            com_world = joint_transform.act(inertia.lever)

            # Update Sphere Position
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
            # Create objects once
            if self.model:
                for frame in self.model.frames:
                    if frame.name == "universe":
                        continue
                    # Triad
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
