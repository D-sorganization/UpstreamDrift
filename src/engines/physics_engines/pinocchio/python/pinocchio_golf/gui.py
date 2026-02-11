"""Pinocchio GUI Wrapper (PyQt6 + meshcat).

Refactored: PinocchioRecorder, LogPanel, and SignalBlocker are in
``pinocchio_recorder.py``. Visualization methods are in
``pinocchio_visualization_mixin.py``. Analysis/plotting methods are in
``pinocchio_analysis_mixin.py``.
"""

import sys
from pathlib import Path
from typing import Any

# Add project root to path for src imports when run as standalone script
# Path: src/engines/physics_engines/pinocchio/python/pinocchio_golf/gui.py
# Need 7 parents to reach root
_project_root = (
    Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent
)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np  # noqa: E402
import pinocchio as pin  # type: ignore  # noqa: E402
from PyQt6 import QtCore, QtWidgets  # noqa: E402

from src.shared.python.common_utils import get_shared_urdf_path  # noqa: E402
from src.shared.python.dashboard.widgets import LivePlotWidget  # noqa: E402
from src.shared.python.logging_config import (  # noqa: E402
    configure_gui_logging,
    get_logger,
)
from src.shared.python.ui.simulation_gui_base import SimulationGUIBase  # noqa: E402

# Mixin and helper imports
from .manipulability import PinocchioManipulabilityAnalyzer  # noqa: E402
from .pinocchio_analysis_mixin import PinocchioAnalysisMixin  # noqa: E402
from .pinocchio_recorder import (  # noqa: E402
    LogPanel,
    PinocchioRecorder,
    SignalBlocker,
)
from .pinocchio_visualization_mixin import (  # noqa: E402
    MESHCAT_AVAILABLE,
    PinocchioVisualizationMixin,
)

if MESHCAT_AVAILABLE:
    import meshcat.geometry as g  # noqa: E402
    import meshcat.visualizer as viz  # noqa: E402
    from pinocchio.visualize import MeshcatVisualizer  # noqa: E402
else:
    g = None  # type: ignore
    viz = None  # type: ignore
    MeshcatVisualizer = object  # type: ignore  # Dummy class if missing

try:
    from .induced_acceleration import InducedAccelerationAnalyzer
except ImportError:
    from induced_acceleration import (  # type: ignore[no-redef]
        InducedAccelerationAnalyzer,
    )

# Set up logging using centralized module
configure_gui_logging()
logger = get_logger(__name__)

# Constants
DT_DEFAULT = 0.01  # [s] Physics time step
SLIDER_RANGE_RAD = 10.0  # [rad] Range for joint sliders
SLIDER_SCALE = 100.0  # Scale factor for QSlider (int) -> rad (float)


class PinocchioGUI(  # type: ignore[misc]
    PinocchioAnalysisMixin,
    PinocchioVisualizationMixin,
    SimulationGUIBase,
):
    """Main GUI widget for Pinocchio robot visualization and computation."""

    WINDOW_TITLE = "Pinocchio Golf Model (Dynamics & Kinematics)"
    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 900

    def __init__(self) -> None:
        """Initialize the Pinocchio GUI."""
        super().__init__()

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
        logger.info("Pinocchio Version: %s", pin_version)
        logger.info("Python Executable: %s", sys.executable)

        # Initialize log panel early so log_write() works during init
        self.log = LogPanel()

        # Meshcat viewer
        self.viewer: viz.Visualizer | None = None  # type: ignore[union-attr]
        if MESHCAT_AVAILABLE:
            try:
                try:
                    self.viewer = viz.Visualizer(server_args=["--port", "7000"])
                except TypeError:
                    logger.warning(
                        "Meshcat Visualizer: server_args not supported. Using default."
                    )
                    self.viewer = viz.Visualizer()

                if callable(self.viewer.url):
                    url = self.viewer.url()
                else:
                    url = self.viewer.url
                logger.info("Internal Meshcat URL: %s", url)

                try:
                    port = url.split(":")[-1].split("/")[0]
                    host_url = f"http://127.0.0.1:{port}/static/"
                    logger.info("Host Access URL: %s", host_url)
                    self.log_write("=" * 40)
                    self.log_write("VISUALIZER READY")
                    self.log_write("Open this URL in your browser:")
                    self.log_write(f"{host_url}")
                    self.log_write("=" * 40)
                except (PermissionError, OSError):
                    logger.info("Could not determine host URL from: %s", url)
            except (ConnectionError, OSError, RuntimeError) as exc:
                logger.error("Failed to initialize Meshcat viewer: %s", exc)
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
        except (RuntimeError, ValueError, OSError) as e:
            logger.error("Failed to scan URDF models: %s", e)

    def _setup_ui(self) -> None:  # noqa: PLR0915
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

        toggles_layout = QtWidgets.QHBoxLayout()
        self.chk_mobility = QtWidgets.QCheckBox("Mobility (Green)")
        self.chk_mobility.toggled.connect(self._update_viewer)
        toggles_layout.addWidget(self.chk_mobility)

        self.chk_force_ellip = QtWidgets.QCheckBox("Force (Red)")
        self.chk_force_ellip.toggled.connect(self._update_viewer)
        toggles_layout.addWidget(self.chk_force_ellip)
        ellip_layout.addLayout(toggles_layout)

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

        # Tab 3: Post-Hoc Analysis & Plotting (from PinocchioAnalysisMixin)
        self._setup_analysis_tab()

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

        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run Simulation")
        self.btn_run.setCheckable(True)
        self.btn_run.clicked.connect(self._toggle_run)
        btn_layout.addWidget(self.btn_run)

        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset_simulation)
        btn_layout.addWidget(self.btn_reset)
        dyn_layout.addLayout(btn_layout)

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

            self.analyzer = InducedAccelerationAnalyzer(self.model, self.data)
            self.manip_analyzer = PinocchioManipulabilityAnalyzer(self.model, self.data)
            self._populate_manipulability_checkboxes()

            self.recorder.reset()
            self.lbl_rec_status.setText("Frames: 0")
            if self.btn_record.isChecked():
                self.btn_record.setChecked(False)
                self.btn_record.setText("Record")

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

            self._build_kinematic_controls()
            self._sync_kinematic_controls()

            self._update_viewer()

            if self.chk_frames.isChecked():
                self._toggle_frames(checked=True)
            if self.chk_coms.isChecked():
                self._toggle_coms(checked=True)

            if not self.timer.isActive():
                self.timer.start(int(self.dt * 1000))

            if hasattr(self, "live_plot"):
                self.live_plot.set_joint_names(self.get_joint_names())

        except (ValueError, RuntimeError) as e:
            self.log_write(f"Error loading URDF (Pinocchio): {e}")
        except (PermissionError, OSError) as e:
            self.log_write(f"Unexpected error loading URDF: {e}")
            logger.exception("Unexpected error loading URDF")

    def _build_kinematic_controls(self) -> None:
        if self.model is None:
            return

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

        self.joint_names = list(self.model.names)[1:]

        self.joint_select_combo.clear()
        self.joint_select_combo.addItems(self.joint_names)

        for i in range(1, self.model.njoints):
            self._add_joint_control_widget(i)

    def _add_joint_control_widget(self, i: int) -> None:
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

    def _sync_kinematic_controls(self) -> None:
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
            self.is_running = False
            self.btn_run.setText("Run Simulation")
            self.btn_run.setChecked(False)
            self._sync_kinematic_controls()

    def _toggle_run(self, checked: bool = False) -> None:  # noqa: FBT001, FBT002
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

        self.recorder.reset()
        self.lbl_rec_status.setText("Frames: 0")
        if self.btn_record.isChecked():
            self.btn_record.setChecked(False)
            self.btn_record.setText("Record")

    def _game_loop(self) -> None:  # noqa: PLR0912, PLR0915
        if self.model is None or self.data is None or self.q is None or self.v is None:
            return

        if hasattr(self, "live_plot"):
            self.live_plot.update_plot()

        if self.operating_mode == "dynamic" and self.is_running:
            tau = np.zeros(self.model.nv)
            a = pin.aba(self.model, self.data, self.q, self.v, tau)
            self.v += a * self.dt
            self.q = pin.integrate(self.model, self.q, self.v * self.dt)
            self.sim_time += self.dt

            # Recording
            if self.recorder.is_recording:
                pin.computeKineticEnergy(self.model, self.data, self.q, self.v)
                pin.computePotentialEnergy(self.model, self.data, self.q)

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

                induced: dict[str, Any] | None = None
                counterfactuals: dict[str, Any] | None = None

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

                if self.chk_live_analysis.isChecked() or config_requests_analysis:
                    if self.analyzer and self.q is not None and self.v is not None:
                        induced = self.analyzer.compute_components(self.q, self.v, tau)
                        self.latest_induced = induced

                        sources_to_compute = []
                        txt = self.combo_induced.currentText()
                        if txt:
                            sources_to_compute.append(txt)

                        has_config = hasattr(self.recorder, "analysis_config")
                        config = getattr(self.recorder, "analysis_config", None)
                        if has_config and isinstance(config, dict):
                            sources = config.get("induced_accel_sources", [])
                            if isinstance(sources, list):
                                sources_to_compute.extend(sources)

                        unique_sources = set()
                        for s in sources_to_compute:
                            if s:
                                unique_sources.add(str(s))

                        for src in unique_sources:
                            if src in ["gravity", "velocity", "total"]:
                                continue

                            spec_tau = np.zeros(self.model.nv)
                            found_joint = False

                            if self.model.existJointName(src):
                                j_id = self.model.getJointId(src)
                                joint = self.model.joints[j_id]
                                if joint.nv == 1:
                                    spec_tau[joint.idx_v] = 1.0
                                    found_joint = True

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
