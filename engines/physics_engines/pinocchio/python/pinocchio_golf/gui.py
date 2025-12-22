"""Pinocchio GUI Wrapper (PyQt6 + meshcat)."""

import logging
import sys
import types
from pathlib import Path

import meshcat.geometry as g
import meshcat.visualizer as viz
import numpy as np  # noqa: TID253
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from PyQt6 import QtCore, QtWidgets

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


class PinocchioGUI(QtWidgets.QMainWindow):
    """Main GUI widget for Pinocchio robot visualization and computation."""

    def __init__(self) -> None:
        """Initialize the Pinocchio GUI."""
        super().__init__()
        self.setWindowTitle("Pinocchio Golf Model (Dynamics & Kinematics)")
        self.resize(600, 800)

        # Internal state
        self.model: pin.Model | None = None
        self.data: pin.Data | None = None
        self.visual_model: pin.VisualModel | None = None
        self.collision_model: pin.CollisionModel | None = None
        self.viz: MeshcatVisualizer | None = None
        self.q: np.ndarray | None = None
        self.v: np.ndarray | None = None

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
        # Do not open browser automatically; user can open Meshcat URL manually if
        # desired.
        self.viewer: viz.Visualizer | None = None
        try:
            self.viewer = viz.Visualizer()  # Let it find port
            url = self.viewer.url() if callable(self.viewer.url) else self.viewer.url
            logger.info("Meshcat URL: %s", url)
        except (ConnectionError, OSError, RuntimeError) as exc:
            logger.error(f"Failed to initialize Meshcat viewer: {exc}")
            self.log_write(f"Error: Failed to initialize Meshcat viewer: {exc}")
            self.log_write("Please ensure meshcat-server is running or try again.")

        # Setup UI
        self._setup_ui()

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._game_loop)
        # Timer will be started after a valid model is loaded

        # Try load default model
        default_urdf = (
            Path(__file__).parent / "../../models/generated/golfer.urdf"
        ).resolve()

        if default_urdf.exists():
            self.load_urdf(str(default_urdf))

    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # 1. Top Bar: Load & Mode
        top_layout = QtWidgets.QHBoxLayout()

        self.load_btn = QtWidgets.QPushButton("Load URDF")
        self.load_btn.clicked.connect(lambda: self.load_urdf())
        top_layout.addWidget(self.load_btn)

        top_layout.addStretch()

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Dynamic (Physics)", "Kinematic (Pose)"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        top_layout.addWidget(QtWidgets.QLabel("Mode:"))
        top_layout.addWidget(self.mode_combo)

        layout.addLayout(top_layout)

        # 2. Controls Stack
        self.controls_stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.controls_stack)

        self._setup_dynamic_tab()
        self._setup_kinematic_tab()

        # 3. Visuals & Logs
        vis_group = QtWidgets.QGroupBox("Visualization")
        vis_layout = QtWidgets.QHBoxLayout()

        self.chk_frames = QtWidgets.QCheckBox("Show Frames")
        self.chk_frames.toggled.connect(self._toggle_frames)
        vis_layout.addWidget(self.chk_frames)

        self.chk_coms = QtWidgets.QCheckBox("Show COMs")
        self.chk_coms.toggled.connect(self._toggle_coms)
        vis_layout.addWidget(self.chk_coms)

        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)

        self.log = LogPanel()
        layout.addWidget(self.log)

    def log_write(self, text: str) -> None:
        self.log.append(text)
        logger.info(text)

    def _setup_dynamic_tab(self) -> None:
        dyn_page = QtWidgets.QWidget()
        dyn_layout = QtWidgets.QVBoxLayout(dyn_page)
        self.btn_run = QtWidgets.QPushButton("Run Simulation")
        self.btn_run.setCheckable(True)
        self.btn_run.clicked.connect(self._toggle_run)
        dyn_layout.addWidget(self.btn_run)

        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset_simulation)
        dyn_layout.addWidget(self.btn_reset)

        dyn_layout.addStretch()
        self.controls_stack.addWidget(dyn_page)

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
            # Multi-DOF joints (e.g., spherical) are not supported in the UI.
            # Such joints are intentionally skipped and will not appear in the
            # kinematic controls.
            msg = (
                f"Skipping joint '{joint_name}' (index {i}): "
                f"{nq_joint} DOFs not supported in kinematic controls."
            )
            self.log_write(msg)
            return

        self.joint_names.append(joint_name)

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
        self.btn_run.setText("Run Simulation")
        self.btn_run.setChecked(False)
        self._update_viewer()
        self._sync_kinematic_controls()

    def _game_loop(self) -> None:
        if self.model is None or self.data is None or self.q is None or self.v is None:
            return

        if self.operating_mode == "dynamic" and self.is_running:
            # --- Physics integration loop ---
            # Set joint torques to zero to simulate passive dynamics (no actuation).
            # This models the system's natural motion under gravity and inertia only.
            tau = np.zeros(self.model.nv)

            # Compute joint accelerations using Articulated-Body Algorithm (ABA).
            a = pin.aba(self.model, self.data, self.q, self.v, tau)

            # Integrate using the semi-implicit (symplectic) Euler method:
            #   1. Update velocity: v_{n+1} = v_n + a * dt
            #   2. Update position: q_{n+1} = integrate(q_n, v_{n+1} * dt)
            # This method is preferred over explicit Euler for mechanical systems
            # because it provides better energy behavior and stability, especially
            # for stiff or underactuated systems.
            self.v += a * self.dt
            self.q = pin.integrate(self.model, self.q, self.v * self.dt)

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

        if self.chk_coms.isChecked():
            pin.centerOfMass(self.model, self.data, self.q)

        # Overlays
        if self.chk_frames.isChecked():
            self._draw_frames()
        if self.chk_coms.isChecked():
            self._draw_coms()

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
                        g.Triad(scale=0.1)
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


def main() -> None:
    """Main entry point for the GUI application."""
    app = QtWidgets.QApplication(sys.argv)
    window = PinocchioGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
