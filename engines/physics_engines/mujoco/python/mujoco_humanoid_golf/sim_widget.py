"""Qt widget encapsulating a MuJoCo simulation and renderer."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import mujoco
import numpy as np

# ... imports ...
from PyQt6 import QtCore, QtGui, QtWidgets

# Removed unused scipy import
from .biomechanics import BiomechanicalAnalyzer, SwingRecorder
from .control_system import ControlSystem, ControlType
from .interactive_manipulation import InteractiveManipulator
from .meshcat_adapter import MuJoCoMeshcatAdapter
from .telemetry import TelemetryRecorder

# Lazy loading globals for OpenCV
CV2_LIB = None
INVALID_CV2 = False


def get_cv2():
    """Lazy import of OpenCV to speed up initial load."""
    global CV2_LIB, INVALID_CV2
    if CV2_LIB is None and not INVALID_CV2:
        try:
            import cv2

            CV2_LIB = cv2
        except ImportError:
            INVALID_CV2 = True
    return CV2_LIB


LOGGER = logging.getLogger(__name__)
MIN_CAMERA_DEPTH: Final[float] = 0.1
FORCE_VISUALIZATION_THRESHOLD: Final[float] = 1e-5


class ModelLoaderThread(QtCore.QThread):
    """Worker thread to load MuJoCo models asynchronously."""

    # Signal returns (model, data) on success, or (None, error_msg) on failure
    finished_loading = QtCore.pyqtSignal(object, object, str)

    def __init__(self, xml_content: str, is_file: bool = False):
        super().__init__()
        self.xml_content = xml_content
        self.is_file = is_file

    def run(self):
        try:
            if self.is_file:
                model = mujoco.MjModel.from_xml_path(self.xml_content)
            else:
                model = mujoco.MjModel.from_xml_string(self.xml_content)

            data = mujoco.MjData(model)
            self.finished_loading.emit(model, data, "")
        except Exception as e:
            self.finished_loading.emit(None, None, str(e))


class MuJoCoSimWidget(QtWidgets.QWidget):
    """Widget that:
    - Holds a MuJoCo model + data
    - Steps the simulation
    - Renders frames with mujoco.Renderer
    - Displays frames in a QLabel
    - Visualizes forces and torques as 3D vectors
    - Records biomechanical data
    """

    # Signal emitted when model loading starts/ends
    loading_started = QtCore.pyqtSignal()
    loading_finished = QtCore.pyqtSignal(bool)  # True = success

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 60,
    ) -> None:
        """Initialize the simulation widget."""
        super().__init__(parent)
        self.setMinimumSize(width, height)

        self.fps = fps
        self.frame_width = width
        self.frame_height = height

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.renderer: mujoco.Renderer | None = None
        self.control_vector: np.ndarray | None = None
        self.control_system: ControlSystem | None = None
        self.camera_name = "side"

        # Visual toggles
        self.show_force_vectors = False
        self.show_torque_vectors = False
        self.force_scale = 0.1
        self.torque_scale = 0.1

        # Ellipsoid Visualization Toggles
        self.show_mobility_ellipsoid = False
        self.show_force_ellipsoid = False

        # Meshcat integration
        self.meshcat_adapter: MuJoCoMeshcatAdapter | None = None
        try:
            self.meshcat_adapter = MuJoCoMeshcatAdapter()
        except Exception:
            LOGGER.warning("Could not initialize Meshcat adapter")

        self.telemetry: TelemetryRecorder | None = None

        self.running = True  # start in "playing" mode

        # Biomechanical analysis
        self.analyzer: BiomechanicalAnalyzer | None = None
        self.recorder = SwingRecorder()

        # Interactive manipulation
        self.manipulator: InteractiveManipulator | None = None
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.azimuth = 90.0
        self.camera.elevation = -20.0
        self.camera.distance = 3.0
        self.camera.lookat[:] = [0, 0, 1]

        # Visualization for selected bodies and constraints
        self.show_selected_body = True
        self.show_constraints = True

        # Per-body visualization flags (set of body IDs)
        self.visible_frames: set[int] = set()
        self.visible_coms: set[int] = set()

        # Operating Mode ("dynamic" or "kinematic")
        self.operating_mode = "dynamic"

        # MuJoCo scene options for vector rendering
        self.scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.scene_option)
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

        # MuJoCo scene for rendering
        self.scene: mujoco.MjvScene | None = None

        # Background color settings (RGBA)
        self.sky_color = np.array(
            [0.2, 0.3, 0.4, 1.0],
            dtype=np.float32,
        )  # Default sky blue
        self.ground_color = np.array(
            [0.2, 0.2, 0.2, 1.0],
            dtype=np.float32,
        )  # Default dark gray

        # UI: a simple label to show the image
        self.label = QtWidgets.QLabel(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

        # Enable mouse tracking for smooth drag
        self.setMouseTracking(True)
        self.label.setMouseTracking(True)

        # Camera manipulation state
        self.last_mouse_pos: tuple[int, int] | None = None
        self.camera_mode = "rotate"  # "rotate", "translate", "zoom"
        self.is_dragging = False

        # Timer for stepping and rendering
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(int(1000 / self.fps))

        self.loader_thread: ModelLoaderThread | None = None

    # -------- MuJoCo setup --------

    def load_model_async(self, xml_source: str, is_file: bool = False) -> None:
        """Load a MuJoCo model asynchronously to prevent UI freeze.

        Args:
            xml_source: XML string or file path.
            is_file: True if xml_source is a path, False if string content.
        """
        if self.loader_thread and self.loader_thread.isRunning():
            LOGGER.warning("Model loading already in progress.")
            return

        self.timer.stop()
        self.loading_started.emit()

        # Show loading indicator in label?
        self.label.setText("Loading Model...")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.loader_thread = ModelLoaderThread(xml_source, is_file)
        self.loader_thread.finished_loading.connect(self._on_model_loaded_async)
        self.loader_thread.start()

    def _on_model_loaded_async(self, model, data, error_msg):
        """Handle completion of async model loading."""
        if error_msg:
            LOGGER.error("Async load failed: %s", error_msg)
            self.label.setText(f"Error loading model: {error_msg}")
            self.loading_finished.emit(False)
            return

        try:
            self._finalize_model_load(model, data)
            self.loading_finished.emit(True)
        except Exception as e:
            LOGGER.error("Finalization failed: %s", e)
            self.label.setText(f"Error initializing renderer: {e}")
            self.loading_finished.emit(False)

    def _finalize_model_load(self, new_model, new_data):
        """Finalize setup on main thread after model/data creation."""
        # Create new renderer (must be on main thread with context)
        new_renderer = mujoco.Renderer(
            new_model,
            width=self.frame_width,
            height=self.frame_height,
        )

        # Create new scene
        new_scene = mujoco.MjvScene(new_model, maxgeom=10000)
        mujoco.mjv_updateScene(
            new_model,
            new_data,
            self.scene_option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            new_scene,
        )

        # Commit changes
        self.model = new_model
        self.data = new_data
        self.renderer = new_renderer
        self.scene = new_scene

        # Important: Reset camera to ensure it points at the new model
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        # Apply custom defaults if we have them
        self._auto_position_camera()

        # Apply background colors
        self._update_background_colors()

        self.telemetry = TelemetryRecorder(self.model)

        # Reset control system
        self.control_system = ControlSystem(self.model.nu)
        self.control_vector = np.zeros(self.model.nu, dtype=np.float64)

        # Reset Biomechanics
        self.analyzer = BiomechanicalAnalyzer(self.model, self.data)
        self.recorder.reset()

        # Reset Interaction
        self.manipulator = InteractiveManipulator(self.model, self.data)

        # Restart timer
        self.timer.start(int(1000 / self.fps))

    def load_model_from_xml(self, xml_string: str) -> None:
        """(Legacy/Sync) Load a MuJoCo model from an MJCF XML string."""
        self.timer.stop()
        try:
            new_model = mujoco.MjModel.from_xml_string(xml_string)
            new_data = mujoco.MjData(new_model)
            self._finalize_model_load(new_model, new_data)
        except Exception as e:
            LOGGER.error("Sync load failed: %s", e)
            raise

    def load_model_from_file(self, xml_path: str) -> None:
        """(Legacy/Sync) Load from file."""
        self.timer.stop()
        try:
            # Convert to absolute path if needed
            if not os.path.isabs(xml_path):
                project_root = Path(__file__).parent.parent.parent
                xml_path = str(project_root / xml_path)

            if not os.path.exists(xml_path):
                raise FileNotFoundError(f"Model file not found: {xml_path}")

            new_model = mujoco.MjModel.from_xml_path(xml_path)
            new_data = mujoco.MjData(new_model)
            self._finalize_model_load(new_model, new_data)
        except Exception as e:
            LOGGER.error("Sync load failed: %s", e)
            raise

    def reset_state(self) -> None:
        """Set golf-like initial joint angles for all model types."""
        if self.model is None or self.data is None:
            return

        mujoco.mj_resetData(self.model, self.data)

        # Zero all positions/velocities first
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0

        # Forward kinematics to update body positions
        mujoco.mj_forward(self.model, self.data)

        # Use nq to distinguish model types:
        #   nq == 2 -> [shoulder, wrist] - Double pendulum
        #   nq == 3 -> [shoulder, elbow, wrist] - Triple pendulum
        #   nq >= 10 -> Complex models (upper body, full body, etc.)
        if self.model.nq == 2:
            # DOUBLE PENDULUM top-of-backswing-ish:
            shoulder = -1.2  # rad (~ -69 deg)
            wrist = 1.3  # rad (~ 75 deg)

            self.data.qpos[0] = shoulder
            self.data.qpos[1] = wrist

        elif self.model.nq == 3:
            # TRIPLE PENDULUM top-of-backswing-ish:
            shoulder = -1.0  # rad
            elbow = 0.7  # rad
            wrist = 1.2  # rad

            self.data.qpos[0] = shoulder
            self.data.qpos[1] = elbow
            self.data.qpos[2] = wrist

        elif self.model.nq >= 10:
            # Complex models: Set to a neutral/address position
            # Most models start at qpos=0 which is typically address position
            # For models with free joints (pelvis), ensure they're at ground level
            # Check if first joint is a free joint (7 DOF: 3 pos + 4 quat)
            if self.model.njnt > 0:
                first_joint_type = self.model.jnt_type[0]
                if first_joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    # Free joint: set position to reasonable height (Z is at index 2)
                    if len(self.data.qpos) >= 3:
                        self.data.qpos[2] = 0.9  # Z position (height)
            # Keep other joints at 0 (address position)

        elif self.model.nq >= 1:
            self.data.qpos[0] = 0.2

        self.data.qvel[:] = 0.0

        # Forward kinematics to update positions
        mujoco.mj_forward(self.model, self.data)

        self._render_once()

        if self.telemetry is not None:
            self.telemetry.reset()
            self.telemetry.record_step(self.data)

    def _auto_position_camera(self) -> None:
        """Automatically position camera to view the entire model."""
        if self.model is None or self.data is None:
            return

        # Compute model bounds
        bounds = self._compute_model_bounds()
        if bounds is None:
            return

        center = bounds["center"]
        size = bounds["size"]
        max_size = max(size)

        # Set camera lookat to model center
        self.camera.lookat[:] = center

        # Set camera distance based on model size
        # Distance should be about 2-3 times the model size
        if max_size > 0:
            self.camera.distance = max(2.0, max_size * 2.5)
        else:
            self.camera.distance = 3.0

        # Set reasonable default viewing angle
        self.camera.azimuth = 90.0
        self.camera.elevation = -20.0

        # Clamp distance to reasonable range
        self.camera.distance = np.clip(self.camera.distance, 0.5, 50.0)

    def _compute_model_bounds(self) -> dict | None:
        """Compute bounding box of all geoms in the model.

        Returns:
            Dict with 'center' and 'size' keys, or None if computation fails
        """
        if self.model is None or self.data is None:
            return None

        try:
            # Forward kinematics to get current positions
            mujoco.mj_forward(self.model, self.data)

            # Get all body positions
            min_pos = np.array([np.inf, np.inf, np.inf])
            max_pos = np.array([-np.inf, -np.inf, -np.inf])

            # Check all bodies, skipping world body (0)
            for i in range(1, self.model.nbody):
                pos = self.data.xpos[i]
                min_pos = np.minimum(min_pos, pos)
                max_pos = np.maximum(max_pos, pos)

            # Also check geom positions (they might extend beyond body centers)
            for i in range(self.model.ngeom):
                geom_id = i
                body_id = self.model.geom_bodyid[geom_id]
                # Skip world body geoms (like huge ground planes)
                if body_id > 0:
                    geom_pos = self.data.xpos[body_id].copy()
                    geom_size = self.model.geom_size[geom_id]

                    # Approximate geom extent (conservative estimate)
                    if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE:
                        extent = geom_size[0]
                    elif self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                        extent = np.linalg.norm(geom_size)
                    elif self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                        extent = geom_size[0] + geom_size[1]
                    else:
                        extent = np.max(geom_size) if len(geom_size) > 0 else 0.5

                    min_pos = np.minimum(min_pos, geom_pos - extent)
                    max_pos = np.maximum(max_pos, geom_pos + extent)

            # If we got valid bounds
            if np.all(np.isfinite(min_pos)) and np.all(np.isfinite(max_pos)):
                center = (min_pos + max_pos) / 2.0
                size = max_pos - min_pos

                # Ensure minimum size
                size = np.maximum(size, [0.5, 0.5, 0.5])

                return {"center": center, "size": size}

            # Fallback: use default center and size
            return {
                "center": np.array([0.0, 0.0, 1.0]),
                "size": np.array([2.0, 2.0, 2.0]),
            }

        except Exception:
            # Fallback to default
            return {
                "center": np.array([0.0, 0.0, 1.0]),
                "size": np.array([2.0, 2.0, 2.0]),
            }

    def set_operating_mode(self, mode: str) -> None:
        """Set the operating mode: 'dynamic' or 'kinematic'."""
        if mode not in ["dynamic", "kinematic"]:
            msg = f"Invalid operating mode: {mode!r}. Must be 'dynamic' or 'kinematic'."
            raise ValueError(msg)
        self.operating_mode = mode
        # If switching to kinematic, ensure we are in a valid state
        if mode == "kinematic" and self.model is not None:
            mujoco.mj_forward(self.model, self.data)

    def get_dof_info(self) -> list[tuple[str, tuple[float, float], float]]:
        """Get info for all Degrees of Freedom (joints).

        Returns:
            List of (name, (min, max), current_value) tuples.
            Note: This assumes 1-DOF joints (hinge/slide) for simplicity of this UI.
            Complex joints (ball/free) might need special handling.
        """
        if self.model is None or self.data is None:
            return []

        dofs = []
        # Iterate over joints
        for j in range(self.model.njnt):
            # Check joint type
            jtype = self.model.jnt_type[j]
            # mjJNT_HINGE=2, mjJNT_SLIDE=3 are scalar 1-DOF
            if jtype not in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                continue

            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if not name:
                name = f"Joint {j}"

            # Address in qpos
            qpos_adr = self.model.jnt_qposadr[j]

            # Range
            range_min, range_max = self.model.jnt_range[j]
            if range_min == 0 and range_max == 0:
                # Use sensitive defaults based on joint type
                if jtype == mujoco.mjtJoint.mjJNT_HINGE:
                    # Rotational: default to (-pi, pi)
                    range_min, range_max = -np.pi, np.pi
                else:
                    # Slide: default to (-1.0, 1.0) meters
                    range_min, range_max = -1.0, 1.0

            current_val = self.data.qpos[qpos_adr]
            dofs.append((name, (range_min, range_max), current_val))

        return dofs

    def set_joint_qpos(self, joint_name: str, value: float) -> None:
        """Set qpos for a specific 1-DOF joint directly (Kinematic Mode)."""
        if self.model is None or self.data is None:
            return

        # Find joint ID
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid == -1:
            return

        qpos_adr = self.model.jnt_qposadr[jid]
        self.data.qpos[qpos_adr] = value

        # Update kinematics immediately
        mujoco.mj_forward(self.model, self.data)
        self._render_once()

    # -------- Control interface --------

    def set_joint_torque(self, index: int, torque: float) -> None:
        """Set desired constant torque for actuator index (if it exists).

        This is a convenience method that sets constant control.
        For advanced control, use get_control_system().
        """
        if self.control_system is not None:
            self.control_system.set_constant_value(index, torque)
            self.control_system.set_control_type(index, ControlType.CONSTANT)
        elif self.control_vector is not None:
            if 0 <= index < len(self.control_vector):
                self.control_vector[index] = torque

    def get_control_system(self) -> ControlSystem | None:
        """Get the advanced control system."""
        return self.control_system

    def reset_control_system(self) -> None:
        """Reset the control system (reset time to 0)."""
        if self.control_system is not None:
            self.control_system.reset()

    def verify_control_system(self) -> bool:
        """Verify that control system matches model actuator count.

        Returns:
            True if control system is properly initialized and matches model
        """
        if self.model is None:
            return False
        if self.control_system is None:
            return False
        return bool(self.control_system.num_actuators == self.model.nu)

    def set_running(self, running: bool) -> None:
        """Set the simulation running state."""
        self.running = running

    def set_camera(self, camera_name: str) -> None:
        """Set the active camera view."""
        self.camera_name = camera_name

        # Update camera parameters based on preset views
        if camera_name == "side":
            self.camera.azimuth = 90.0
            self.camera.elevation = -20.0
            self.camera.distance = 3.0
            self.camera.lookat[:] = [0, 0, 1]
        elif camera_name == "front":
            self.camera.azimuth = 0.0
            self.camera.elevation = -20.0
            self.camera.distance = 3.0
            self.camera.lookat[:] = [0, 0, 1]
        elif camera_name == "top":
            self.camera.azimuth = 90.0
            self.camera.elevation = -90.0
            self.camera.distance = 4.0
            self.camera.lookat[:] = [0, 0, 1]
        elif camera_name == "follow":
            self.camera.azimuth = 135.0
            self.camera.elevation = -15.0
            self.camera.distance = 2.5
            self.camera.lookat[:] = [0, 0, 1]
        elif camera_name == "down-the-line":
            self.camera.azimuth = 180.0
            self.camera.elevation = -10.0
            self.camera.distance = 3.5
            self.camera.lookat[:] = [0, 0, 1]

        self._render_once()

    def set_torque_visualization(
        self,
        enabled: bool,
        scale: float | None = None,
    ) -> None:
        """Enable/disable torque vector visualization.

        Args:
            enabled: Whether to show torque vectors
            scale: Optional scale factor for arrow length
        """
        self.show_torque_vectors = enabled
        if scale is not None:
            self.torque_scale = scale

    def set_force_visualization(
        self,
        enabled: bool,
        scale: float | None = None,
    ) -> None:
        """Enable/disable force vector visualization.

        Args:
            enabled: Whether to show force vectors
            scale: Optional scale factor for arrow length
        """
        self.show_force_vectors = enabled
        if scale is not None:
            self.force_scale = scale

    def set_ellipsoid_visualization(
        self, mobility_enabled: bool, force_enabled: bool
    ) -> None:
        """Enable/disable mobility and force ellipsoid visualization."""
        self.show_mobility_ellipsoid = mobility_enabled
        self.show_force_ellipsoid = force_enabled

    def set_contact_force_visualization(self, enabled: bool) -> None:
        """Enable/disable contact force visualization."""
        self.show_contact_forces = enabled
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = enabled
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = enabled

    def get_recorder(self) -> SwingRecorder:
        """Get the swing data recorder."""
        return self.recorder

    def get_analyzer(self) -> BiomechanicalAnalyzer | None:
        """Get the biomechanical analyzer."""
        return self.analyzer

    def get_jacobian_and_rank(self) -> dict[str, Any]:
        """Compute Jacobian and Constraint Jacobian Rank.

        Returns:
            Dictionary containing:
                - jacobian_condition: condition number of end-effector jacobian
                - constraint_rank: rank of constraint jacobian
                - nefc: number of active constraints
        """
        if self.model is None or self.data is None:
            return {"jacobian_condition": 0.0, "constraint_rank": 0, "nefc": 0}

        # 1. End Effector Jacobian Condition
        # Use selected body or last body as EE
        body_id = self.model.nbody - 1
        if (
            self.manipulator
            and self.manipulator.selected_body_id is not None
            and self.manipulator.selected_body_id > 0
        ):
            body_id = self.manipulator.selected_body_id

        # Allocate Jacobian arrays
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)

        # Full 6D Jacobian
        J = np.vstack([jacp, jacr])

        # Condition Number (using singular values)
        # Handle cases where nv < 6 (underactuated) or J is singular
        try:
            s = np.linalg.svd(J, compute_uv=False)
            cond = s[0] / s[-1] if s[-1] > 1e-9 else float("inf")
        except Exception:
            cond = 0.0

        # 2. Constraint Jacobian Rank
        nefc = self.data.nefc
        rank = 0
        if nefc > 0:
            # Reshape efc_J to (nefc, nv)
            # data.efc_J is stored as flat array of size (nefc * nv)
            try:
                # MuJoCo C-struct usually stores efc_J as row-major flat array
                Jc = self.data.efc_J.reshape((nefc, self.model.nv))
                rank = np.linalg.matrix_rank(Jc, tol=1e-5)
            except Exception:
                rank = 0

        # Update telemetry if available
        if self.telemetry:
            self.telemetry.add_custom_metric("jacobian_cond", cond)
            self.telemetry.add_custom_metric("constraint_rank", float(rank))
            self.telemetry.add_custom_metric("nefc", float(nefc))

        return {"jacobian_condition": cond, "constraint_rank": rank, "nefc": nefc}

    def set_body_color(self, body_name: str, rgba: list[float]) -> None:
        """Set color of all geoms in a body."""
        if self.model is None:
            return
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            return

        start_geom = self.model.body_geomadr[body_id]
        num_geoms = self.model.body_geomnum[body_id]
        if start_geom >= 0 and num_geoms > 0:
            for i in range(num_geoms):
                geom_id = start_geom + i
                self.model.geom_rgba[geom_id] = rgba

        self._render_once()

    def reset_body_color(self, body_name: str) -> None:
        """Reset body color to default."""
        # For now, just set to a generic gray default
        self.set_body_color(body_name, [0.5, 0.5, 0.5, 1.0])

    def compute_ellipsoids(self) -> None:
        """Compute and draw Mobility and Force Ellipsoids for selected body."""
        if (
            not self.show_mobility_ellipsoid and not self.show_force_ellipsoid
        ) or self.meshcat_adapter is None:
            # Ensure cleared if disabled
            if self.meshcat_adapter:
                self.meshcat_adapter.clear_ellipsoids()
            return

        if self.model is None or self.data is None:
            return

        # Use selected body or last body
        body_id = self.model.nbody - 1
        if (
            self.manipulator
            and self.manipulator.selected_body_id is not None
            and self.manipulator.selected_body_id > 0
        ):
            body_id = self.manipulator.selected_body_id

        # 1. Get Jacobian (Translation only for 3D ellipsoid visualization usually)
        # We focus on Translational Mobility/Force for visualization
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
        J = jacp  # Use translational part (3 x nv)

        # 2. Get Mass Matrix
        # mj_fullM returns dense M (nv x nv)
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)

        # Add damping/regularization to M for invertibility if needed?
        # M should be PD.
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            Minv = np.linalg.pinv(M)

        # 3. Compute Core Matrix
        # Lambda_inv = J * M^-1 * J^T
        Lambda_inv = J @ Minv @ J.T

        # Mobility Ellipsoid (Dynamic): v^T (J M^-1 J^T)^-1 v <= 1 ??
        # Actually, usually defined such that axes are singular values of J*M^(-1/2)
        # Or simply visualize the covariance defined by Lambda_inv?
        # Let's use the standard definitions:
        #
        # Dynamic Manipulability (Velocity) Ellipsoid:
        # Describes the set of velocities realizable with unit energy/torque?
        # Core matrix A = (J M^-1 J^T).
        # The ellipsoid is x^T A^-1 x = 1.
        # Axes aligned with eigenvectors of A. Lengths = sqrt(eigenvalues of A).

        # Force Ellipsoid:
        # Core matrix B = (J M^-1 J^T)^-1 = Lambda.
        # The ellipsoid is f^T B^-1 f = 1 => f^T (J M^-1 J^T) f = 1.
        # Axes aligned with eigenvectors of B. Lengths = sqrt(eigenvalues of B).
        # Note: Eigenvalues of B are 1/eigenvalues of A.
        # So Force axes are sqrt(1/lambda_A) = 1/sqrt(lambda_A).
        # Force ellipsoid is reciprocal to Velocity ellipsoid.

        try:
            eigvals, eigvecs = np.linalg.eigh(Lambda_inv)
            # eigvals are the squares of the axis lengths for the velocity ellipsoid
            # because the ellipsoid is defined by x^T (V D V^T)^-1 x = 1
            # The semi-axes are sqrt(eigvals) * eigvecs.

            body_pos = self.data.xpos[body_id]

            if self.show_mobility_ellipsoid:
                # Radii = sqrt(eigenvalues)
                radii = np.sqrt(np.maximum(eigvals, 1e-6))
                # Scale for visibility?
                radii *= 1.0  # Unit scale?

                # Use Meshcat to draw
                self.meshcat_adapter.draw_ellipsoid(
                    "mobility",
                    body_pos,
                    eigvecs,  # Rotation matrix (columns are axes)
                    radii,
                    color=0x00FF00,  # Green
                    opacity=0.3,
                )

            if self.show_force_ellipsoid:
                # Force ellipsoid radii are reciprocal
                # Radii = 1 / sqrt(eigenvalues) = 1 / radii_mobility
                radii_force = 1.0 / np.sqrt(np.maximum(eigvals, 1e-6))

                # Scale for visibility - force ellipsoids can be huge near singularities
                radii_force = np.clip(radii_force, 0.01, 5.0)

                self.meshcat_adapter.draw_ellipsoid(
                    "force",
                    body_pos,
                    eigvecs,
                    radii_force,
                    color=0xFF0000,  # Red
                    opacity=0.3,
                )

        except Exception as e:
            LOGGER.warning(f"Failed to compute ellipsoids: {e}")

    # -------- Internal stepping / rendering --------

    def _on_timer(self) -> None:
        """Handle timer event for simulation stepping."""
        if self.model is None or self.data is None:
            return

        if self.running:
            # If in Kinematic mode, we don't step physics, but may render/update
            if self.operating_mode == "kinematic":
                self._enforce_interactive_constraints()
                self._render_once()
                return

            # Steps per frame so that physical dt ≈ 1/fps
            steps_per_frame = max(1, int(1.0 / (self.fps * self.model.opt.timestep)))

            for _ in range(steps_per_frame):
                # Update control system time
                if self.control_system is not None:
                    self.control_system.update_time(self.data.time)

                # Apply control - use advanced control system if available
                if self.control_system is not None:
                    # Get joint velocities for damping
                    velocities = (
                        self.data.qvel[: self.model.nu]
                        if self.model.nu <= len(self.data.qvel)
                        else None
                    )
                    control_torques = self.control_system.compute_control_vector(
                        velocities,
                    )
                    self.data.ctrl[:] = control_torques[:]
                elif self.control_vector is not None:
                    # Fallback to simple constant control
                    self.data.ctrl[:] = self.control_vector[:]

                mujoco.mj_step(self.model, self.data)
                if self.telemetry is not None:
                    self.telemetry.record_step(self.data)

            # Record biomechanical data if recording is active
            if self.analyzer is not None and self.recorder.is_recording:
                bio_data = self.analyzer.extract_full_state()
                self.recorder.record_frame(bio_data)

        self._enforce_interactive_constraints()

        # Compute and visualize ellipsoids (done every frame if enabled)
        self.compute_ellipsoids()

        self._render_once()

    def render(self) -> None:  # type: ignore[override]
        """Render the scene immediately."""
        self._render_once()

    def _render_once(self) -> None:
        """Render one frame of the simulation."""
        if self.renderer is None or self.model is None or self.data is None:
            return

        # Update scene with current state (this updates the scene geometry)
        if self.scene is not None:
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.scene_option,
                None,
                self.camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scene,
            )
            # Apply background colors
            self._update_background_colors()

        # Render using camera (renderer uses camera to render the scene)
        if hasattr(self, "camera") and self.camera is not None:
            self.renderer.update_scene(
                self.data,
                camera=self.camera,
                scene_option=self.scene_option,
            )
        else:
            self.renderer.update_scene(
                self.data,
                camera=self.camera_name,
                scene_option=self.scene_option,
            )

        rgb = self.renderer.render()

        # Add force/torque overlays before manipulation overlays
        if self.show_torque_vectors or self.show_force_vectors:
            rgb = self._add_force_torque_overlays(rgb)

        # Add visual overlays for interactive manipulation
        if self.manipulator is not None and (
            self.show_selected_body or self.show_constraints
        ):
            rgb = self._add_manipulation_overlays(rgb)

        # Add coordinate frames and centers of mass
        if self.visible_frames or self.visible_coms:
            rgb = self._add_frame_and_com_overlays(rgb)

        # Update Meshcat
        if self.meshcat_adapter:
            try:
                self.meshcat_adapter.update(self.data)
                self.meshcat_adapter.draw_vectors(
                    self.data,
                    self.show_force_vectors,
                    self.show_torque_vectors,
                    self.force_scale,
                    self.torque_scale,
                )
            except Exception:
                pass  # Avoid crashing main loop if meshcat fails

        # Convert to QImage / QPixmap
        h, w, _ = rgb.shape
        image = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        image = image.copy()  # deep copy
        pixmap = QtGui.QPixmap.fromImage(image)

        self.label.setPixmap(pixmap)

    def _update_background_colors(self) -> None:
        """Update the scene's background colors."""
        if self.scene is not None:
            # Try to set background colors if the scene supports it
            try:
                if hasattr(self.scene, "sky_rgba"):
                    self.scene.sky_rgba[:] = self.sky_color
                if hasattr(self.scene, "ground_rgba"):
                    self.scene.ground_rgba[:] = self.ground_color
            except (AttributeError, TypeError):
                # Background color setting not supported in this MuJoCo version
                pass

    def set_background_color(self, sky_color=None, ground_color=None) -> None:
        """Set the background colors for the scene.

        Args:
            sky_color: RGBA tuple/list for sky color (default: None to keep current)
            ground_color: RGBA tuple/list for ground color (default: None for current)
        """
        if sky_color is not None:
            self.sky_color = np.array(sky_color, dtype=np.float32)
        if ground_color is not None:
            self.ground_color = np.array(ground_color, dtype=np.float32)

        if self.scene is not None:
            self._update_background_colors()
            self._render_once()

    def _add_force_torque_overlays(self, rgb: np.ndarray) -> np.ndarray:
        """Overlay torque/force vectors using screen-space arrows."""
        if self.model is None or self.data is None:
            return rgb

        cv2 = get_cv2()
        if cv2 is None:
            LOGGER.warning("OpenCV not installed, cannot draw force/torque overlays.")
            return rgb

        img = rgb.copy()

        def draw_arrow(
            start: np.ndarray,
            end: np.ndarray,
            color: tuple[int, int, int],
        ) -> None:
            """Draw a screen-space arrow for the provided world-space segment."""
            start_px = self._world_to_screen(start)
            end_px = self._world_to_screen(end)
            if start_px is None or end_px is None:
                return
            cv2.arrowedLine(
                img,
                start_px,
                end_px,
                color,
                thickness=2,
                tipLength=0.2,
            )

        if self.show_torque_vectors:
            self._draw_torque_vectors(draw_arrow)

        if self.show_force_vectors:
            self._draw_force_vectors(draw_arrow)

        return img

    def _draw_torque_vectors(self, draw_arrow_func: Callable) -> None:
        """Helper to draw torque vectors."""
        if self.model is None or self.data is None:
            return

        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            if joint_id < 0 or joint_id >= self.model.njnt:
                continue
            body_id = self.model.jnt_bodyid[joint_id]
            if body_id < 0 or body_id >= self.model.nbody:
                continue
            torque = float(self.data.ctrl[i])
            if abs(torque) < 1e-6:
                continue
            joint_axis = self.data.xaxis[3 * joint_id : 3 * joint_id + 3]
            joint_pos = self.data.xpos[body_id].copy()
            arrow_length = abs(torque) * self.torque_scale
            arrow_dir = joint_axis * np.sign(torque) * arrow_length
            arrow_end = joint_pos + arrow_dir
            color = (255, 0, 0) if torque >= 0 else (0, 0, 255)
            draw_arrow_func(joint_pos, arrow_end, color)

    def _draw_force_vectors(self, draw_arrow_func: Callable) -> None:
        """Helper to draw force vectors (external and internal)."""
        if self.data is None or self.model is None:
            return

        # External forces (Green)
        external_forces = self.data.cfrc_ext.reshape(-1, 6)
        for body_id in range(1, self.model.nbody):
            world_force = external_forces[body_id, 3:6]
            magnitude = float(np.linalg.norm(world_force))
            if magnitude < FORCE_VISUALIZATION_THRESHOLD:  # Threshold
                continue
            body_pos = self.data.xpos[body_id].copy()
            arrow_end = body_pos + world_force * self.force_scale
            draw_arrow_func(body_pos, arrow_end, (0, 255, 0))

        # Internal/Joint reaction forces (Cyan)
        internal_forces = self.data.cfrc_int.reshape(-1, 6)
        for body_id in range(1, self.model.nbody):
            joint_force = internal_forces[body_id, 3:6]
            magnitude = float(np.linalg.norm(joint_force))
            if magnitude < FORCE_VISUALIZATION_THRESHOLD:
                continue
            body_pos = self.data.xpos[body_id].copy()
            arrow_end = body_pos + joint_force * self.force_scale
            # Cyan color (R=0, G=255, B=255)
            draw_arrow_func(body_pos, arrow_end, (0, 255, 255))

    def _add_manipulation_overlays(self, rgb: np.ndarray) -> np.ndarray:
        """Add visual overlays for selected bodies and constraints.

        Args:
            rgb: Rendered image array

        Returns:
            Image array with overlays
        """
        cv2 = get_cv2()
        if cv2 is None:
            # OpenCV not available, return original image
            return rgb

        if self.model is None or self.data is None:
            return rgb

        # Make a copy to avoid modifying original
        img = rgb.copy()

        # Highlight selected body
        if (
            self.show_selected_body
            and self.manipulator is not None
            and self.manipulator.selected_body_id is not None
        ):
            body_pos = self.data.xpos[self.manipulator.selected_body_id].copy()

            # Project 3D position to screen
            screen_pos = self._world_to_screen(body_pos)
            if screen_pos is not None:
                x, y = screen_pos
                # Draw circle around selected body
                cv2.circle(img, (x, y), 20, (0, 255, 255), 3)  # Cyan circle
                cv2.circle(img, (x, y), 3, (0, 255, 255), -1)  # Center dot

                # Draw label
                body_name = self.manipulator.get_body_name(
                    self.manipulator.selected_body_id,
                )
                cv2.putText(
                    img,
                    body_name,
                    (x + 25, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

        # Highlight constrained bodies
        if self.show_constraints and self.manipulator is not None:
            for body_id in self.manipulator.get_constrained_bodies():
                body_pos = self.data.xpos[body_id].copy()
                screen_pos = self._world_to_screen(body_pos)

                if screen_pos is not None:
                    x, y = screen_pos
                    # Draw square for constrained body
                    cv2.rectangle(
                        img,
                        (x - 15, y - 15),
                        (x + 15, y + 15),
                        (255, 0, 255),
                        2,
                    )  # Magenta square
                    cv2.putText(
                        img,
                        "FIXED",
                        (x + 20, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 255),
                        1,
                    )

        return img

    def _world_to_screen(self, world_pos: np.ndarray) -> tuple[int, int] | None:
        """Project 3D world position to 2D screen coordinates.

        Args:
            world_pos: 3D position in world coordinates

        Returns:
            Tuple of (x, y) screen coordinates or None if behind camera
        """
        # Get camera parameters
        cam_azimuth = np.deg2rad(self.camera.azimuth)
        cam_elevation = np.deg2rad(self.camera.elevation)
        cam_distance = self.camera.distance
        cam_lookat = self.camera.lookat.copy()

        # Camera position
        forward = np.array(
            [
                np.cos(cam_elevation) * np.sin(cam_azimuth),
                np.cos(cam_elevation) * np.cos(cam_azimuth),
                np.sin(cam_elevation),
            ],
        )
        cam_pos = cam_lookat - forward * cam_distance

        # Camera frame
        up_world = np.array([0, 0, 1])
        right = np.cross(up_world, forward)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(forward, right)

        # Transform to camera space
        to_point = world_pos - cam_pos
        z = np.dot(to_point, forward)

        # Check if behind camera
        if z < MIN_CAMERA_DEPTH:
            return None

        # Project to screen
        fovy = 45.0
        aspect = self.frame_width / self.frame_height

        x_cam = np.dot(to_point, right)
        y_cam = np.dot(to_point, up)

        # Perspective projection
        x_ndc = x_cam / (z * np.tan(np.deg2rad(fovy / 2)) * aspect)
        y_ndc = y_cam / (z * np.tan(np.deg2rad(fovy / 2)))

        # Convert to screen coordinates
        x_screen = int((x_ndc + 1.0) * 0.5 * self.frame_width)
        y_screen = int((1.0 - y_ndc) * 0.5 * self.frame_height)

        # Clamp to screen bounds
        if 0 <= x_screen < self.frame_width and 0 <= y_screen < self.frame_height:
            return (x_screen, y_screen)

        return None

    def _enforce_interactive_constraints(self) -> None:
        """Ensure interactive constraints remain active during simulation."""
        if self.manipulator is None or self.model is None or self.data is None:
            return
        if not self.manipulator.constraints:
            return

        self.manipulator.enforce_constraints()

    def generate_report(self) -> Any | None:
        """Generate a telemetry report for the current simulation."""

        if self.telemetry is None:
            return None

        return self.telemetry.generate_report()

    def get_manipulator(self) -> InteractiveManipulator | None:
        """Get the interactive manipulator."""
        return self.manipulator

    # -------- Mouse event handling for interactive manipulation --------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        """Handle mouse press for body selection and camera control."""
        modifiers = event.modifiers()
        button = event.button()
        pos = event.position()
        x = int(pos.x())
        y = int(pos.y())

        # Right button or Ctrl+Left logic is handled below
        # (Split to allow context menu on right click)

        # Middle button or Shift+Left = camera translation
        if button == QtCore.Qt.MouseButton.MiddleButton or (
            button == QtCore.Qt.MouseButton.LeftButton
            and modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier
        ):
            x = int(event.position().x())
            y = int(event.position().y())
            self.last_mouse_pos = (x, y)
            self.is_dragging = True
            self.camera_mode = "translate"
            return

        # Left button (no modifiers) = body selection
        if (
            button == QtCore.Qt.MouseButton.LeftButton
            and modifiers == QtCore.Qt.KeyboardModifier.NoModifier
            and self.manipulator is not None
            and self.model is not None
        ):
            # Select body at mouse position
            body_id = self.manipulator.select_body(
                x,
                y,
                self.frame_width,
                self.frame_height,
                self.camera,
            )

            if body_id is not None:
                body_name = self.manipulator.get_body_name(body_id)
                LOGGER.debug("Selected body via mouse: %s (id=%s)", body_name, body_id)
                self._render_once()
            else:
                # Start camera rotation if no body selected
                self.last_mouse_pos = (x, y)
                self.is_dragging = True
                self.camera_mode = "rotate"

        # Right button: Context menu if on body, else Camera Rotate
        if button == QtCore.Qt.MouseButton.RightButton:
            # Check for body under cursor
            if self.manipulator is not None and self.model is not None:
                body_id = self.manipulator.select_body(
                    x,
                    y,
                    self.frame_width,
                    self.frame_height,
                    self.camera,
                )
                if body_id is not None:
                    # Show context menu
                    self.show_context_menu(event.globalPosition().toPoint(), body_id)
                    return

            # Default to rotate if no body selected
            self.last_mouse_pos = (x, y)
            self.is_dragging = True
            self.camera_mode = "rotate"
            return

        # Ctrl+Left = Camera Rotate
        if (
            button == QtCore.Qt.MouseButton.LeftButton
            and modifiers & QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self.last_mouse_pos = (x, y)
            self.is_dragging = True
            self.camera_mode = "rotate"
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent | None) -> None:  # type: ignore[override]
        """Handle mouse move for dragging bodies or camera."""
        if event is None:
            return
        pos = event.position()
        x = int(pos.x())
        y = int(pos.y())

        # Camera manipulation
        if self.is_dragging and self.last_mouse_pos is not None:
            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]

            if self.camera_mode == "rotate":
                # Rotate camera: azimuth and elevation
                sensitivity = 0.5
                self.camera.azimuth -= dx * sensitivity
                self.camera.elevation = np.clip(
                    self.camera.elevation + dy * sensitivity,
                    -90.0,
                    90.0,
                )
                self._render_once()
            elif self.camera_mode == "translate":
                # Translate camera lookat point
                sensitivity = 0.01 * self.camera.distance
                # Calculate right and up vectors
                az_rad = np.deg2rad(self.camera.azimuth)
                el_rad = np.deg2rad(self.camera.elevation)
                forward = np.array(
                    [
                        np.cos(el_rad) * np.sin(az_rad),
                        np.cos(el_rad) * np.cos(az_rad),
                        np.sin(el_rad),
                    ],
                )
                up_world = np.array([0, 0, 1])
                right = np.cross(up_world, forward)
                right = right / (np.linalg.norm(right) + 1e-8)
                up = np.cross(forward, right)

                self.camera.lookat[:] += (
                    -dx * sensitivity * right + dy * sensitivity * up
                )
                self._render_once()

            self.last_mouse_pos = (x, y)
            return

        # Body dragging
        if self.manipulator is not None and self.model is not None:
            if self.manipulator.selected_body_id is not None:
                # Drag body to new position
                success = self.manipulator.drag_to(
                    x,
                    y,
                    self.frame_width,
                    self.frame_height,
                    self.camera,
                )

                if success:
                    self._render_once()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent | None) -> None:  # type: ignore[override]
        """Handle mouse release to end dragging."""
        if event is None:
            return
        # End camera manipulation
        if self.is_dragging:
            self.is_dragging = False
            self.last_mouse_pos = None

        # Handle body selection release
        if (
            self.manipulator is not None
            and self.model is not None
            and event.button() == QtCore.Qt.MouseButton.LeftButton
            and self.manipulator.selected_body_id is not None
        ):
            body_name = self.manipulator.get_body_name(
                self.manipulator.selected_body_id,
            )
            LOGGER.debug("Released body via mouse: %s", body_name)
            self.manipulator.deselect_body()
            self._render_once()

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent | None) -> None:  # type: ignore[override]
        """Handle mouse wheel for camera zoom."""
        if event is None:
            return
        if self.camera is not None:
            # Zoom in/out with smooth scaling
            delta = event.angleDelta().y()
            zoom_factor = 1.0 + (delta / 1200.0)  # Smooth zoom
            self.camera.distance *= zoom_factor
            self.camera.distance = max(0.1, min(50.0, self.camera.distance))
            self._render_once()

        super().wheelEvent(event)

    def set_camera_azimuth(self, azimuth: float) -> None:
        """Set camera azimuth angle in degrees."""
        if self.camera is not None:
            self.camera.azimuth = azimuth
            self._render_once()

    def set_camera_elevation(self, elevation: float) -> None:
        """Set camera elevation angle in degrees."""
        if self.camera is not None:
            self.camera.elevation = np.clip(elevation, -90.0, 90.0)
            self._render_once()

    def set_camera_distance(self, distance: float) -> None:
        """Set camera distance."""
        if self.camera is not None:
            self.camera.distance = np.clip(distance, 0.1, 50.0)
            self._render_once()

    def set_camera_lookat(self, x: float, y: float, z: float) -> None:
        """Set camera lookat point."""
        if self.camera is not None:
            self.camera.lookat[:] = [x, y, z]
            self._render_once()

    def reset_camera(self) -> None:
        """Reset camera to default position."""
        if self.camera is not None:
            mujoco.mjv_defaultCamera(self.camera)
            self.camera.azimuth = 90.0
            self.camera.elevation = -20.0
            self.camera.distance = 3.0
            self.camera.lookat[:] = [0, 0, 1]
            self._render_once()

    def show_context_menu(self, global_pos: QtCore.QPoint, body_id: int) -> None:
        """Show context menu for a body."""
        if self.manipulator is None:
            return

        body_name = self.manipulator.get_body_name(body_id)
        menu = QtWidgets.QMenu(self)
        menu.setTitle(f"Body: {body_name}")

        # Section header
        header = menu.addAction(f"Selected: {body_name}")
        if header is not None:
            header.setEnabled(False)
        menu.addSeparator()

        # Toggle Coordinate Frame
        action_frame = menu.addAction("Show Coordinate System")
        if action_frame is not None:
            action_frame.setCheckable(True)
            action_frame.setChecked(body_id in self.visible_frames)
            action_frame.triggered.connect(
                lambda: self.toggle_frame_visibility(body_id)
            )

        # Toggle Center of Mass
        action_com = menu.addAction("Show Center of Mass")
        if action_com is not None:
            action_com.setCheckable(True)
            action_com.setChecked(body_id in self.visible_coms)
            action_com.triggered.connect(lambda: self.toggle_com_visibility(body_id))

        menu.exec(global_pos)

    def toggle_frame_visibility(self, body_id: int) -> None:
        """Toggle coordinate frame visibility for a body."""
        if body_id in self.visible_frames:
            self.visible_frames.remove(body_id)
        else:
            self.visible_frames.add(body_id)
        self._render_once()

    def toggle_com_visibility(self, body_id: int) -> None:
        """Toggle center of mass visibility for a body."""
        if body_id in self.visible_coms:
            self.visible_coms.remove(body_id)
        else:
            self.visible_coms.add(body_id)
        self._render_once()

    def _add_frame_and_com_overlays(self, rgb: np.ndarray) -> np.ndarray:
        """Overlay coordinate frames and center of mass markers."""
        cv2 = get_cv2()
        if self.model is None or self.data is None or cv2 is None:
            return rgb

        img = rgb.copy()

        def draw_line(
            start_px: tuple[int, int],
            end_px: tuple[int, int],
            color: tuple[int, int, int],
            thickness: int = 2,
        ) -> None:
            """Draw a line on the image."""
            cv2.line(img, start_px, end_px, color, thickness)

        # Draw Frames
        axis_length = 0.2
        for body_id in self.visible_frames:
            pos = self.data.xpos[body_id].copy()
            rot = self.data.xmat[body_id].reshape(3, 3)

            origin = self._world_to_screen(pos)
            if origin is None:
                continue

            # Draw X-axis (Red)
            x_end = pos + rot[:, 0] * axis_length
            x_px = self._world_to_screen(x_end)
            if x_px:
                # RGB: Red axis
                draw_line(origin, x_px, (255, 0, 0))

            # Draw Y-axis (Green)
            y_end = pos + rot[:, 1] * axis_length
            y_px = self._world_to_screen(y_end)
            if y_px:
                draw_line(origin, y_px, (0, 255, 0))

            # Draw Z-axis (Blue)
            z_end = pos + rot[:, 2] * axis_length
            z_px = self._world_to_screen(z_end)
            if z_px:
                draw_line(origin, z_px, (0, 0, 255))

        # Draw COMs
        for body_id in self.visible_coms:
            # xipos is center of mass in global frame
            com_pos = self.data.xipos[body_id].copy()
            screen_pos = self._world_to_screen(com_pos)
            if screen_pos:
                cv2.circle(img, screen_pos, 5, (0, 255, 255), -1)  # Cyan dot
                cv2.circle(img, screen_pos, 7, (0, 0, 0), 1)  # Black outline

                # Label
                cv2.putText(
                    img,
                    "COM",
                    (screen_pos[0] + 10, screen_pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                )

        return img
