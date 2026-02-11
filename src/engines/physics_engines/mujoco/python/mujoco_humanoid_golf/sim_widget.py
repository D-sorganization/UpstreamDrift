"""Qt widget encapsulating a MuJoCo simulation and renderer.

Refactored: Rendering/overlay logic lives in ``sim_rendering_mixin.py``.
Camera/mouse interaction lives in ``sim_camera_mixin.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

import mujoco
import numpy as np
from PyQt6 import QtCore, QtWidgets

from src.shared.python.biomechanics.swing_plane_visualization import (
    SwingPlaneVisualizer,
)
from src.shared.python.biomechanics_data import BiomechanicalData
from src.shared.python.logging_config import get_logger

from .biomechanics import BiomechanicalAnalyzer, SwingRecorder
from .control_system import ControlSystem, ControlType
from .interactive_manipulation import InteractiveManipulator
from .meshcat_adapter import MuJoCoMeshcatAdapter
from .physics_engine import MuJoCoPhysicsEngine
from .sim_camera_mixin import SimCameraMixin
from .sim_rendering_mixin import SimRenderingMixin
from .telemetry import TelemetryRecorder

logger = get_logger(__name__)
FORCE_VISUALIZATION_THRESHOLD: Final[float] = 1e-5


class ModelLoaderThread(QtCore.QThread):
    """Worker thread to load MuJoCo models asynchronously."""

    finished_loading = QtCore.pyqtSignal(object, object, str)

    def __init__(self, xml_content: str, is_file: bool = False) -> None:
        super().__init__()
        self.xml_content = xml_content
        self.is_file = is_file

    def run(self) -> None:
        try:
            if self.is_file:
                model = mujoco.MjModel.from_xml_path(self.xml_content)
            else:
                model = mujoco.MjModel.from_xml_string(self.xml_content)

            data = mujoco.MjData(model)
            self.finished_loading.emit(model, data, "")
        except (RuntimeError, ValueError, OSError) as e:
            self.finished_loading.emit(None, None, str(e))


class MuJoCoSimWidget(  # type: ignore[misc]
    SimCameraMixin,
    SimRenderingMixin,
    QtWidgets.QWidget,
):
    """Widget that:
    - Holds a MuJoCo model + data
    - Steps the simulation
    - Renders frames with mujoco.Renderer
    - Displays frames in a QLabel
    - Visualizes forces and torques as 3D vectors
    - Records biomechanical data
    """

    loading_started = QtCore.pyqtSignal()
    loading_finished = QtCore.pyqtSignal(bool)

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

        self.engine = MuJoCoPhysicsEngine()
        self.renderer: mujoco.Renderer | None = None
        self.control_vector: np.ndarray | None = None
        self.control_system: ControlSystem | None = None
        self.camera_name = "side"

        # Visual toggles
        self.show_force_vectors = False
        self.show_torque_vectors = False
        self.force_scale = 0.1
        self.torque_scale = 0.1

        # Advanced Visual Toggles
        self.show_induced_vectors = False
        self.show_cf_vectors = False
        self.induced_vector_source = "gravity"
        self.cf_vector_type = "ztcf_accel"
        self.isolate_forces_visualization = False

        # Ellipsoid Visualization Toggles
        self.show_mobility_ellipsoid = False
        self.show_force_ellipsoid = False

        # Swing Plane & Trajectory Visualization
        self.swing_plane_visualizer = SwingPlaneVisualizer()
        self.show_swing_plane = False
        self.show_club_trajectory = False
        self.show_reference_trajectory = False
        self.swing_plane_body_name = "clubhead"
        self.reference_trajectory: np.ndarray | None = None

        # Real-time Analysis
        self.enable_live_analysis = False
        self.latest_bio_data: BiomechanicalData | None = None

        # Meshcat integration
        self.meshcat_adapter: MuJoCoMeshcatAdapter | None = None
        try:
            self.meshcat_adapter = MuJoCoMeshcatAdapter()
        except (RuntimeError, ValueError, OSError):
            logger.warning("Could not initialize Meshcat adapter")

        self.telemetry: TelemetryRecorder | None = None

        self.running = True

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

        # Per-body visibility
        self.visible_frames: set[int] = set()
        self.visible_coms: set[int] = set()

        # Operating Mode
        self.operating_mode = "dynamic"

        # MuJoCo scene options
        self.scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.scene_option)
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

        self.scene: mujoco.MjvScene | None = None

        # Background color settings (RGBA)
        self.sky_color = np.array([0.2, 0.3, 0.4, 1.0], dtype=np.float32)
        self.ground_color = np.array([0.2, 0.2, 0.2, 1.0], dtype=np.float32)

        # UI
        self.label = QtWidgets.QLabel(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

        self.setMouseTracking(True)
        self.label.setMouseTracking(True)

        # Camera manipulation state
        self.last_mouse_pos: tuple[int, int] | None = None
        self.camera_mode = "rotate"
        self.is_dragging = False

        # Timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(int(1000 / self.fps))

        self.loader_thread: ModelLoaderThread | None = None

    @property
    def model(self) -> mujoco.MjModel | None:
        return self.engine.model

    @model.setter
    def model(self, value: mujoco.MjModel | None) -> None:
        self.engine.model = value

    @property
    def data(self) -> mujoco.MjData | None:
        return self.engine.data

    @data.setter
    def data(self, value: mujoco.MjData | None) -> None:
        self.engine.data = value

    # -------- MuJoCo setup --------

    def load_model_async(self, xml_source: str, is_file: bool = False) -> None:
        """Load a MuJoCo model asynchronously to prevent UI freeze."""
        if self.loader_thread and self.loader_thread.isRunning():
            logger.warning("Model loading already in progress.")
            return

        self.timer.stop()
        self.loading_started.emit()

        self.label.setText("Loading Model...")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.loader_thread = ModelLoaderThread(xml_source, is_file)
        self.loader_thread.finished_loading.connect(self._on_model_loaded_async)
        self.loader_thread.start()

    def _on_model_loaded_async(self, model: Any, data: Any, error_msg: str) -> None:
        """Handle completion of async model loading."""
        if error_msg:
            logger.error("Async load failed: %s", error_msg)
            self.label.setText(f"Error loading model: {error_msg}")
            self.loading_finished.emit(False)
            return

        try:
            self._finalize_model_load(model, data)
            self.loading_finished.emit(True)
        except (RuntimeError, ValueError, OSError) as e:
            logger.error("Finalization failed: %s", e)
            self.label.setText(f"Error initializing renderer: {e}")
            self.loading_finished.emit(False)

    def _finalize_model_load(self, new_model: Any, new_data: Any) -> None:
        """Finalize setup on main thread after model/data creation."""
        new_renderer = mujoco.Renderer(
            new_model,
            width=self.frame_width,
            height=self.frame_height,
        )

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

        self.model = new_model
        self.data = new_data
        self.renderer = new_renderer
        self.scene = new_scene

        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self._auto_position_camera()
        self._update_background_colors()

        self.telemetry = TelemetryRecorder(self.model)
        self.control_system = ControlSystem(self.model.nu)
        self.control_vector = np.zeros(self.model.nu, dtype=np.float64)
        self.analyzer = BiomechanicalAnalyzer(self.model, self.data)
        self.recorder.reset()
        self.latest_bio_data = None
        self.swing_plane_visualizer.reset()
        self.manipulator = InteractiveManipulator(self.model, self.data)
        self.timer.start(int(1000 / self.fps))

    def load_model_from_xml(self, xml_string: str) -> None:
        """(Legacy/Sync) Load a MuJoCo model from an MJCF XML string."""
        self.timer.stop()
        try:
            new_model = mujoco.MjModel.from_xml_string(xml_string)
            new_data = mujoco.MjData(new_model)
            self._finalize_model_load(new_model, new_data)
        except (RuntimeError, TypeError, ValueError) as e:
            logger.error("Sync load failed: %s", e)
            raise

    def load_model_from_file(self, xml_path: str) -> None:
        """(Legacy/Sync) Load from file."""
        self.timer.stop()
        try:
            if not os.path.isabs(xml_path):
                project_root = Path(__file__).parent.parent.parent
                xml_path = str(project_root / xml_path)

            if not os.path.exists(xml_path):
                raise FileNotFoundError(f"Model file not found: {xml_path}")

            new_model = mujoco.MjModel.from_xml_path(xml_path)
            new_data = mujoco.MjData(new_model)
            self._finalize_model_load(new_model, new_data)
        except (FileNotFoundError, OSError) as e:
            logger.error("Sync load failed: %s", e)
            raise

    def reset_state(self) -> None:
        """Set golf-like initial joint angles for all model types."""
        if self.model is None or self.data is None:
            return

        self.engine.reset()
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.engine.forward()

        if self.model.nq == 2:
            self.data.qpos[0] = -1.2
            self.data.qpos[1] = 1.3
        elif self.model.nq == 3:
            self.data.qpos[0] = -1.0
            self.data.qpos[1] = 0.7
            self.data.qpos[2] = 1.2
        elif self.model.nq >= 10:
            if self.model.njnt > 0:
                first_joint_type = self.model.jnt_type[0]
                if first_joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    if len(self.data.qpos) >= 3:
                        self.data.qpos[2] = 0.9
        elif self.model.nq >= 1:
            self.data.qpos[0] = 0.2

        self.data.qvel[:] = 0.0
        self.engine.forward()
        self._render_once()

        if self.telemetry is not None:
            self.telemetry.reset()
            self.telemetry.record_step(self.data)

    def _auto_position_camera(self) -> None:
        """Automatically position camera to view the entire model."""
        if self.model is None or self.data is None:
            return

        bounds = self._compute_model_bounds()
        if bounds is None:
            return

        center = bounds["center"]
        size = bounds["size"]
        max_size = max(size)

        self.camera.lookat[:] = center
        if max_size > 0:
            self.camera.distance = max(2.0, max_size * 2.5)
        else:
            self.camera.distance = 3.0

        self.camera.azimuth = 90.0
        self.camera.elevation = -20.0
        self.camera.distance = np.clip(self.camera.distance, 0.5, 50.0)

    def _compute_model_bounds(self) -> dict | None:
        """Compute bounding box of all geoms in the model."""
        if self.model is None or self.data is None:
            return None

        try:
            mujoco.mj_forward(self.model, self.data)

            min_pos = np.array([np.inf, np.inf, np.inf])
            max_pos = np.array([-np.inf, -np.inf, -np.inf])

            for i in range(1, self.model.nbody):
                pos = self.data.xpos[i]
                min_pos = np.minimum(min_pos, pos)
                max_pos = np.maximum(max_pos, pos)

            for i in range(self.model.ngeom):
                geom_id = i
                body_id = self.model.geom_bodyid[geom_id]
                if body_id > 0:
                    geom_pos = self.data.xpos[body_id].copy()
                    geom_size = self.model.geom_size[geom_id]

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

            if np.all(np.isfinite(min_pos)) and np.all(np.isfinite(max_pos)):
                center = (min_pos + max_pos) / 2.0
                size = max_pos - min_pos
                size = np.maximum(size, [0.5, 0.5, 0.5])
                return {"center": center, "size": size}

            return {
                "center": np.array([0.0, 0.0, 1.0]),
                "size": np.array([2.0, 2.0, 2.0]),
            }

        except (ValueError, TypeError, RuntimeError):
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
        if mode == "kinematic" and self.model is not None:
            mujoco.mj_forward(self.model, self.data)

    def get_dof_info(
        self,
    ) -> list[tuple[str, tuple[float, float], float]]:
        """Get info for all Degrees of Freedom (joints)."""
        if self.model is None or self.data is None:
            return []

        dofs = []
        for j in range(self.model.njnt):
            jtype = self.model.jnt_type[j]
            if jtype not in [
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ]:
                continue

            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if not name:
                name = f"Joint {j}"

            qpos_adr = self.model.jnt_qposadr[j]
            range_min, range_max = self.model.jnt_range[j]
            if range_min == 0 and range_max == 0:
                if jtype == mujoco.mjtJoint.mjJNT_HINGE:
                    range_min, range_max = -np.pi, np.pi
                else:
                    range_min, range_max = -1.0, 1.0

            current_val = self.data.qpos[qpos_adr]
            dofs.append((name, (range_min, range_max), current_val))

        return dofs

    def set_joint_qpos(self, joint_name: str, value: float) -> None:
        """Set qpos for a specific 1-DOF joint directly (Kinematic Mode)."""
        if self.model is None or self.data is None:
            return

        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid == -1:
            return

        qpos_adr = self.model.jnt_qposadr[jid]
        self.data.qpos[qpos_adr] = value
        mujoco.mj_forward(self.model, self.data)
        self._render_once()

    # -------- Control interface --------

    def set_joint_torque(self, index: int, torque: float) -> None:
        if self.control_system is not None:
            self.control_system.set_constant_value(index, torque)
            self.control_system.set_control_type(index, ControlType.CONSTANT)
        elif self.control_vector is not None:
            if 0 <= index < len(self.control_vector):
                self.control_vector[index] = torque

    def get_control_system(self) -> ControlSystem | None:
        return self.control_system

    def reset_control_system(self) -> None:
        if self.control_system is not None:
            self.control_system.reset()

    def verify_control_system(self) -> bool:
        if self.model is None:
            return False
        if self.control_system is None:
            return False
        return bool(self.control_system.num_actuators == self.model.nu)

    def set_running(self, running: bool) -> None:
        self.running = running

    def set_camera(self, camera_name: str) -> None:
        """Set the active camera view."""
        self.camera_name = camera_name
        presets = {
            "side": (90.0, -20.0, 3.0),
            "front": (0.0, -20.0, 3.0),
            "top": (90.0, -90.0, 4.0),
            "follow": (135.0, -15.0, 2.5),
            "down-the-line": (180.0, -10.0, 3.5),
        }
        if camera_name in presets:
            az, el, dist = presets[camera_name]
            self.camera.azimuth = az
            self.camera.elevation = el
            self.camera.distance = dist
            self.camera.lookat[:] = [0, 0, 1]

        self._render_once()

    def set_torque_visualization(
        self, enabled: bool, scale: float | None = None
    ) -> None:
        self.show_torque_vectors = enabled
        if scale is not None:
            self.torque_scale = scale

    def set_force_visualization(
        self, enabled: bool, scale: float | None = None
    ) -> None:
        self.show_force_vectors = enabled
        if scale is not None:
            self.force_scale = scale

    def set_ellipsoid_visualization(
        self, mobility_enabled: bool, force_enabled: bool
    ) -> None:
        self.show_mobility_ellipsoid = mobility_enabled
        self.show_force_ellipsoid = force_enabled

    def set_swing_plane_visualization(
        self,
        show_plane: bool,
        show_trajectory: bool,
        show_reference: bool | None = None,
    ) -> None:
        """Toggle swing plane and trajectory overlay rendering."""
        self.show_swing_plane = show_plane
        self.show_club_trajectory = show_trajectory
        if show_reference is not None:
            self.show_reference_trajectory = show_reference

    def set_reference_trajectory(self, trajectory: np.ndarray | None) -> None:
        self.reference_trajectory = trajectory

    def reset_swing_plane(self) -> None:
        self.swing_plane_visualizer.reset()

    def _record_club_trajectory_point(self) -> None:
        """Record the current clubhead position for swing plane analysis."""
        if self.model is None or self.data is None:
            return
        if not (self.show_swing_plane or self.show_club_trajectory):
            return

        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.swing_plane_body_name,
        )
        if body_id == -1:
            body_id = self.model.nbody - 1

        position = self.data.xpos[body_id].copy()
        timestamp = self.data.time
        self.swing_plane_visualizer.record_trajectory_point(position, timestamp)

    def _update_swing_plane_overlays(self) -> None:  # noqa: PLR0912
        """Push swing plane and trajectory data to meshcat."""
        if self.meshcat_adapter is None:
            return
        if not (self.show_swing_plane or self.show_club_trajectory):
            self.meshcat_adapter.clear_swing_plane()
            return

        if self.show_club_trajectory:
            traj_vis = self.swing_plane_visualizer.get_trajectory_visualization()
            if traj_vis is not None:
                self.meshcat_adapter.draw_trajectory(
                    "club_trajectory", traj_vis.points, color=0x00FF00
                )

        hist = self.swing_plane_visualizer.trajectory_history
        if self.show_swing_plane and len(hist) >= 3:
            if self.model is not None and self.data is not None:
                body_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.swing_plane_body_name,
                )
                if body_id == -1:
                    body_id = self.model.nbody - 1

                if body_id < self.model.nbody:
                    vel_adr = self.model.body_dofadr[body_id]
                else:
                    vel_adr = -1
                if vel_adr >= 0 and vel_adr + 2 < len(self.data.qvel):
                    clubhead_velocity = self.data.qvel[vel_adr : vel_adr + 3]
                else:
                    hist = self.swing_plane_visualizer.trajectory_history
                    if len(hist) >= 2:
                        dt = (
                            self.swing_plane_visualizer.timestamp_history[-1]
                            - self.swing_plane_visualizer.timestamp_history[-2]
                        )
                        if dt > 0:
                            clubhead_velocity = (hist[-1] - hist[-2]) / dt
                        else:
                            clubhead_velocity = np.zeros(3)
                    else:
                        clubhead_velocity = np.zeros(3)

                clubhead_pos = self.data.xpos[body_id].copy()
                parent_id = self.model.body_parentid[body_id]
                grip_pos = self.data.xpos[max(parent_id, 1)].copy()

                if np.linalg.norm(clubhead_velocity) > 1e-3:
                    try:
                        spv = self.swing_plane_visualizer
                        plane_vis = spv.update_instantaneous_plane(
                            clubhead_velocity,
                            grip_pos,
                            clubhead_pos,
                        )
                        self.meshcat_adapter.draw_swing_plane(
                            "instantaneous_plane",
                            plane_vis.vertices,
                            color=0x4488FF,
                            opacity=0.25,
                        )
                        self.meshcat_adapter.draw_arrow_line(
                            "plane_normal",
                            plane_vis.normal_arrow_start,
                            plane_vis.normal_arrow_end,
                            color=0x4488FF,
                        )
                    except (
                        RuntimeError,
                        ValueError,
                        AttributeError,
                    ):
                        pass

        if self.show_reference_trajectory and self.reference_trajectory is not None:
            self.meshcat_adapter.draw_trajectory(
                "reference_trajectory",
                self.reference_trajectory,
                color=0xFF8844,
            )

    def set_advanced_vector_visualization(
        self,
        induced_enabled: bool,
        induced_source: str,
        cf_enabled: bool,
        cf_type: str,
    ) -> None:
        self.show_induced_vectors = induced_enabled
        self.induced_vector_source = induced_source
        self.show_cf_vectors = cf_enabled
        self.cf_vector_type = cf_type

    def set_contact_force_visualization(self, enabled: bool) -> None:
        self.show_contact_forces = enabled
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = enabled
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = enabled

    def set_isolate_forces_visualization(self, enabled: bool) -> None:
        self.isolate_forces_visualization = enabled

    def get_recorder(self) -> SwingRecorder:
        return self.recorder

    def get_analyzer(self) -> BiomechanicalAnalyzer | None:
        return self.analyzer

    def get_jacobian_and_rank(self) -> dict[str, Any]:
        """Compute Jacobian and Constraint Jacobian Rank."""
        if self.model is None or self.data is None:
            return {
                "jacobian_condition": 0.0,
                "constraint_rank": 0,
                "nefc": 0,
            }

        body_id = self.model.nbody - 1
        if (
            self.manipulator
            and self.manipulator.selected_body_id is not None
            and self.manipulator.selected_body_id > 0
        ):
            body_id = self.manipulator.selected_body_id

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
        J = np.vstack([jacp, jacr])

        try:
            s = np.linalg.svd(J, compute_uv=False)
            cond = s[0] / s[-1] if s[-1] > 1e-9 else float("inf")
        except (ValueError, TypeError, RuntimeError):
            cond = 0.0

        nefc = self.data.nefc
        rank = 0
        if nefc > 0:
            try:
                Jc = self.data.efc_J.reshape((nefc, self.model.nv))
                rank = np.linalg.matrix_rank(Jc, tol=1e-5)
            except (ValueError, TypeError, RuntimeError):
                rank = 0

        if self.telemetry:
            self.telemetry.add_custom_metric("jacobian_cond", cond)
            self.telemetry.add_custom_metric("constraint_rank", float(rank))
            self.telemetry.add_custom_metric("nefc", float(nefc))

        return {
            "jacobian_condition": cond,
            "constraint_rank": rank,
            "nefc": nefc,
        }

    def set_body_color(self, body_name: str, rgba: list[float]) -> None:
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
        self.set_body_color(body_name, [0.5, 0.5, 0.5, 1.0])

    def compute_ellipsoids(self) -> None:
        if (
            not self.show_mobility_ellipsoid and not self.show_force_ellipsoid
        ) or self.meshcat_adapter is None:
            if self.meshcat_adapter:
                self.meshcat_adapter.clear_ellipsoids()
            return

        if self.model is None or self.data is None:
            return

        body_id = self.model.nbody - 1
        if (
            self.manipulator
            and self.manipulator.selected_body_id is not None
            and self.manipulator.selected_body_id > 0
        ):
            body_id = self.manipulator.selected_body_id

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
        J = jacp

        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)

        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            Minv = np.linalg.pinv(M)

        Lambda_inv = J @ Minv @ J.T

        try:
            eigvals, eigvecs = np.linalg.eigh(Lambda_inv)
            body_pos = self.data.xpos[body_id]

            if self.show_mobility_ellipsoid:
                radii = np.sqrt(np.maximum(eigvals, 1e-6))
                self.meshcat_adapter.draw_ellipsoid(
                    "mobility",
                    body_pos,
                    eigvecs,
                    radii,
                    color=0x00FF00,
                    opacity=0.3,
                )

            if self.show_force_ellipsoid:
                radii_force = 1.0 / np.sqrt(np.maximum(eigvals, 1e-6))
                radii_force = np.clip(radii_force, 0.01, 5.0)
                self.meshcat_adapter.draw_ellipsoid(
                    "force",
                    body_pos,
                    eigvecs,
                    radii_force,
                    color=0xFF0000,
                    opacity=0.3,
                )

        except (ValueError, TypeError, RuntimeError) as e:
            logger.warning("Failed to compute ellipsoids: %s", e)

    # -------- Internal stepping --------

    def _on_timer(self) -> None:  # noqa: PLR0912, PLR0915
        """Handle timer event for simulation stepping."""
        if self.model is None or self.data is None:
            return

        if self.running:
            if self.operating_mode == "kinematic":
                self._enforce_interactive_constraints()
                self._render_once()
                return

            steps_per_frame = max(1, int(1.0 / (self.fps * self.model.opt.timestep)))

            for _ in range(steps_per_frame):
                if self.control_system is not None:
                    self.control_system.update_time(self.data.time)

                if self.control_system is not None:
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
                    self.data.ctrl[:] = self.control_vector[:]

                mujoco.mj_step(self.model, self.data)
                if self.telemetry is not None:
                    self.telemetry.record_step(self.data)

            config_requests_analysis = False
            config_selected_actuator = None

            if hasattr(self.recorder, "analysis_config") and isinstance(
                self.recorder.analysis_config, dict
            ):
                cfg = self.recorder.analysis_config
                if (
                    cfg.get("ztcf")
                    or cfg.get("zvcf")
                    or cfg.get("track_drift")
                    or cfg.get("track_total_control")
                ):
                    config_requests_analysis = True

                sources = cfg.get("induced_accel_sources", [])
                if sources:
                    config_requests_analysis = True
                    for src in sources:
                        if src not in [
                            "gravity",
                            "velocity",
                            "total",
                            "actuator",
                        ]:
                            config_selected_actuator = str(src)
                            break

            should_compute = self.enable_live_analysis or config_requests_analysis

            selected_actuator = config_selected_actuator
            if selected_actuator is None:
                if self.show_induced_vectors and self.induced_vector_source not in [
                    "gravity",
                    "actuator",
                ]:
                    selected_actuator = self.induced_vector_source

            if self.analyzer is not None and self.recorder.is_recording:
                bio_data = self.analyzer.extract_full_state(
                    selected_actuator_name=selected_actuator,
                    compute_advanced_metrics=should_compute,
                )
                self.recorder.record_frame(bio_data)
                self.latest_bio_data = bio_data
            elif should_compute and self.analyzer:
                self.latest_bio_data = self.analyzer.extract_full_state(
                    selected_actuator_name=selected_actuator,
                    compute_advanced_metrics=True,
                )

        self._enforce_interactive_constraints()
        self.compute_ellipsoids()
        self._record_club_trajectory_point()
        self._update_swing_plane_overlays()
        self._render_once()

    def _enforce_interactive_constraints(self) -> None:
        if self.manipulator is None or self.model is None or self.data is None:
            return
        if not self.manipulator.constraints:
            return

        self.manipulator.enforce_constraints()

    def generate_report(self) -> Any | None:
        if self.telemetry is None:
            return None
        return self.telemetry.generate_report()

    def get_manipulator(self) -> InteractiveManipulator | None:
        return self.manipulator
