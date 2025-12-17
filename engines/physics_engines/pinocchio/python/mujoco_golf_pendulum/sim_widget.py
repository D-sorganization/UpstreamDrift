"""Qt widget encapsulating a MuJoCo simulation and renderer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

import mujoco
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .biomechanics import BiomechanicalAnalyzer, SwingRecorder
from .control_system import ControlSystem, ControlType
from .interactive_manipulation import InteractiveManipulator
from .telemetry import TelemetryRecorder

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

LOGGER = logging.getLogger(__name__)
MIN_CAMERA_DEPTH: Final[float] = 0.1


class MuJoCoSimWidget(QtWidgets.QWidget):
    """Widget that:
    - Holds a MuJoCo model + data
    - Steps the simulation
    - Renders frames with mujoco.Renderer
    - Displays frames in a QLabel
    - Visualizes forces and torques as 3D vectors
    - Records biomechanical data
    """

    def __init__(self, parent=None, width=640, height=480, fps=60) -> None:
        """Initialize the simulation widget."""
        super().__init__(parent)
        self.setMinimumSize(width, height)

        self.fps = fps
        self.frame_width = width
        self.frame_height = height

        self.model = None
        self.data = None
        self.renderer = None
        self.control_vector = None
        self.control_system: ControlSystem | None = None
        self.camera_name = "side"

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

        # Force/torque visualization settings
        self.show_torque_vectors = False
        self.show_force_vectors = False
        self.show_contact_forces = False
        self.torque_scale = 0.01  # Scale factor for torque arrow length
        self.force_scale = 0.1  # Scale factor for force arrow length

        # Visualization for selected bodies and constraints
        self.show_selected_body = True
        self.show_constraints = True

        # MuJoCo scene options for vector rendering
        self.scene_option = mujoco.MjvOption()
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

        # MuJoCo scene for rendering
        self.scene = None

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
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

        # Enable mouse tracking for smooth drag
        self.setMouseTracking(True)
        self.label.setMouseTracking(True)

        # Camera manipulation state
        self.last_mouse_pos = None
        self.camera_mode = "rotate"  # "rotate", "translate", "zoom"
        self.is_dragging = False

        # Timer for stepping and rendering
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(int(1000 / self.fps))

    # -------- MuJoCo setup --------

    def load_model_from_xml(self, xml_string: str) -> None:
        """(Re)load a MuJoCo model from an MJCF XML string."""
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)

        # Create new renderer
        self.renderer = mujoco.Renderer(
            self.model,
            width=self.frame_width,
            height=self.frame_height,
        )

        # Create new scene
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
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

        self.telemetry = TelemetryRecorder(self.model)

        # Control vector length = number of actuators
        self.control_vector = np.zeros(self.model.nu, dtype=np.float64)

        # Initialize advanced control system
        self.control_system = ControlSystem(self.model.nu)

        # Create biomechanical analyzer
        self.analyzer = BiomechanicalAnalyzer(self.model, self.data)

        # Create interactive manipulator
        self.manipulator = InteractiveManipulator(self.model, self.data)

        # Reset state
        mujoco.mj_resetData(self.model, self.data)
        self.reset_state()

        # Reset control system
        if self.control_system is not None:
            self.control_system.reset()

        # Auto-position camera based on model bounds
        self._auto_position_camera()

        self._render_once()

    def load_model_from_file(self, xml_path: str) -> None:
        """(Re)load a MuJoCo model from an MJCF XML file path.

        Args:
            xml_path: Path to the XML model file (absolute or relative to project root)
        """

        # Convert to absolute path if needed
        if not Path(xml_path).is_absolute():
            project_root = Path(__file__).parent.parent.parent
            xml_path = str(project_root / xml_path)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Create new renderer
        self.renderer = mujoco.Renderer(
            self.model,
            width=self.frame_width,
            height=self.frame_height,
        )

        # Create new scene
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
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

        # Control vector length = number of actuators
        self.control_vector = np.zeros(self.model.nu, dtype=np.float64)

        # Initialize advanced control system
        self.control_system = ControlSystem(self.model.nu)

        # Create biomechanical analyzer
        self.analyzer = BiomechanicalAnalyzer(self.model, self.data)

        # Create interactive manipulator
        self.manipulator = InteractiveManipulator(self.model, self.data)

        # Reset state
        mujoco.mj_resetData(self.model, self.data)
        self.reset_state()

        # Reset control system
        if self.control_system is not None:
            self.control_system.reset()

        # Auto-position camera based on model bounds
        self._auto_position_camera()

        self._render_once()

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
            if (
                self.model.njnt > 0
                and self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
                and len(self.data.qpos) >= 3
            ):
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

            # Check all bodies
            for i in range(self.model.nbody):
                pos = self.data.xpos[i]
                min_pos = np.minimum(min_pos, pos)
                max_pos = np.maximum(max_pos, pos)

            # Also check geom positions (they might extend beyond body centers)
            for i in range(self.model.ngeom):
                geom_id = i
                body_id = self.model.geom_bodyid[geom_id]
                if body_id >= 0:
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
        return self.control_system.num_actuators == self.model.nu

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

    # -------- Internal stepping / rendering --------

    def _on_timer(self) -> None:
        """Handle timer event for simulation stepping."""
        if self.model is None or self.data is None:
            return

        if self.running:
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

    def set_background_color(
        self,
        sky_color: np.ndarray | list[float] | None = None,
        ground_color: np.ndarray | list[float] | None = None,
    ) -> None:
        """Set the background colors for the scene.

        Args:
            sky_color: RGBA tuple/list for sky color (default: None to keep current)
            ground_color: RGBA tuple/list for ground color (default: None to keep current)
        """
        if sky_color is not None:
            self.sky_color = np.array(sky_color, dtype=np.float32)
        if ground_color is not None:
            self.ground_color = np.array(ground_color, dtype=np.float32)

        if self.scene is not None:
            self._update_background_colors()
            self._render_once()

    def _add_force_torque_overlays(
        self,
        rgb: np.ndarray,
    ) -> np.ndarray:
        """Overlay torque/force vectors using screen-space arrows."""
        if self.model is None or self.data is None:
            return rgb

        if cv2 is None:
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
                draw_arrow(joint_pos, arrow_end, color)

        if self.show_force_vectors:
            external_forces = self.data.cfrc_ext.reshape(-1, 6)
            for body_id in range(1, self.model.nbody):
                world_force = external_forces[body_id, 3:6]
                magnitude = float(np.linalg.norm(world_force))
                if magnitude < 1e-5:
                    continue
                body_pos = self.data.xpos[body_id].copy()
                arrow_end = body_pos + world_force * self.force_scale
                draw_arrow(body_pos, arrow_end, (0, 255, 0))

        return img

    def _add_manipulation_overlays(self, rgb: np.ndarray) -> np.ndarray:
        """Add visual overlays for selected bodies and constraints.

        Args:
            rgb: Rendered image array

        Returns:
            Image array with overlays
        """
        if cv2 is None:
            # OpenCV not available, return original image
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

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        """Handle mouse press for body selection and camera control."""
        modifiers = event.modifiers()
        button = event.button()
        pos = event.position()
        x = int(pos.x())
        y = int(pos.y())

        # Right button or Ctrl+Left = camera rotation
        if button == QtCore.Qt.MouseButton.RightButton or (
            button == QtCore.Qt.MouseButton.LeftButton
            and modifiers & QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self.last_mouse_pos = (x, y)
            self.is_dragging = True
            self.camera_mode = "rotate"
            return

        # Middle button or Shift+Left = camera translation
        if button == QtCore.Qt.MouseButton.MiddleButton or (
            button == QtCore.Qt.MouseButton.LeftButton
            and modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier
        ):
            self.last_mouse_pos = (x, y)
            self.is_dragging = True
            self.camera_mode = "translate"
            return

        # Left button (no modifiers) = body selection
        if (
            button == QtCore.Qt.MouseButton.LeftButton
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

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        """Handle mouse move for dragging bodies or camera."""
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
        if (
            self.manipulator is not None
            and self.model is not None
            and self.manipulator.selected_body_id is not None
        ):
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

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        """Handle mouse release to end dragging."""
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

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802
        """Handle mouse wheel for camera zoom."""
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
