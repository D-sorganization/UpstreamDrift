#!/usr/bin/env python3
"""
Golf Swing Visualizer - Advanced Camera System
Sophisticated camera controls with smooth animations, presets, and cinematic features
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from PyQt6.QtCore import QEasingCurve, QObject, QTimer, pyqtSignal

# ============================================================================
# CAMERA DATA STRUCTURES
# ============================================================================


class CameraMode(Enum):
    """Camera operation modes"""

    ORBIT = "orbit"
    FLY = "fly"
    FOLLOW = "follow"
    CINEMATIC = "cinematic"


class CameraPreset(Enum):
    """Predefined camera positions"""

    DEFAULT = "default"
    SIDE_VIEW = "side_view"
    TOP_DOWN = "top_down"
    FRONT_VIEW = "front_view"
    BEHIND_GOLFER = "behind_golfer"
    IMPACT_ZONE = "impact_zone"
    FOLLOW_THROUGH = "follow_through"


@dataclass
class CameraState:
    """Complete camera state"""

    position: np.ndarray = field(
        default_factory=lambda: np.array([3.0, 2.0, 3.0], dtype=np.float32)
    )
    target: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )
    up: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )
    fov: float = 45.0
    near_plane: float = 0.1
    far_plane: float = 100.0

    # Spherical coordinates (for orbit mode)
    distance: float = 5.0
    azimuth: float = 45.0  # degrees
    elevation: float = 20.0  # degrees

    # Animation properties
    is_animating: bool = False
    animation_time: float = 0.0
    animation_duration: float = 1.0


@dataclass
class CameraKeyframe:
    """Camera keyframe for cinematic animations"""

    time: float
    state: CameraState
    easing: QEasingCurve.Type = QEasingCurve.Type.InOutCubic


@dataclass
class CameraConstraints:
    """Camera movement constraints"""

    min_distance: float = 0.5
    max_distance: float = 20.0
    min_elevation: float = -89.0
    max_elevation: float = 89.0
    min_fov: float = 10.0
    max_fov: float = 120.0

    # Movement limits (world coordinates)
    position_bounds: tuple[np.ndarray, np.ndarray] | None = None


# ============================================================================
# SMOOTH ANIMATION UTILITIES
# ============================================================================


class SmoothAnimator:
    """Smooth interpolation for camera animations"""

    @staticmethod
    def ease_in_out_cubic(t: float) -> float:
        """Cubic ease-in-out interpolation"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2

    @staticmethod
    def ease_in_out_quart(t: float) -> float:
        """Quartic ease-in-out interpolation"""
        if t < 0.5:
            return 8 * t * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 4) / 2

    @staticmethod
    def ease_elastic_out(t: float) -> float:
        """Elastic ease-out for dramatic effects"""
        c4 = (2 * math.pi) / 3
        if t == 0:
            return 0
        elif t == 1:
            return 1
        else:
            return pow(2, -10 * t) * math.sin((t * 10 - 0.75) * c4) + 1

    @staticmethod
    def interpolate_vectors(
        start: np.ndarray,
        end: np.ndarray,
        t: float,
        easing_func: Callable | None = None,
    ) -> np.ndarray:
        """Interpolate between two vectors with optional easing"""
        if easing_func:
            t = easing_func(np.clip(t, 0.0, 1.0))
        else:
            t = np.clip(t, 0.0, 1.0)

        return start + (end - start) * t

    @staticmethod
    def spherical_interpolation(
        start_spherical: tuple[float, float, float],
        end_spherical: tuple[float, float, float],
        t: float,
        easing_func: Callable | None = None,
    ) -> tuple[float, float, float]:
        """Spherical interpolation for smooth orbit camera movement"""
        if easing_func:
            t = easing_func(np.clip(t, 0.0, 1.0))
        else:
            t = np.clip(t, 0.0, 1.0)

        start_dist, start_azim, start_elev = start_spherical
        end_dist, end_azim, end_elev = end_spherical

        # Handle azimuth wrap-around (shortest path)
        azim_diff = end_azim - start_azim
        if azim_diff > 180:
            azim_diff -= 360
        elif azim_diff < -180:
            azim_diff += 360

        new_dist = start_dist + (end_dist - start_dist) * t
        new_azim = start_azim + azim_diff * t
        new_elev = start_elev + (end_elev - start_elev) * t

        return new_dist, new_azim, new_elev


# ============================================================================
# ADVANCED CAMERA CONTROLLER
# ============================================================================


class CameraController(QObject):
    """Advanced camera controller with multiple modes and smooth animations"""

    # Signals
    cameraChanged = pyqtSignal()
    animationFinished = pyqtSignal()
    modeChanged = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        # Core state
        self.current_state = CameraState()
        self.target_state = CameraState()
        self.constraints = CameraConstraints()

        # Mode and behavior
        self.mode = CameraMode.ORBIT
        self.auto_frame_data = True
        self.smooth_transitions = True

        # Animation system
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._update_animation)
        self.animation_start_time = 0.0
        self.animation_start_state = CameraState()

        # Interaction settings
        self.mouse_sensitivity = 0.5
        self.zoom_sensitivity = 0.1
        self.pan_sensitivity = 0.001
        self.inertia_enabled = True
        self.inertia_damping = 0.95

        # Velocity tracking for inertia
        self.velocity_azimuth = 0.0
        self.velocity_elevation = 0.0
        self.velocity_zoom = 0.0
        self.velocity_pan = np.array([0.0, 0.0], dtype=np.float32)

        # Cinematic features
        self.keyframes: list[CameraKeyframe] = []
        self.cinematic_time = 0.0
        self.cinematic_duration = 10.0
        self.cinematic_loop = False

        # Preset configurations
        self._setup_presets()

        print("📷 Advanced camera controller initialized")

    def _setup_presets(self):
        """Setup predefined camera presets"""
        self.presets = {
            CameraPreset.DEFAULT: CameraState(
                distance=5.0, azimuth=45.0, elevation=20.0, fov=45.0
            ),
            CameraPreset.SIDE_VIEW: CameraState(
                distance=4.0, azimuth=90.0, elevation=0.0, fov=50.0
            ),
            CameraPreset.TOP_DOWN: CameraState(
                distance=3.0, azimuth=0.0, elevation=89.0, fov=60.0
            ),
            CameraPreset.FRONT_VIEW: CameraState(
                distance=4.0, azimuth=0.0, elevation=0.0, fov=50.0
            ),
            CameraPreset.BEHIND_GOLFER: CameraState(
                distance=3.0, azimuth=180.0, elevation=10.0, fov=45.0
            ),
            CameraPreset.IMPACT_ZONE: CameraState(
                distance=2.0, azimuth=45.0, elevation=-10.0, fov=35.0
            ),
            CameraPreset.FOLLOW_THROUGH: CameraState(
                distance=3.5, azimuth=135.0, elevation=15.0, fov=40.0
            ),
        }

    # ========================================================================
    # CORE CAMERA OPERATIONS
    # ========================================================================

    def get_view_matrix(self) -> np.ndarray:
        """Get current view matrix"""
        eye = self._spherical_to_cartesian()
        target = self.current_state.target
        up = self.current_state.up

        return self._create_look_at_matrix(eye, target, up)

    def get_projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        """Get current projection matrix"""
        return self._create_perspective_matrix(
            np.radians(self.current_state.fov),
            aspect_ratio,
            self.current_state.near_plane,
            self.current_state.far_plane,
        )

    def get_camera_position(self) -> np.ndarray:
        """Get current camera position in world coordinates"""
        return self._spherical_to_cartesian()

    def _spherical_to_cartesian(self) -> np.ndarray:
        """Convert spherical coordinates to Cartesian position"""
        distance = self.current_state.distance
        azimuth_rad = np.radians(self.current_state.azimuth)
        elevation_rad = np.radians(self.current_state.elevation)

        x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = distance * np.sin(elevation_rad)
        z = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)

        return self.current_state.target + np.array([x, y, z], dtype=np.float32)

    def _cartesian_to_spherical(
        self, position: np.ndarray
    ) -> tuple[float, float, float]:
        """Convert Cartesian position to spherical coordinates"""
        offset = position - self.current_state.target
        distance = np.linalg.norm(offset)

        if distance < 1e-6:
            return (
                self.current_state.distance,
                self.current_state.azimuth,
                self.current_state.elevation,
            )

        elevation = np.degrees(np.arcsin(np.clip(offset[1] / distance, -1, 1)))
        azimuth = np.degrees(np.arctan2(offset[2], offset[0]))

        return distance, azimuth, elevation

    @staticmethod
    def _create_look_at_matrix(
        eye: np.ndarray, target: np.ndarray, up: np.ndarray
    ) -> np.ndarray:
        """Create look-at view matrix"""
        f = target - eye
        f_norm = np.linalg.norm(f)
        if f_norm > 1e-6:
            f = f / f_norm
        else:
            f = np.array([0, 0, -1], dtype=np.float32)

        s = np.cross(f, up)
        s_norm = np.linalg.norm(s)
        if s_norm > 1e-6:
            s = s / s_norm
        else:
            s = np.array([1, 0, 0], dtype=np.float32)

        u = np.cross(s, f)

        result = np.eye(4, dtype=np.float32)
        result[0, 0:3] = s
        result[1, 0:3] = u
        result[2, 0:3] = -f
        result[0, 3] = -np.dot(s, eye)
        result[1, 3] = -np.dot(u, eye)
        result[2, 3] = np.dot(f, eye)

        return result

    @staticmethod
    def _create_perspective_matrix(
        fov: float, aspect: float, near: float, far: float
    ) -> np.ndarray:
        """Create perspective projection matrix"""
        f = 1.0 / np.tan(fov / 2.0)

        result = np.zeros((4, 4), dtype=np.float32)
        result[0, 0] = f / aspect
        result[1, 1] = f
        result[2, 2] = (far + near) / (near - far)
        result[2, 3] = (2 * far * near) / (near - far)
        result[3, 2] = -1.0

        return result

    # ========================================================================
    # INTERACTION HANDLING
    # ========================================================================

    def handle_mouse_orbit(self, dx: float, dy: float):
        """Handle mouse orbital movement"""
        if self.mode != CameraMode.ORBIT:
            return

        # Apply sensitivity and update spherical coordinates
        azimuth_delta = dx * self.mouse_sensitivity
        elevation_delta = -dy * self.mouse_sensitivity

        self.current_state.azimuth += azimuth_delta
        self.current_state.elevation = np.clip(
            self.current_state.elevation + elevation_delta,
            self.constraints.min_elevation,
            self.constraints.max_elevation,
        )

        # Update velocity for inertia
        if self.inertia_enabled:
            self.velocity_azimuth = azimuth_delta
            self.velocity_elevation = elevation_delta

        self._apply_constraints()
        self.cameraChanged.emit()

    def handle_mouse_pan(self, dx: float, dy: float):
        """Handle mouse panning movement"""
        if self.mode not in [CameraMode.ORBIT, CameraMode.FLY]:
            return

        # Calculate camera right and up vectors
        eye = self._spherical_to_cartesian()
        forward = self.current_state.target - eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, self.current_state.up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        # Apply pan movement
        pan_speed = self.current_state.distance * self.pan_sensitivity
        pan_offset = right * dx * pan_speed + up * dy * pan_speed

        self.current_state.target += pan_offset

        # Update velocity for inertia
        if self.inertia_enabled:
            self.velocity_pan = np.array([dx, dy], dtype=np.float32) * pan_speed

        self.cameraChanged.emit()

    def handle_mouse_zoom(self, delta: float):
        """Handle mouse wheel zoom"""
        zoom_factor = 1.0 + (delta * self.zoom_sensitivity)
        new_distance = self.current_state.distance / zoom_factor

        self.current_state.distance = np.clip(
            new_distance, self.constraints.min_distance, self.constraints.max_distance
        )

        # Update velocity for inertia
        if self.inertia_enabled:
            self.velocity_zoom = delta * self.zoom_sensitivity * 0.1

        self.cameraChanged.emit()

    def update_inertia(self):
        """Update camera movement with inertia"""
        if not self.inertia_enabled:
            return

        # Apply inertial movement
        if abs(self.velocity_azimuth) > 0.01:
            self.current_state.azimuth += self.velocity_azimuth
            self.velocity_azimuth *= self.inertia_damping

        if abs(self.velocity_elevation) > 0.01:
            self.current_state.elevation = np.clip(
                self.current_state.elevation + self.velocity_elevation,
                self.constraints.min_elevation,
                self.constraints.max_elevation,
            )
            self.velocity_elevation *= self.inertia_damping

        if abs(self.velocity_zoom) > 0.001:
            zoom_factor = 1.0 + self.velocity_zoom
            new_distance = self.current_state.distance / zoom_factor
            self.current_state.distance = np.clip(
                new_distance,
                self.constraints.min_distance,
                self.constraints.max_distance,
            )
            self.velocity_zoom *= self.inertia_damping

        if np.linalg.norm(self.velocity_pan) > 0.001:
            # Calculate camera vectors for pan
            eye = self._spherical_to_cartesian()
            forward = self.current_state.target - eye
            forward = forward / np.linalg.norm(forward)

            right = np.cross(forward, self.current_state.up)
            right = right / np.linalg.norm(right)

            up = np.cross(right, forward)

            pan_offset = right * self.velocity_pan[0] + up * self.velocity_pan[1]
            self.current_state.target += pan_offset
            self.velocity_pan *= self.inertia_damping

        # Check if any velocity is significant enough to continue
        total_velocity = (
            abs(self.velocity_azimuth)
            + abs(self.velocity_elevation)
            + abs(self.velocity_zoom)
            + np.linalg.norm(self.velocity_pan)
        )

        if total_velocity > 0.01:
            self.cameraChanged.emit()

    # ========================================================================
    # CAMERA PRESETS AND ANIMATION
    # ========================================================================

    def set_preset(
        self, preset: CameraPreset, animate: bool = True, duration: float = 1.0
    ):
        """Set camera to predefined preset"""
        if preset not in self.presets:
            print(f"Warning: Preset {preset} not found")
            return

        target_state = self.presets[preset]

        if animate and self.smooth_transitions:
            self.animate_to_state(target_state, duration)
        else:
            self._copy_state(target_state, self.current_state)
            self._apply_constraints()
            self.cameraChanged.emit()

        print(f"📷 Camera preset: {preset.value}")

    def animate_to_state(
        self,
        target_state: CameraState,
        duration: float = 1.0,
        easing: Callable | None = None,
    ):
        """Animate camera to target state"""
        # Stop any current animation
        self.stop_animation()

        # Setup animation
        self._copy_state(self.current_state, self.animation_start_state)
        self._copy_state(target_state, self.target_state)

        self.current_state.animation_duration = duration
        self.animation_start_time = time.time()

        # Start animation timer
        self.animation_timer.start(16)  # ~60 FPS
        self.current_state.is_animating = True

        print(f"📷 Animating camera over {duration:.1f}s")

    def _update_animation(self):
        """Update ongoing camera animation"""
        if not self.current_state.is_animating:
            return

        # Calculate animation progress
        elapsed = time.time() - self.animation_start_time
        progress = elapsed / self.current_state.animation_duration

        if progress >= 1.0:
            # Animation complete
            progress = 1.0
            self.stop_animation()

        # Apply easing
        eased_progress = SmoothAnimator.ease_in_out_cubic(progress)

        # Interpolate spherical coordinates
        start_spherical = (
            self.animation_start_state.distance,
            self.animation_start_state.azimuth,
            self.animation_start_state.elevation,
        )
        end_spherical = (
            self.target_state.distance,
            self.target_state.azimuth,
            self.target_state.elevation,
        )

        (
            new_distance,
            new_azimuth,
            new_elevation,
        ) = SmoothAnimator.spherical_interpolation(
            start_spherical, end_spherical, eased_progress
        )

        self.current_state.distance = new_distance
        self.current_state.azimuth = new_azimuth
        self.current_state.elevation = new_elevation

        # Interpolate target and other properties
        self.current_state.target = SmoothAnimator.interpolate_vectors(
            self.animation_start_state.target, self.target_state.target, eased_progress
        )

        self.current_state.fov = (
            self.animation_start_state.fov
            + (self.target_state.fov - self.animation_start_state.fov) * eased_progress
        )

        self._apply_constraints()
        self.cameraChanged.emit()

        if progress >= 1.0:
            self.animationFinished.emit()

    def stop_animation(self):
        """Stop any ongoing animation"""
        self.animation_timer.stop()
        self.current_state.is_animating = False

    def _copy_state(self, source: CameraState, destination: CameraState):
        """Copy camera state"""
        destination.position = source.position.copy()
        destination.target = source.target.copy()
        destination.up = source.up.copy()
        destination.fov = source.fov
        destination.distance = source.distance
        destination.azimuth = source.azimuth
        destination.elevation = source.elevation

    # ========================================================================
    # CINEMATIC CAMERA SYSTEM
    # ========================================================================

    def add_keyframe(
        self,
        time: float,
        state: CameraState | None = None,
        easing: QEasingCurve.Type = QEasingCurve.Type.InOutCubic,
    ):
        """Add a keyframe for cinematic animation"""
        if state is None:
            state = CameraState()
            self._copy_state(self.current_state, state)

        keyframe = CameraKeyframe(time=time, state=state, easing=easing)

        # Insert in chronological order
        inserted = False
        for i, existing in enumerate(self.keyframes):
            if existing.time > time:
                self.keyframes.insert(i, keyframe)
                inserted = True
                break

        if not inserted:
            self.keyframes.append(keyframe)

        print(f"📷 Added keyframe at {time:.1f}s")

    def clear_keyframes(self):
        """Clear all cinematic keyframes"""
        self.keyframes.clear()
        print("📷 Cleared all keyframes")

    def start_cinematic_playback(
        self, duration: float | None = None, loop: bool = False
    ):
        """Start cinematic camera playback"""
        if not self.keyframes:
            print("Warning: No keyframes defined for cinematic playback")
            return

        self.mode = CameraMode.CINEMATIC
        self.cinematic_time = 0.0
        self.cinematic_loop = loop

        if duration:
            self.cinematic_duration = duration
        else:
            self.cinematic_duration = max(kf.time for kf in self.keyframes)

        self.animation_timer.start(16)  # ~60 FPS
        self.modeChanged.emit(self.mode.value)

        print(f"📷 Started cinematic playback: {self.cinematic_duration:.1f}s")

    def update_cinematic_camera(self, time_delta: float):
        """Update camera position during cinematic playback"""
        if self.mode != CameraMode.CINEMATIC:
            return

        self.cinematic_time += time_delta

        # Handle looping
        if self.cinematic_time > self.cinematic_duration:
            if self.cinematic_loop:
                self.cinematic_time = 0.0
            else:
                self.stop_cinematic_playback()
                return

        # Find surrounding keyframes
        prev_keyframe = None
        next_keyframe = None

        for keyframe in self.keyframes:
            if keyframe.time <= self.cinematic_time:
                prev_keyframe = keyframe
            elif keyframe.time > self.cinematic_time and next_keyframe is None:
                next_keyframe = keyframe
                break

        if prev_keyframe and next_keyframe:
            # Interpolate between keyframes
            time_span = next_keyframe.time - prev_keyframe.time
            local_progress = (self.cinematic_time - prev_keyframe.time) / time_span

            # Apply easing from next keyframe
            if next_keyframe.easing == QEasingCurve.Type.InOutCubic:
                eased_progress = SmoothAnimator.ease_in_out_cubic(local_progress)
            elif next_keyframe.easing == QEasingCurve.Type.InOutQuart:
                eased_progress = SmoothAnimator.ease_in_out_quart(local_progress)
            elif next_keyframe.easing == QEasingCurve.Type.OutElastic:
                eased_progress = SmoothAnimator.ease_elastic_out(local_progress)
            else:
                eased_progress = local_progress

            # Interpolate all camera properties
            self._interpolate_between_states(
                prev_keyframe.state, next_keyframe.state, eased_progress
            )

        elif prev_keyframe:
            # Use the last keyframe
            self._copy_state(prev_keyframe.state, self.current_state)

        self.cameraChanged.emit()

    def stop_cinematic_playback(self):
        """Stop cinematic camera playback"""
        self.mode = CameraMode.ORBIT
        self.animation_timer.stop()
        self.modeChanged.emit(self.mode.value)
        print("📷 Stopped cinematic playback")

    def _interpolate_between_states(
        self, state1: CameraState, state2: CameraState, t: float
    ):
        """Interpolate between two camera states"""
        # Spherical interpolation
        spherical1 = (state1.distance, state1.azimuth, state1.elevation)
        spherical2 = (state2.distance, state2.azimuth, state2.elevation)

        (
            new_distance,
            new_azimuth,
            new_elevation,
        ) = SmoothAnimator.spherical_interpolation(spherical1, spherical2, t)

        self.current_state.distance = new_distance
        self.current_state.azimuth = new_azimuth
        self.current_state.elevation = new_elevation

        # Linear interpolation for other properties
        self.current_state.target = SmoothAnimator.interpolate_vectors(
            state1.target, state2.target, t
        )

        self.current_state.fov = state1.fov + (state2.fov - state1.fov) * t

    # ========================================================================
    # AUTO-FRAMING AND SMART FEATURES
    # ========================================================================

    def frame_data(self, data_points: list[np.ndarray], margin: float = 1.5):
        """Automatically frame camera to view all data points"""
        if not data_points:
            return

        # Filter valid points
        valid_points = [p for p in data_points if np.isfinite(p).all()]
        if not valid_points:
            return

        # Calculate bounding box
        all_points = np.array(valid_points)
        center = np.mean(all_points, axis=0)
        extents = np.max(all_points, axis=0) - np.min(all_points, axis=0)
        max_extent = np.max(extents)

        # Set target to center of data
        self.current_state.target = center.astype(np.float32)

        # Calculate appropriate distance
        fov_rad = np.radians(self.current_state.fov)
        required_distance = (max_extent * margin) / (2 * np.tan(fov_rad / 2))

        self.current_state.distance = np.clip(
            required_distance,
            self.constraints.min_distance,
            self.constraints.max_distance,
        )

        self.cameraChanged.emit()
        print(
            f"📷 Auto-framed data: center={center}, "
            f"distance={self.current_state.distance:.2f}"
        )

    def follow_point(self, point: np.ndarray, smooth_factor: float = 0.1):
        """Smoothly follow a moving point"""
        if self.mode != CameraMode.FOLLOW:
            return

        if not np.isfinite(point).all():
            return

        # Smooth interpolation toward target point
        current_target = self.current_state.target
        new_target = current_target + (point - current_target) * smooth_factor

        self.current_state.target = new_target.astype(np.float32)
        self.cameraChanged.emit()

    def look_at_point(
        self, point: np.ndarray, animate: bool = True, duration: float = 0.5
    ):
        """Look at a specific point"""
        if not np.isfinite(point).all():
            return

        if animate and self.smooth_transitions:
            # Create target state with new target
            target_state = CameraState()
            self._copy_state(self.current_state, target_state)
            target_state.target = point.astype(np.float32)

            self.animate_to_state(target_state, duration)
        else:
            self.current_state.target = point.astype(np.float32)
            self.cameraChanged.emit()

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _apply_constraints(self):
        """Apply camera constraints"""
        # Distance constraints
        self.current_state.distance = np.clip(
            self.current_state.distance,
            self.constraints.min_distance,
            self.constraints.max_distance,
        )

        # Elevation constraints
        self.current_state.elevation = np.clip(
            self.current_state.elevation,
            self.constraints.min_elevation,
            self.constraints.max_elevation,
        )

        # FOV constraints
        self.current_state.fov = np.clip(
            self.current_state.fov, self.constraints.min_fov, self.constraints.max_fov
        )

        # Position bounds (if set)
        if self.constraints.position_bounds:
            min_bounds, max_bounds = self.constraints.position_bounds
            self.current_state.target = np.clip(
                self.current_state.target, min_bounds, max_bounds
            )

    def set_mode(self, mode: CameraMode):
        """Set camera operation mode"""
        if mode != self.mode:
            self.mode = mode
            self.modeChanged.emit(mode.value)
            print(f"📷 Camera mode: {mode.value}")

    def reset_to_default(self, animate: bool = True):
        """Reset camera to default position"""
        self.set_preset(CameraPreset.DEFAULT, animate)

    def get_state_dict(self) -> dict:
        """Get camera state as dictionary for saving"""
        return {
            "position": self.current_state.position.tolist(),
            "target": self.current_state.target.tolist(),
            "distance": self.current_state.distance,
            "azimuth": self.current_state.azimuth,
            "elevation": self.current_state.elevation,
            "fov": self.current_state.fov,
            "mode": self.mode.value,
        }

    def load_state_dict(self, state_dict: dict, animate: bool = True):
        """Load camera state from dictionary"""
        target_state = CameraState()
        target_state.position = np.array(
            state_dict.get("position", [0, 0, 0]), dtype=np.float32
        )
        target_state.target = np.array(
            state_dict.get("target", [0, 0, 0]), dtype=np.float32
        )
        target_state.distance = state_dict.get("distance", 5.0)
        target_state.azimuth = state_dict.get("azimuth", 45.0)
        target_state.elevation = state_dict.get("elevation", 20.0)
        target_state.fov = state_dict.get("fov", 45.0)

        if animate:
            self.animate_to_state(target_state)
        else:
            self._copy_state(target_state, self.current_state)
            self.cameraChanged.emit()

        mode_str = state_dict.get("mode", "orbit")
        self.set_mode(CameraMode(mode_str))


# ============================================================================
# USAGE EXAMPLE AND TESTING
# ============================================================================

if __name__ == "__main__":
    """Test the camera system"""
    print("📷 Golf Swing Visualizer - Camera System Test")

    # Create camera controller
    camera = CameraController()

    # Test presets
    print("\n🎯 Testing camera presets...")
    for preset in CameraPreset:
        print(
            f"   {preset.value}: dist={camera.presets[preset].distance:.1f}, "
            f"azim={camera.presets[preset].azimuth:.0f}°, "
            f"elev={camera.presets[preset].elevation:.0f}°"
        )

    # Test matrix calculations
    print("\n🔢 Testing matrix calculations...")
    view_matrix = camera.get_view_matrix()
    proj_matrix = camera.get_projection_matrix(16 / 9)
    position = camera.get_camera_position()

    print(
        f"   Camera position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]"
    )
    print(f"   View matrix shape: {view_matrix.shape}")
    print(f"   Projection matrix shape: {proj_matrix.shape}")

    # Test animation
    print("\n🎬 Testing animation system...")
    animator = SmoothAnimator()

    start_vec = np.array([0, 0, 0])
    end_vec = np.array([1, 1, 1])
    mid_vec = animator.interpolate_vectors(
        start_vec, end_vec, 0.5, animator.ease_in_out_cubic
    )
    print(f"   Interpolation test: {start_vec} -> {mid_vec} -> {end_vec}")

    # Test spherical interpolation
    start_spherical = (5.0, 45.0, 20.0)
    end_spherical = (3.0, 135.0, -10.0)
    mid_spherical = animator.spherical_interpolation(
        start_spherical, end_spherical, 0.5
    )
    print(
        f"   Spherical interpolation: {start_spherical} -> "
        f"{mid_spherical} -> {end_spherical}"
    )

    print("\n🎉 Camera system ready for integration!")
