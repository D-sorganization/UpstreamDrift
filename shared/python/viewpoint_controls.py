"""Multi-Perspective Viewpoint Controls Module.

Guideline L1 Implementation: Multi-Perspective Viewpoint Controls.

Provides easy viewpoint switching for video comparison and ML workflows:
- Preset camera views (face-on, DTL, overhead, sides)
- Custom angle specification (azimuth/elevation)
- Camera tracking (clubhead, COM, fixed world)
- Smooth transitions between preset views
- Synchronized multi-view rendering support
- Video matching calibration tools

Camera convention:
- Azimuth: 0° = looking at golfer's chest (face-on), CCW positive
- Elevation: 0° = level, positive = looking down
- Distance: Distance from look-at point to camera
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

LOGGER = logging.getLogger(__name__)


class CameraPreset(Enum):
    """Standard camera view presets."""

    FACE_ON = auto()  # Looking at golfer's chest
    DOWN_TARGET_LINE = auto()  # Behind golfer, looking toward target (DTL)
    OVERHEAD = auto()  # Bird's eye view
    RIGHT_SIDE = auto()  # 90° right of target line
    LEFT_SIDE = auto()  # 90° left of target line
    BEHIND_BALL = auto()  # Behind ball, looking at golfer
    IMPACT_CLOSE = auto()  # Close-up of impact zone
    CUSTOM = auto()  # User-defined


class TrackingTarget(Enum):
    """Camera tracking targets."""

    FIXED_WORLD = auto()  # Fixed world position
    CLUBHEAD = auto()  # Track clubhead
    BALL = auto()  # Track ball
    GOLFER_COM = auto()  # Track golfer center of mass
    SYSTEM_COM = auto()  # Track golfer+club center of mass
    HANDS = auto()  # Track hand position


@dataclass
class CameraState:
    """Complete camera state for 3D visualization.

    Attributes:
        position: Camera position in world frame [m] (3,)
        look_at: Point the camera is looking at [m] (3,)
        up_vector: Camera up direction [unitless] (3,)
        fov: Field of view [degrees]
        near_clip: Near clipping plane [m]
        far_clip: Far clipping plane [m]
        preset: Current preset (CUSTOM if manually set)
    """

    position: np.ndarray
    look_at: np.ndarray
    up_vector: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    fov: float = 45.0
    near_clip: float = 0.1
    far_clip: float = 100.0
    preset: CameraPreset = CameraPreset.CUSTOM


@dataclass
class ViewportLayout:
    """Layout specification for multi-view rendering.

    Attributes:
        rows: Number of viewport rows
        cols: Number of viewport columns
        camera_states: Camera state for each viewport (row-major)
    """

    rows: int
    cols: int
    camera_states: list[CameraState]


# Default reference positions (can be overridden per-model)
DEFAULT_GOLFER_POSITION = np.array([0.0, 0.0, 0.0])  # Golfer at origin
DEFAULT_TARGET_DIRECTION = np.array([1.0, 0.0, 0.0])  # Target along +X
DEFAULT_CAMERA_DISTANCE = 3.0  # [m] Default distance from look-at


def spherical_to_cartesian(
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
    center: np.ndarray,
) -> np.ndarray:
    """Convert spherical coordinates to Cartesian camera position.

    Args:
        azimuth_deg: Azimuth angle [degrees], 0° = +X (face-on)
        elevation_deg: Elevation angle [degrees], 0° = level
        distance: Distance from center [m]
        center: Look-at point [m] (3,)

    Returns:
        Camera position in world frame [m] (3,)
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    # Spherical to Cartesian
    x = distance * np.cos(el) * np.cos(az)
    y = distance * np.cos(el) * np.sin(az)
    z = distance * np.sin(el)

    return np.asarray(center + np.array([x, y, z]))


def get_preset_camera_params(
    preset: CameraPreset,
    golfer_position: np.ndarray | None = None,
    target_direction: np.ndarray | None = None,
    distance: float = DEFAULT_CAMERA_DISTANCE,
) -> tuple[float, float, np.ndarray]:
    """Get azimuth, elevation, and look-at point for a preset.

    Args:
        preset: Camera preset
        golfer_position: Golfer reference position [m] (3,)
        target_direction: Direction toward target [unitless] (3,)
        distance: Camera distance [m]

    Returns:
        Tuple of (azimuth_deg, elevation_deg, look_at_point)
    """
    if golfer_position is None:
        golfer_position = DEFAULT_GOLFER_POSITION.copy()
    if target_direction is None:
        target_direction = DEFAULT_TARGET_DIRECTION.copy()

    # Normalize target direction
    target_dir = target_direction / np.linalg.norm(target_direction)

    # Compute azimuth from target direction (0° = looking at golfer's chest)
    # Face-on means camera is in -target_direction
    base_azimuth = np.degrees(np.arctan2(target_dir[1], target_dir[0]))

    # Look-at is typically at golfer height
    look_at = golfer_position + np.array([0.0, 0.0, 1.0])  # ~1m up

    if preset == CameraPreset.FACE_ON:
        # Looking at golfer's chest, camera behind ball
        azimuth = base_azimuth + 180  # Opposite of target direction
        elevation = 0.0
    elif preset == CameraPreset.DOWN_TARGET_LINE:
        # Behind golfer, looking toward target
        azimuth = base_azimuth  # Same as target direction
        elevation = 10.0  # Slightly elevated
    elif preset == CameraPreset.OVERHEAD:
        azimuth = base_azimuth
        elevation = 80.0  # Near vertical
    elif preset == CameraPreset.RIGHT_SIDE:
        azimuth = base_azimuth + 90
        elevation = 0.0
    elif preset == CameraPreset.LEFT_SIDE:
        azimuth = base_azimuth - 90
        elevation = 0.0
    elif preset == CameraPreset.BEHIND_BALL:
        azimuth = base_azimuth + 180
        elevation = 15.0
        look_at = golfer_position + np.array([0.5, 0.0, 0.0])  # Ball position
    elif preset == CameraPreset.IMPACT_CLOSE:
        azimuth = base_azimuth + 135  # 45° from face-on
        elevation = 20.0
        look_at = golfer_position + np.array([0.3, 0.0, 0.3])  # Impact zone
    else:
        azimuth = 0.0
        elevation = 0.0

    return azimuth, elevation, look_at


def create_camera_from_preset(
    preset: CameraPreset,
    golfer_position: np.ndarray | None = None,
    target_direction: np.ndarray | None = None,
    distance: float = DEFAULT_CAMERA_DISTANCE,
    fov: float = 45.0,
) -> CameraState:
    """Create a camera state from a preset.

    Args:
        preset: Camera preset
        golfer_position: Golfer reference position [m] (3,)
        target_direction: Direction toward target [unitless] (3,)
        distance: Camera distance [m]
        fov: Field of view [degrees]

    Returns:
        CameraState configured for the preset
    """
    azimuth, elevation, look_at = get_preset_camera_params(
        preset, golfer_position, target_direction, distance
    )

    position = spherical_to_cartesian(azimuth, elevation, distance, look_at)

    return CameraState(
        position=position,
        look_at=look_at,
        up_vector=np.array([0.0, 0.0, 1.0]),
        fov=fov,
        preset=preset,
    )


def create_custom_camera(
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
    look_at: np.ndarray,
    fov: float = 45.0,
) -> CameraState:
    """Create a custom camera with spherical coordinates.

    Args:
        azimuth_deg: Azimuth angle [degrees]
        elevation_deg: Elevation angle [degrees]
        distance: Distance from look-at point [m]
        look_at: Point to look at [m] (3,)
        fov: Field of view [degrees]

    Returns:
        CameraState with custom configuration
    """
    position = spherical_to_cartesian(azimuth_deg, elevation_deg, distance, look_at)

    return CameraState(
        position=position,
        look_at=look_at.copy(),
        up_vector=np.array([0.0, 0.0, 1.0]),
        fov=fov,
        preset=CameraPreset.CUSTOM,
    )


def interpolate_camera_states(
    start: CameraState,
    end: CameraState,
    t: float,
) -> CameraState:
    """Interpolate between two camera states for smooth transition.

    Args:
        start: Starting camera state
        end: Ending camera state
        t: Interpolation parameter (0 = start, 1 = end)

    Returns:
        Interpolated camera state
    """
    t = np.clip(t, 0.0, 1.0)

    # Smooth step for nicer transitions
    t_smooth = t * t * (3.0 - 2.0 * t)

    position = (1 - t_smooth) * start.position + t_smooth * end.position
    look_at = (1 - t_smooth) * start.look_at + t_smooth * end.look_at
    up_vector = (1 - t_smooth) * start.up_vector + t_smooth * end.up_vector
    fov = (1 - t_smooth) * start.fov + t_smooth * end.fov

    # Normalize up vector
    up_vector = up_vector / np.linalg.norm(up_vector)

    return CameraState(
        position=position,
        look_at=look_at,
        up_vector=up_vector,
        fov=fov,
        preset=CameraPreset.CUSTOM,
    )


def create_transition_sequence(
    start: CameraState,
    end: CameraState,
    num_frames: int,
) -> list[CameraState]:
    """Create a sequence of camera states for smooth transition.

    Args:
        start: Starting camera state
        end: Ending camera state
        num_frames: Number of frames in transition

    Returns:
        List of interpolated camera states
    """
    if num_frames < 2:
        return [end]

    states = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        states.append(interpolate_camera_states(start, end, t))

    return states


def compute_tracking_look_at(
    target: TrackingTarget,
    clubhead_position: np.ndarray | None = None,
    ball_position: np.ndarray | None = None,
    golfer_com: np.ndarray | None = None,
    system_com: np.ndarray | None = None,
    hand_position: np.ndarray | None = None,
    fixed_position: np.ndarray | None = None,
) -> np.ndarray:
    """Compute look-at point based on tracking target.

    Args:
        target: Tracking target type
        clubhead_position: Current clubhead position [m] (3,)
        ball_position: Current ball position [m] (3,)
        golfer_com: Current golfer COM [m] (3,)
        system_com: Current system COM [m] (3,)
        hand_position: Current hand position [m] (3,)
        fixed_position: Fixed world position [m] (3,)

    Returns:
        Look-at point [m] (3,)
    """
    if target == TrackingTarget.CLUBHEAD and clubhead_position is not None:
        return clubhead_position.copy()
    elif target == TrackingTarget.BALL and ball_position is not None:
        return ball_position.copy()
    elif target == TrackingTarget.GOLFER_COM and golfer_com is not None:
        return golfer_com.copy()
    elif target == TrackingTarget.SYSTEM_COM and system_com is not None:
        return system_com.copy()
    elif target == TrackingTarget.HANDS and hand_position is not None:
        return hand_position.copy()
    elif target == TrackingTarget.FIXED_WORLD and fixed_position is not None:
        return fixed_position.copy()

    # Default fallback
    return np.asarray(DEFAULT_GOLFER_POSITION + np.array([0.0, 0.0, 1.0]))


def create_multiview_layout(
    presets: list[CameraPreset],
    golfer_position: np.ndarray | None = None,
    target_direction: np.ndarray | None = None,
    distance: float = DEFAULT_CAMERA_DISTANCE,
) -> ViewportLayout:
    """Create a multi-view layout with specified presets.

    Automatically determines grid layout based on number of views.

    Args:
        presets: List of camera presets for each viewport
        golfer_position: Golfer reference position
        target_direction: Direction toward target
        distance: Camera distance

    Returns:
        ViewportLayout with camera states
    """
    n = len(presets)

    # Determine grid size
    if n == 1:
        rows, cols = 1, 1
    elif n == 2:
        rows, cols = 1, 2
    elif n <= 4:
        rows, cols = 2, 2
    elif n <= 6:
        rows, cols = 2, 3
    else:
        # Approximate square grid
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

    cameras = [
        create_camera_from_preset(preset, golfer_position, target_direction, distance)
        for preset in presets
    ]

    return ViewportLayout(rows=rows, cols=cols, camera_states=cameras)


def create_standard_2x2_layout(
    golfer_position: np.ndarray | None = None,
    target_direction: np.ndarray | None = None,
) -> ViewportLayout:
    """Create a standard 2x2 multi-view layout.

    Views:
    - Top-left: Face-on
    - Top-right: Down-target-line
    - Bottom-left: Overhead
    - Bottom-right: Right side

    Args:
        golfer_position: Golfer reference position
        target_direction: Direction toward target

    Returns:
        2x2 ViewportLayout
    """
    presets = [
        CameraPreset.FACE_ON,
        CameraPreset.DOWN_TARGET_LINE,
        CameraPreset.OVERHEAD,
        CameraPreset.RIGHT_SIDE,
    ]
    return create_multiview_layout(presets, golfer_position, target_direction)


class ViewpointController:
    """High-level controller for camera viewpoint management.

    Provides:
    - Preset switching with smooth transitions
    - Camera tracking mode
    - Custom angle control
    - Multi-view synchronization
    """

    def __init__(
        self,
        golfer_position: np.ndarray | None = None,
        target_direction: np.ndarray | None = None,
    ) -> None:
        """Initialize view controller.

        Args:
            golfer_position: Reference golfer position
            target_direction: Direction toward target
        """
        self.golfer_position = (
            golfer_position.copy()
            if golfer_position is not None
            else DEFAULT_GOLFER_POSITION.copy()
        )
        self.target_direction = (
            target_direction.copy()
            if target_direction is not None
            else DEFAULT_TARGET_DIRECTION.copy()
        )

        # Current state
        self.current_camera = create_camera_from_preset(
            CameraPreset.FACE_ON, self.golfer_position, self.target_direction
        )
        self.tracking_target = TrackingTarget.FIXED_WORLD
        self.transition_in_progress = False
        self.transition_frames: list[CameraState] = []
        self.transition_index = 0

    def set_preset(
        self,
        preset: CameraPreset,
        distance: float = DEFAULT_CAMERA_DISTANCE,
        transition_frames: int = 0,
    ) -> CameraState:
        """Switch to a camera preset.

        Args:
            preset: Target preset
            distance: Camera distance
            transition_frames: Number of frames for smooth transition (0 = instant)

        Returns:
            New camera state (or first frame of transition)
        """
        target_camera = create_camera_from_preset(
            preset, self.golfer_position, self.target_direction, distance
        )

        if transition_frames > 0:
            self.transition_frames = create_transition_sequence(
                self.current_camera, target_camera, transition_frames
            )
            self.transition_index = 0
            self.transition_in_progress = True
            return self.transition_frames[0]
        else:
            self.current_camera = target_camera
            self.transition_in_progress = False
            return target_camera

    def set_custom_view(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        distance: float,
        look_at: np.ndarray | None = None,
    ) -> CameraState:
        """Set a custom camera view.

        Args:
            azimuth_deg: Azimuth angle [degrees]
            elevation_deg: Elevation angle [degrees]
            distance: Camera distance [m]
            look_at: Look-at point (uses default if None)

        Returns:
            New camera state
        """
        if look_at is None:
            look_at = self.golfer_position + np.array([0.0, 0.0, 1.0])

        self.current_camera = create_custom_camera(
            azimuth_deg, elevation_deg, distance, look_at
        )
        self.transition_in_progress = False
        return self.current_camera

    def set_tracking_target(self, target: TrackingTarget) -> None:
        """Set the camera tracking target.

        Args:
            target: What the camera should track
        """
        self.tracking_target = target

    def update(
        self,
        clubhead_position: np.ndarray | None = None,
        ball_position: np.ndarray | None = None,
        golfer_com: np.ndarray | None = None,
    ) -> CameraState:
        """Update camera state for current frame.

        Handles transitions and tracking.

        Args:
            clubhead_position: Current clubhead position
            ball_position: Current ball position
            golfer_com: Current golfer COM

        Returns:
            Current camera state
        """
        # Handle ongoing transition
        if self.transition_in_progress:
            if self.transition_index < len(self.transition_frames):
                self.current_camera = self.transition_frames[self.transition_index]
                self.transition_index += 1
            else:
                self.transition_in_progress = False

        # Handle tracking
        if self.tracking_target != TrackingTarget.FIXED_WORLD:
            new_look_at = compute_tracking_look_at(
                self.tracking_target,
                clubhead_position=clubhead_position,
                ball_position=ball_position,
                golfer_com=golfer_com,
                fixed_position=self.current_camera.look_at,
            )

            # Maintain relative camera position
            offset = self.current_camera.position - self.current_camera.look_at
            self.current_camera.position = new_look_at + offset
            self.current_camera.look_at = new_look_at

        return self.current_camera

    def get_multiview_layout(
        self,
        presets: list[CameraPreset] | None = None,
    ) -> ViewportLayout:
        """Get a multi-view layout.

        Args:
            presets: List of presets (uses standard 2x2 if None)

        Returns:
            ViewportLayout for rendering
        """
        if presets is None:
            return create_standard_2x2_layout(
                self.golfer_position, self.target_direction
            )
        else:
            return create_multiview_layout(
                presets, self.golfer_position, self.target_direction
            )
