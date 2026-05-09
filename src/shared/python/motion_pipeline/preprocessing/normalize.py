"""
Coordinate normalization for motion capture data.

Part of issue #4564. World-up axis conversion, unit conversion, origin re-centering.
"""

from __future__ import annotations

from enum import Enum
import numpy as np

from ..contracts import KeypointFrame, KeypointSequence, MarkerFrame, MarkerTrajectory


class UpAxis(str, Enum):
    """Up axis conventions."""

    Y_UP = "+Y"  # Y is up (BVH, Maya, MotionBuilder)
    Z_UP = "+Z"  # Z is up (3ds Max, Blender default)
    X_UP = "+X"  # X is up (some CAD systems)


class UnitSystem(str, Enum):
    """Unit systems."""

    METERS = "m"
    MILLIMETERS = "mm"
    CENTIMETERS = "cm"
    INCHES = "in"


def normalize_coordinates(
    data: KeypointSequence | MarkerTrajectory,
    target_up: UpAxis = UpAxis.Y_UP,
    source_up: UpAxis | None = None,
    center_origin: bool = True,
) -> KeypointSequence | MarkerTrajectory:
    """
    Normalize coordinate system of motion capture data.

    Args:
        data: Input keypoint sequence or marker trajectory
        target_up: Target up axis
        source_up: Source up axis (auto-detected if not provided)
        center_origin: Whether to center at origin

    Returns:
        Data with normalized coordinates

    Raises:
        ValueError: If data type is unsupported
    """
    if isinstance(data, KeypointSequence):
        return _normalize_keypoints(data, target_up, source_up, center_origin)
    if isinstance(data, MarkerTrajectory):
        return _normalize_markers(data, target_up, source_up, center_origin)
    raise ValueError(f"Unsupported data type: {type(data)}")


def convert_units(
    data: KeypointSequence | MarkerTrajectory,
    target_unit: UnitSystem = UnitSystem.METERS,
    source_unit: UnitSystem | None = None,
) -> KeypointSequence | MarkerTrajectory:
    """
    Convert units of motion capture data.

    Args:
        data: Input keypoint sequence or marker trajectory
        target_unit: Target unit system
        source_unit: Source unit system (auto-detected if not provided)

    Returns:
        Data with converted units

    Raises:
        ValueError: If data type is unsupported
    """
    if isinstance(data, KeypointSequence):
        return _convert_keypoint_units(data, target_unit, source_unit)
    if isinstance(data, MarkerTrajectory):
        return _convert_marker_units(data, target_unit, source_unit)
    raise ValueError(f"Unsupported data type: {type(data)}")


def _normalize_keypoints(
    seq: KeypointSequence,
    target_up: UpAxis,
    source_up: UpAxis | None,
    center_origin: bool,
) -> KeypointSequence:
    """Normalize keypoint coordinates."""
    if not seq.frames:
        return seq

    # Auto-detect source up axis
    if source_up is None:
        source_up = _detect_up_axis_keypoints(seq.frames)

    # Compute transformation matrix
    transform = _get_up_axis_transform(source_up, target_up)

    # Apply transformation
    new_frames = []
    for frame in seq.frames:
        new_keypoints = []
        for kp in frame.keypoints:
            pos = np.array([kp.x, kp.y, kp.z if kp.z is not None else 0.0])
            new_pos = transform @ pos

            new_kp = Keypoint(
                x=new_pos[0],
                y=new_pos[1],
                z=new_pos[2] if kp.z is not None else None,
                confidence=kp.confidence,
                name=kp.name,
            )
            new_keypoints.append(new_kp)

        new_frames.append(
            KeypointFrame(
                timestamp=frame.timestamp,
                keypoints=new_keypoints,
                schema_name=frame.schema_name,
                frame_index=frame.frame_index,
            )
        )

    # Center origin if requested
    if center_origin:
        new_frames = _center_keypoints_origin(new_frames)

    return KeypointSequence(
        id=seq.id,
        frames=new_frames,
        calibration=seq.calibration,
        metadata={
            **seq.metadata,
            "normalized": True,
            "source_up": source_up.value,
            "target_up": target_up.value,
            "centered": center_origin,
        },
    )


def _normalize_markers(
    traj: MarkerTrajectory,
    target_up: UpAxis,
    source_up: UpAxis | None,
    center_origin: bool,
) -> MarkerTrajectory:
    """Normalize marker coordinates."""
    if not traj.frames:
        return traj

    # Auto-detect source up axis
    if source_up is None:
        source_up = _detect_up_axis_markers(traj.frames)

    # Compute transformation matrix
    transform = _get_up_axis_transform(source_up, target_up)

    # Apply transformation
    new_frames = []
    for frame in traj.frames:
        new_markers = {}
        for name, marker in frame.markers.items():
            pos = np.array([marker.x, marker.y, marker.z])
            new_pos = transform @ pos

            new_markers[name] = Marker(
                name=name,
                x=new_pos[0],
                y=new_pos[1],
                z=new_pos[2],
                residual=marker.residual,
                occluded=marker.occluded,
            )

        new_frames.append(
            MarkerFrame(
                timestamp=frame.timestamp,
                markers=new_markers,
                frame_index=frame.frame_index,
            )
        )

    # Center origin if requested
    if center_origin:
        new_frames = _center_markers_origin(new_frames)

    return MarkerTrajectory(
        id=traj.id,
        frames=new_frames,
        calibration=traj.calibration,
        subject_id=traj.subject_id,
        metadata={
            **traj.metadata,
            "normalized": True,
            "source_up": source_up.value,
            "target_up": target_up.value,
            "centered": center_origin,
        },
    )


def _convert_keypoint_units(
    seq: KeypointSequence,
    target_unit: UnitSystem,
    source_unit: UnitSystem | None,
) -> KeypointSequence:
    """Convert keypoint units."""
    if not seq.frames:
        return seq

    # Auto-detect source unit
    if source_unit is None:
        source_unit = _detect_unit_keypoints(seq.frames)

    # Compute scale factor
    scale = _get_unit_scale(source_unit, target_unit)

    # Apply scale
    new_frames = []
    for frame in seq.frames:
        new_keypoints = []
        for kp in frame.keypoints:
            new_kp = Keypoint(
                x=kp.x * scale,
                y=kp.y * scale,
                z=kp.z * scale if kp.z is not None else None,
                confidence=kp.confidence,
                name=kp.name,
            )
            new_keypoints.append(new_kp)

        new_frames.append(
            KeypointFrame(
                timestamp=frame.timestamp,
                keypoints=new_keypoints,
                schema_name=frame.schema_name,
                frame_index=frame.frame_index,
            )
        )

    return KeypointSequence(
        id=seq.id,
        frames=new_frames,
        calibration=seq.calibration,
        metadata={
            **seq.metadata,
            "units_converted": True,
            "source_unit": source_unit.value,
            "target_unit": target_unit.value,
        },
    )


def _convert_marker_units(
    traj: MarkerTrajectory,
    target_unit: UnitSystem,
    source_unit: UnitSystem | None,
) -> MarkerTrajectory:
    """Convert marker units."""
    if not traj.frames:
        return traj

    # Auto-detect source unit
    if source_unit is None:
        source_unit = _detect_unit_markers(traj.frames)

    # Compute scale factor
    scale = _get_unit_scale(source_unit, target_unit)

    # Apply scale
    new_frames = []
    for frame in traj.frames:
        new_markers = {}
        for name, marker in frame.markers.items():
            new_markers[name] = Marker(
                name=name,
                x=marker.x * scale,
                y=marker.y * scale,
                z=marker.z * scale,
                residual=marker.residual,
                occluded=marker.occluded,
            )

        new_frames.append(
            MarkerFrame(
                timestamp=frame.timestamp,
                markers=new_markers,
                frame_index=frame.frame_index,
            )
        )

    return MarkerTrajectory(
        id=traj.id,
        frames=new_frames,
        calibration=traj.calibration,
        subject_id=traj.subject_id,
        metadata={
            **traj.metadata,
            "units_converted": True,
            "source_unit": source_unit.value,
            "target_unit": target_unit.value,
        },
    )


def _detect_up_axis_keypoints(frames: list[KeypointFrame]) -> UpAxis:
    """Detect up axis from keypoint data."""
    # Heuristic: check which axis has the largest variance (typically up)
    if not frames:
        return UpAxis.Y_UP

    all_x = [kp.x for f in frames for kp in f.keypoints]
    all_y = [kp.y for f in frames for kp in f.keypoints]
    all_z = [kp.z for f in frames for kp in f.keypoints if kp.z is not None]

    var_x = np.var(all_x)
    var_y = np.var(all_y)
    var_z = np.var(all_z) if all_z else 0

    if var_y > var_x and var_y > var_z:
        return UpAxis.Y_UP
    if var_z > var_x and var_z > var_y:
        return UpAxis.Z_UP
    return UpAxis.X_UP


def _detect_up_axis_markers(frames: list[MarkerFrame]) -> UpAxis:
    """Detect up axis from marker data."""
    if not frames:
        return UpAxis.Y_UP

    all_x = [m.x for f in frames for m in f.markers.values()]
    all_y = [m.y for f in frames for m in f.markers.values()]
    all_z = [m.z for f in frames for m in f.markers.values()]

    var_x = np.var(all_x)
    var_y = np.var(all_y)
    var_z = np.var(all_z)

    if var_y > var_x and var_y > var_z:
        return UpAxis.Y_UP
    if var_z > var_x and var_z > var_y:
        return UpAxis.Z_UP
    return UpAxis.X_UP


def _detect_unit_keypoints(frames: list[KeypointFrame]) -> UnitSystem:
    """Detect unit from keypoint data."""
    # Heuristic: check typical human height (~1.7m)
    if not frames:
        return UnitSystem.METERS

    # Get vertical extent
    all_y = [kp.y for f in frames for kp in f.keypoints]
    extent = max(all_y) - min(all_y)

    if extent > 100:  # Likely mm
        return UnitSystem.MILLIMETERS
    if extent > 10:  # Likely cm
        return UnitSystem.CENTIMETERS
    return UnitSystem.METERS


def _detect_unit_markers(frames: list[MarkerFrame]) -> UnitSystem:
    """Detect unit from marker data."""
    if not frames:
        return UnitSystem.METERS

    all_y = [m.y for f in frames for m in f.markers.values()]
    extent = max(all_y) - min(all_y)

    if extent > 100:
        return UnitSystem.MILLIMETERS
    if extent > 10:
        return UnitSystem.CENTIMETERS
    return UnitSystem.METERS


def _get_up_axis_transform(source: UpAxis, target: UpAxis) -> np.ndarray:
    """Get transformation matrix between up axes."""
    if source == target:
        return np.eye(3)

    # Define axis mappings
    transforms = {
        (UpAxis.Y_UP, UpAxis.Z_UP): np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        (UpAxis.Z_UP, UpAxis.Y_UP): np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
        (UpAxis.Y_UP, UpAxis.X_UP): np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
        (UpAxis.X_UP, UpAxis.Y_UP): np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        (UpAxis.Z_UP, UpAxis.X_UP): np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
        (UpAxis.X_UP, UpAxis.Z_UP): np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
    }

    return transforms.get((source, target), np.eye(3))


def _get_unit_scale(source: UnitSystem, target: UnitSystem) -> float:
    """Get scale factor for unit conversion."""
    # Convert to meters first, then to target
    to_meters = {
        UnitSystem.METERS: 1.0,
        UnitSystem.MILLIMETERS: 0.001,
        UnitSystem.CENTIMETERS: 0.01,
        UnitSystem.INCHES: 0.0254,
    }

    from_meters = {
        UnitSystem.METERS: 1.0,
        UnitSystem.MILLIMETERS: 1000.0,
        UnitSystem.CENTIMETERS: 100.0,
        UnitSystem.INCHES: 39.3701,
    }

    return from_meters[target] * to_meters[source]


def _center_keypoints_origin(frames: list[KeypointFrame]) -> list[KeypointFrame]:
    """Center keypoints at origin."""
    if not frames:
        return frames

    # Compute centroid of first frame
    first_frame = frames[0]
    centroid_x = np.mean([kp.x for kp in first_frame.keypoints])
    centroid_y = np.mean([kp.y for kp in first_frame.keypoints])
    centroid_z = np.mean([kp.z for kp in first_frame.keypoints if kp.z is not None])

    # Center all frames
    new_frames = []
    for frame in frames:
        new_keypoints = []
        for kp in frame.keypoints:
            new_kp = Keypoint(
                x=kp.x - centroid_x,
                y=kp.y - centroid_y,
                z=(kp.z - centroid_z) if kp.z is not None else None,
                confidence=kp.confidence,
                name=kp.name,
            )
            new_keypoints.append(new_kp)

        new_frames.append(
            KeypointFrame(
                timestamp=frame.timestamp,
                keypoints=new_keypoints,
                schema_name=frame.schema_name,
                frame_index=frame.frame_index,
            )
        )

    return new_frames


def _center_markers_origin(frames: list[MarkerFrame]) -> list[MarkerFrame]:
    """Center markers at origin."""
    if not frames:
        return frames

    # Compute centroid of first frame
    first_frame = frames[0]
    all_markers = list(first_frame.markers.values())
    centroid_x = np.mean([m.x for m in all_markers])
    centroid_y = np.mean([m.y for m in all_markers])
    centroid_z = np.mean([m.z for m in all_markers])

    # Center all frames
    new_frames = []
    for frame in frames:
        new_markers = {}
        for name, marker in frame.markers.items():
            new_markers[name] = Marker(
                name=name,
                x=marker.x - centroid_x,
                y=marker.y - centroid_y,
                z=marker.z - centroid_z,
                residual=marker.residual,
                occluded=marker.occluded,
            )

        new_frames.append(
            MarkerFrame(
                timestamp=frame.timestamp,
                markers=new_markers,
                frame_index=frame.frame_index,
            )
        )

    return new_frames


# Import for type hints
from ..contracts import Keypoint, Marker
