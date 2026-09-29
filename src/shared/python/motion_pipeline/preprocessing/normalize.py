"""
Coordinate normalization for motion capture data.

Part of issue #4564. World-up axis conversion, unit conversion, origin re-centering.

Vectorized per issue #8925: the per-marker/per-keypoint transform, unit
conversion, centering and up-axis detection passes are implemented as
array ops over ``(frames, points, xyz)`` ndarrays (via the shared
``_frame_arrays`` helpers) instead of per-element Python loops that
allocated a ``np.array`` and constructed a validated Pydantic object for
every marker/keypoint in every frame. Final frame reconstruction uses
``model_construct`` (skips Pydantic validation) since the arithmetic
applied — a fixed rotation/permutation matmul, a finite scalar unit
scale, and a finite centroid subtraction — is finite-preserving given
already-validated finite input, mirroring the precedent in
``sources/c3d_adapter.py``.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from ..contracts import (
    KeypointFrame,
    KeypointSequence,
    MarkerFrame,
    MarkerTrajectory,
)
from ._frame_arrays import (
    array_to_keypoint_frames,
    array_to_marker_frames,
    keypoints_to_array,
    markers_to_array,
)


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


def _keypoint_z_mask(frames: list[KeypointFrame]) -> np.ndarray:
    """Return an ``(frames, keypoints)`` bool mask of ``kp.z is not None``."""
    if not frames:
        return np.zeros((0, 0), dtype=bool)
    return np.array(
        [[kp.z is not None for kp in frame.keypoints] for frame in frames],
        dtype=bool,
    )


def _keypoint_centroid_from_first_frame(
    data: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Centroid of frame 0, matching the original per-element semantics.

    x/y average over all keypoints; z averages only keypoints whose z is
    not None (mirrors ``np.mean([]) -> nan`` when frame 0 has no z values,
    exactly as the original list-comprehension implementation did).
    """
    first = data[0]
    first_mask = mask[0]
    centroid_x = np.mean(first[:, 0])
    centroid_y = np.mean(first[:, 1])
    centroid_z = np.mean(first[first_mask, 2])
    return np.array([centroid_x, centroid_y, centroid_z])


def _marker_centroid_from_first_frame(data: np.ndarray) -> np.ndarray:
    """Centroid of frame 0 markers (all axes averaged over all markers)."""
    return np.mean(data[0], axis=0)


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

    data = keypoints_to_array(seq.frames)
    mask = _keypoint_z_mask(seq.frames)

    # Single vectorized expression: transform + (optional) centering, in
    # place of the old per-element matmul loop followed by a second full
    # Python-loop centering pass.
    transformed = data @ transform.T
    if center_origin:
        centroid = _keypoint_centroid_from_first_frame(transformed, mask)
        transformed = transformed - centroid

    new_frames = array_to_keypoint_frames(
        seq.frames, transformed, use_model_construct=True
    )

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

    data = markers_to_array(traj.frames)

    # Single vectorized expression: transform + (optional) centering.
    transformed = data @ transform.T
    if center_origin:
        centroid = _marker_centroid_from_first_frame(transformed)
        transformed = transformed - centroid

    new_frames = array_to_marker_frames(
        traj.frames, transformed, use_model_construct=True
    )

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

    data = keypoints_to_array(seq.frames)
    scaled = data * scale

    new_frames = array_to_keypoint_frames(seq.frames, scaled, use_model_construct=True)

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

    data = markers_to_array(traj.frames)
    scaled = data * scale

    new_frames = array_to_marker_frames(traj.frames, scaled, use_model_construct=True)

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

    data = keypoints_to_array(frames)
    mask = _keypoint_z_mask(frames)

    var_x = np.var(data[..., 0])
    var_y = np.var(data[..., 1])
    z_values = data[..., 2][mask]
    var_z = np.var(z_values) if z_values.size else 0

    if var_y > var_x and var_y > var_z:
        return UpAxis.Y_UP
    if var_z > var_x and var_z > var_y:
        return UpAxis.Z_UP
    return UpAxis.X_UP


def _detect_up_axis_markers(frames: list[MarkerFrame]) -> UpAxis:
    """Detect up axis from marker data."""
    if not frames:
        return UpAxis.Y_UP

    data = markers_to_array(frames)

    var_x = np.var(data[..., 0])
    var_y = np.var(data[..., 1])
    var_z = np.var(data[..., 2])

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

    data = keypoints_to_array(frames)
    mask = _keypoint_z_mask(frames)
    centroid = _keypoint_centroid_from_first_frame(data, mask)
    centered = data - centroid

    return array_to_keypoint_frames(frames, centered)


def _center_markers_origin(frames: list[MarkerFrame]) -> list[MarkerFrame]:
    """Center markers at origin."""
    if not frames:
        return frames

    data = markers_to_array(frames)
    centroid = _marker_centroid_from_first_frame(data)
    centered = data - centroid

    return array_to_marker_frames(frames, centered)
