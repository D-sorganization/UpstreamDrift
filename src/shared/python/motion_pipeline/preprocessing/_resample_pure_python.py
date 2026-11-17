"""
Resampling for motion capture data.

Part of issue #4564. Frame-rate conversion with anti-aliasing.
"""

from __future__ import annotations


from ..contracts import KeypointSequence, MarkerTrajectory
from ._frame_arrays import (
    array_to_keypoint_frames_at_timestamps as _array_to_keypoint_frames_at_timestamps,
    array_to_marker_frames_at_timestamps as _array_to_marker_frames_at_timestamps,
    estimate_fps as _estimate_fps,  # noqa: F401  (re-exported for tests)
    keypoints_to_array as _keypoints_to_array,
    markers_to_array as _markers_to_array,
    prepare_resample_inputs as _prepare_resample_inputs,
    vectorized_interp_axes as _vectorized_interp_axes,
)


def resample(
    data: KeypointSequence | MarkerTrajectory,
    target_fps: float,
    source_fps: float | None = None,
) -> KeypointSequence | MarkerTrajectory:
    """
    Resample motion capture data to target frame rate.

    Args:
        data: Input keypoint sequence or marker trajectory
        target_fps: Target frame rate in Hz
        source_fps: Source frame rate (auto-detected if not provided)

    Returns:
        Resampled data at target_fps

    Raises:
        ValueError: If data type is unsupported
    """
    if isinstance(data, KeypointSequence):
        return _resample_keypoints(data, target_fps, source_fps)
    if isinstance(data, MarkerTrajectory):
        return _resample_markers(data, target_fps, source_fps)
    raise ValueError(f"Unsupported data type: {type(data)}")


def _resample_keypoints(
    seq: KeypointSequence,
    target_fps: float,
    source_fps: float | None,
) -> KeypointSequence:
    """Resample keypoint sequence to target FPS."""
    if len(seq.frames) < 2:
        return seq

    source_fps, new_timestamps, data, timestamps = _prepare_resample_inputs(
        seq.frames, target_fps, source_fps, _keypoints_to_array
    )

    # Resample each keypoint dimension
    resampled_data = _vectorized_interp_axes(new_timestamps, timestamps, data)

    # Reconstruct frames
    new_frames = _array_to_keypoint_frames_at_timestamps(
        seq, resampled_data, new_timestamps
    )

    return KeypointSequence(
        id=seq.id,
        frames=new_frames,
        calibration=seq.calibration,
        metadata={
            **seq.metadata,
            "resampled": True,
            "source_fps": source_fps,
            "target_fps": target_fps,
        },
    )


def _resample_markers(
    traj: MarkerTrajectory,
    target_fps: float,
    source_fps: float | None,
) -> MarkerTrajectory:
    """Resample marker trajectory to target FPS."""
    if len(traj.frames) < 2:
        return traj

    source_fps, new_timestamps, data, timestamps = _prepare_resample_inputs(
        traj.frames, target_fps, source_fps, _markers_to_array
    )

    # Resample each marker dimension
    resampled_data = _vectorized_interp_axes(new_timestamps, timestamps, data)

    # Reconstruct frames
    new_frames = _array_to_marker_frames_at_timestamps(
        traj, resampled_data, new_timestamps
    )

    return MarkerTrajectory(
        id=traj.id,
        frames=new_frames,
        calibration=traj.calibration,
        subject_id=traj.subject_id,
        metadata={
            **traj.metadata,
            "resampled": True,
            "source_fps": source_fps,
            "target_fps": target_fps,
        },
    )
