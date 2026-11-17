"""
Resampling for motion capture data.

Part of issue #4564. Frame-rate conversion with anti-aliasing.

Linear-interpolation inner loops are routed through the Rust
``upstream-mocap-preproc`` wheel when available; the pure-Python ``np.interp``
fallback in ``_resample_pure_python.py`` is used otherwise.
"""

from __future__ import annotations


import numpy as np

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

try:  # pragma: no cover - import guard
    import upstream_mocap_preproc as _rust_kernel  # type: ignore[import-not-found]

    _RUST_AVAILABLE = True
except ImportError:  # pragma: no cover
    _rust_kernel = None  # type: ignore[assignment]
    _RUST_AVAILABLE = False


def _rust_interp_axes(
    data: np.ndarray,
    source_ts: np.ndarray,
    target_ts: np.ndarray,
) -> np.ndarray:
    """Inner numpy-only resample helper. Dispatches to Rust when available."""
    if _RUST_AVAILABLE:
        return np.asarray(
            _rust_kernel.resample_fps(  # type: ignore[union-attr]
                np.ascontiguousarray(data, dtype=np.float64),
                np.ascontiguousarray(source_ts, dtype=np.float64),
                np.ascontiguousarray(target_ts, dtype=np.float64),
            )
        )
    # Pure-Python fallback: vectorized per-axis linear interpolation.
    return _vectorized_interp_axes(target_ts, source_ts, data)


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

    # Resample each keypoint dimension (Rust kernel or numpy fallback)
    resampled_data = _rust_interp_axes(data, timestamps, new_timestamps)

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

    # Resample each marker dimension (Rust kernel or numpy fallback)
    resampled_data = _rust_interp_axes(data, timestamps, new_timestamps)

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
