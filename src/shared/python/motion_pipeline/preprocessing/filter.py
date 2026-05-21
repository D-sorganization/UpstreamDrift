"""
Filtering for motion capture data.

Part of issue #4564. Butterworth, Savitzky-Golay, and Kalman filters
for smoothing noisy motion capture data.

Hot-loop numeric kernels are routed through the Rust ``upstream-mocap-preproc``
wheel when available (see ``rust_core/upstream-mocap-preproc/``). If the wheel
is not installed, the pure-Python implementation in
``_filter_pure_python.py`` is used as a transparent fallback. The public API
(``apply_filter``, ``FilterType``) is identical either way.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from ..contracts import KeypointFrame, KeypointSequence, MarkerFrame, MarkerTrajectory

try:  # pragma: no cover - import guard exercised in CI matrix
    import upstream_mocap_preproc as _rust_kernel  # type: ignore[import-not-found]

    _RUST_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path
    _rust_kernel = None  # type: ignore[assignment]
    _RUST_AVAILABLE = False


class FilterType(str, Enum):
    """Filter types."""

    BUTTERWORTH = "butterworth"  # Low-pass Butterworth filter
    SAVITZKY_GOLAY = "savgol"  # Savitzky-Golay smoothing filter
    KALMAN = "kalman"  # Kalman filter
    MEDIAN = "median"  # Median filter
    GAUSSIAN = "gaussian"  # Gaussian filter


def apply_filter(
    data: KeypointSequence | MarkerTrajectory,
    filter_type: FilterType = FilterType.BUTTERWORTH,
    cutoff: float = 6.0,  # Hz
    order: int = 2,
    fps: float | None = None,
) -> KeypointSequence | MarkerTrajectory:
    """
    Apply filter to motion capture data.

    Args:
        data: Input keypoint sequence or marker trajectory
        filter_type: Type of filter to apply
        cutoff: Cutoff frequency in Hz (for Butterworth)
        order: Filter order
        fps: Frame rate (auto-detected if not provided)

    Returns:
        Filtered data

    Raises:
        ValueError: If data type is unsupported
    """
    if isinstance(data, KeypointSequence):
        return _filter_keypoints(data, filter_type, cutoff, order, fps)
    if isinstance(data, MarkerTrajectory):
        return _filter_markers(data, filter_type, cutoff, order, fps)
    raise ValueError(f"Unsupported data type: {type(data)}")


def _filter_keypoints(
    seq: KeypointSequence,
    filter_type: FilterType,
    cutoff: float,
    order: int,
    fps: float | None,
) -> KeypointSequence:
    """Apply filter to keypoint sequence."""
    if len(seq.frames) < 2:
        return seq

    # Auto-detect FPS
    if fps is None:
        fps = _estimate_fps(seq.frames)

    # Extract data arrays
    data = _keypoints_to_array(seq.frames)

    # Apply filter
    if filter_type == FilterType.BUTTERWORTH:
        filtered = _butterworth_filter(data, cutoff, order, fps)
    elif filter_type == FilterType.SAVITZKY_GOLAY:
        filtered = _savgol_filter(
            data, window_length=min(11, len(seq.frames)), polyorder=2
        )
    elif filter_type == FilterType.MEDIAN:
        filtered = _median_filter(data, kernel_size=3)
    elif filter_type == FilterType.GAUSSIAN:
        filtered = _gaussian_filter(data, sigma=1.0)
    elif filter_type == FilterType.KALMAN:
        filtered = _kalman_filter(data)
    else:
        filtered = data

    # Reconstruct frames
    filtered_frames = _array_to_keypoint_frames(seq.frames, filtered)

    return KeypointSequence(
        id=seq.id,
        frames=filtered_frames,
        calibration=seq.calibration,
        metadata={**seq.metadata, "filtered": True, "filter_type": filter_type.value},
    )


def _filter_markers(
    traj: MarkerTrajectory,
    filter_type: FilterType,
    cutoff: float,
    order: int,
    fps: float | None,
) -> MarkerTrajectory:
    """Apply filter to marker trajectory."""
    if len(traj.frames) < 2:
        return traj

    # Auto-detect FPS
    if fps is None:
        fps = _estimate_fps(traj.frames)

    # Extract data arrays
    data = _markers_to_array(traj.frames)

    # Apply filter
    if filter_type == FilterType.BUTTERWORTH:
        filtered = _butterworth_filter(data, cutoff, order, fps)
    elif filter_type == FilterType.SAVITZKY_GOLAY:
        filtered = _savgol_filter(
            data, window_length=min(11, len(traj.frames)), polyorder=2
        )
    elif filter_type == FilterType.MEDIAN:
        filtered = _median_filter(data, kernel_size=3)
    elif filter_type == FilterType.GAUSSIAN:
        filtered = _gaussian_filter(data, sigma=1.0)
    elif filter_type == FilterType.KALMAN:
        filtered = _kalman_filter(data)
    else:
        filtered = data

    # Reconstruct frames
    filtered_frames = _array_to_marker_frames(traj.frames, filtered)

    return MarkerTrajectory(
        id=traj.id,
        frames=filtered_frames,
        calibration=traj.calibration,
        subject_id=traj.subject_id,
        metadata={**traj.metadata, "filtered": True, "filter_type": filter_type.value},
    )


def _estimate_fps(frames: list) -> float:
    """Estimate FPS from frame timestamps."""
    if len(frames) < 2:
        return 30.0

    timestamps = [f.timestamp for f in frames]
    dt = np.mean(np.diff(timestamps))

    if dt <= 0:
        return 30.0

    return 1.0 / dt


def _keypoints_to_array(frames: list[KeypointFrame]) -> np.ndarray:
    """Convert keypoint frames to array."""
    if not frames:
        return np.array([])

    num_frames = len(frames)
    num_keypoints = len(frames[0].keypoints)

    # Shape: (num_frames, num_keypoints, 3)
    data = np.zeros((num_frames, num_keypoints, 3))

    for i, frame in enumerate(frames):
        for j, kp in enumerate(frame.keypoints):
            data[i, j, 0] = kp.x
            data[i, j, 1] = kp.y
            if kp.z is not None:
                data[i, j, 2] = kp.z

    return data


def _markers_to_array(frames: list[MarkerFrame]) -> np.ndarray:
    """Convert marker frames to array."""
    if not frames:
        return np.array([])

    num_frames = len(frames)
    marker_names = list(frames[0].markers.keys())
    num_markers = len(marker_names)

    # Shape: (num_frames, num_markers, 3)
    data = np.zeros((num_frames, num_markers, 3))

    for i, frame in enumerate(frames):
        for j, name in enumerate(marker_names):
            if name in frame.markers:
                m = frame.markers[name]
                data[i, j, 0] = m.x
                data[i, j, 1] = m.y
                data[i, j, 2] = m.z

    return data


def _array_to_keypoint_frames(
    frames: list[KeypointFrame],
    data: np.ndarray,
) -> list[KeypointFrame]:
    """Convert array back to keypoint frames."""
    new_frames = []

    for i, frame in enumerate(frames):
        new_keypoints = []
        for j, kp in enumerate(frame.keypoints):
            new_kp = Keypoint(
                x=data[i, j, 0],
                y=data[i, j, 1],
                z=data[i, j, 2] if kp.z is not None else None,
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


def _array_to_marker_frames(
    frames: list[MarkerFrame],
    data: np.ndarray,
) -> list[MarkerFrame]:
    """Convert array back to marker frames."""
    new_frames = []
    marker_names = list(frames[0].markers.keys())

    for i, frame in enumerate(frames):
        new_markers = {}
        for j, name in enumerate(marker_names):
            if name in frame.markers:
                m = frame.markers[name]
                new_markers[name] = Marker(
                    name=name,
                    x=data[i, j, 0],
                    y=data[i, j, 1],
                    z=data[i, j, 2],
                    residual=m.residual,
                    occluded=m.occluded,
                )

        new_frames.append(
            MarkerFrame(
                timestamp=frame.timestamp,
                markers=new_markers,
                frame_index=frame.frame_index,
            )
        )

    return new_frames


def _butterworth_filter(
    data: np.ndarray,
    cutoff: float,
    order: int,
    fps: float,
) -> np.ndarray:
    """Apply Butterworth low-pass filter.

    Routes to the Rust kernel when ``upstream_mocap_preproc`` is importable;
    otherwise dispatches to the pure-Python implementation that mirrors the
    historical SciPy-backed code path.
    """
    if _RUST_AVAILABLE:
        return np.asarray(
            _rust_kernel.butterworth_filter(  # type: ignore[union-attr]
                np.ascontiguousarray(data, dtype=np.float64),
                float(cutoff),
                int(order),
                float(fps),
            )
        )
    from ._filter_pure_python import _butterworth_filter as _py_impl

    return _py_impl(data, cutoff, order, fps)


def _savgol_filter(
    data: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
) -> np.ndarray:
    """Apply Savitzky-Golay filter (Rust kernel with Python fallback)."""
    if _RUST_AVAILABLE:
        return np.asarray(
            _rust_kernel.savgol_filter(  # type: ignore[union-attr]
                np.ascontiguousarray(data, dtype=np.float64),
                int(window_length),
                int(polyorder),
            )
        )
    from ._filter_pure_python import _savgol_filter as _py_impl

    return _py_impl(data, window_length, polyorder)


def _median_filter(
    data: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Apply median filter (Rust kernel with Python fallback)."""
    if _RUST_AVAILABLE:
        return np.asarray(
            _rust_kernel.median_filter(  # type: ignore[union-attr]
                np.ascontiguousarray(data, dtype=np.float64),
                int(kernel_size),
            )
        )
    from ._filter_pure_python import _median_filter as _py_impl

    return _py_impl(data, kernel_size)


def _gaussian_filter(
    data: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Apply Gaussian filter (Rust kernel with Python fallback)."""
    if _RUST_AVAILABLE:
        return np.asarray(
            _rust_kernel.gaussian_filter(  # type: ignore[union-attr]
                np.ascontiguousarray(data, dtype=np.float64),
                float(sigma),
            )
        )
    from ._filter_pure_python import _gaussian_filter as _py_impl

    return _py_impl(data, sigma)


def _moving_average(
    data: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """Simple moving average filter."""
    if window < 2:
        return data

    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            filtered[:, i, j] = np.convolve(
                data[:, i, j], np.ones(window) / window, mode="same"
            )

    return filtered


# Import Keypoint and Marker for type hints
from ..contracts import Keypoint, Marker


def _kalman_filter(
    data: np.ndarray,
    process_noise: float = 0.01,
    measurement_noise: float = 0.1,
) -> np.ndarray:  # noqa: D401
    """Apply Kalman filter — Rust kernel preferred, pure-Python fallback below."""
    if _RUST_AVAILABLE:
        return np.asarray(
            _rust_kernel.kalman_filter(  # type: ignore[union-attr]
                np.ascontiguousarray(data, dtype=np.float64),
                float(process_noise),
                float(measurement_noise),
            )
        )
    return _kalman_filter_python(data, process_noise, measurement_noise)


def _kalman_filter_python(
    data: np.ndarray,
    process_noise: float = 0.01,
    measurement_noise: float = 0.1,
) -> np.ndarray:
    """Apply a 1D random-walk Kalman smoother (RTS) to each marker/keypoint coord.

    Uses a forward-pass Kalman filter followed by a backward-pass RTS smoother.
    This eliminates the forward-pass transient and gives optimal smoothing for
    offline (batch) mocap data where all frames are available.

    The steady-state P initialization avoids the long high-gain transient that
    would otherwise reduce smoothing quality for the first 20–30 frames when
    starting with P=1.0.

    Args:
        data: Array of shape (n_frames, n_points, n_dims).
        process_noise: Scalar process-noise variance Q.
        measurement_noise: Scalar measurement-noise variance R.

    Returns:
        Smoothed array of identical shape.
    """
    if data.size == 0:
        return data

    q = float(process_noise)
    r = float(measurement_noise)

    # Steady-state P for a random-walk model solves the DARE:
    #   P_ss = (P_ss + q) * r / (P_ss + q + r)
    # Positive root: P_ss = 0.5 * (-q + np.sqrt(q**2 + 4.0 * q * r))
    p_steady = 0.5 * (-q + np.sqrt(q**2 + 4.0 * q * r))

    filtered = np.zeros_like(data)
    n_frames = data.shape[0]

    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            series = data[:, i, j]

            # --- Forward pass ---
            x_fwd = np.empty(n_frames)
            p_fwd = np.empty(n_frames)

            p = p_steady
            x = float(series[0])

            for t in range(n_frames):
                # Predict (random-walk: x_k = x_{k-1}, P_k = P_{k-1} + Q)
                p_pred = p + q
                # Update (Kalman gain and correction)
                k_gain = p_pred / (p_pred + r)
                x = x + k_gain * (series[t] - x)
                p = (1.0 - k_gain) * p_pred
                x_fwd[t] = x
                p_fwd[t] = p

            # --- Backward RTS smoother pass ---
            smoothed = np.empty(n_frames)
            smoothed[-1] = x_fwd[-1]
            p_s = p_fwd[-1]

            for t in range(n_frames - 2, -1, -1):
                # Predicted covariance at t+1 (using forward filter's P at t)
                p_pred = p_fwd[t] + q
                # RTS smoother gain
                g_s = p_fwd[t] / p_pred
                smoothed[t] = x_fwd[t] + g_s * (smoothed[t + 1] - x_fwd[t])
                p_s = p_fwd[t] + g_s**2 * (p_s - p_pred)

            filtered[:, i, j] = smoothed

    return filtered


def _ewma(data: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Exponentially-weighted moving average fallback (no SciPy required)."""
    if data.size == 0:
        return data
    out = np.zeros_like(data)
    out[0] = data[0]
    for t in range(1, data.shape[0]):
        out[t] = alpha * data[t] + (1.0 - alpha) * out[t - 1]
    return out
