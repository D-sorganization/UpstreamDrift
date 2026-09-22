"""Explicit calibration records for constrained upper-body capture campaigns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.motion_matching.body_target import BodyTarget
from src.shared.python.motion_matching.projection_2d import estimate_swing_plane
from src.shared.python.pendulum_simulator.upper_body_replay import (
    UpperBodyCaptureFrame,
)


@dataclass(frozen=True)
class CaptureSourceClock:
    """Native C3D clock retained beside a resampled evaluation target."""

    raw_frame_count: int
    raw_frame_rate_hz: float

    def __post_init__(self) -> None:
        if self.raw_frame_count < 2:
            raise ValueError("raw_frame_count must be at least 2")
        if not np.isfinite(self.raw_frame_rate_hz) or self.raw_frame_rate_hz <= 0.0:
            raise ValueError("raw_frame_rate_hz must be finite and positive")


@dataclass(frozen=True)
class CaptureFrameCalibration:
    """One rigid capture frame estimated from named body markers."""

    frame: UpperBodyCaptureFrame
    marker_names: tuple[str, ...]
    plane_rmse_m: float
    plane_max_deviation_m: float


def _selected_marker_points(
    body_target: BodyTarget,
    marker_names: tuple[str, ...],
) -> np.ndarray:
    unavailable = sorted(set(marker_names).difference(body_target.marker_names))
    if unavailable:
        raise ValueError(f"body target does not provide markers: {unavailable}")
    indices = [body_target.marker_names.index(name) for name in marker_names]
    points = body_target.marker_xyz[:, indices, :].reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    selected = points[finite]
    if len(selected) < 3:
        raise ValueError("capture frame requires at least three finite marker samples")
    return selected


def calibrate_capture_frame(
    body_target: BodyTarget,
    marker_names: tuple[str, ...],
) -> CaptureFrameCalibration:
    """Fit one rigid plane to the declared marker observations.

    The calibration is fixed for the complete target clock. It records the
    unavoidable out-of-plane residual instead of flattening measurements.
    """
    if len(marker_names) < 3 or len(set(marker_names)) != len(marker_names):
        raise ValueError("marker_names must contain at least three unique names")
    plane = estimate_swing_plane(_selected_marker_points(body_target, marker_names))
    return CaptureFrameCalibration(
        frame=UpperBodyCaptureFrame(
            origin_m=np.asarray(plane.origin, dtype=np.float64),
            plane_basis=np.asarray(plane.basis[:, :2], dtype=np.float64),
        ),
        marker_names=marker_names,
        plane_rmse_m=float(plane.residual.rmse),
        plane_max_deviation_m=float(plane.residual.max_deviation),
    )
