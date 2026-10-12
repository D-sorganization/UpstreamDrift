"""Neutral swing posture metrics on the canonical event frames (Issue #11026).

Package 3 of epic #11007. The module reports four continuous geometric
metrics computed from a pelvis position series and a trunk axis series,
expressed in an address-aligned frame defined by two caller-supplied unit
vectors:

- ``target_line``: horizontal unit vector toward the target.
- ``toward_ball``: horizontal unit vector from the golfer toward the ball.

Metric names are neutral and geometric. Any mapping to coaching vocabulary
belongs in documentation only, as with ``kinematic_sequence.py``.

Thresholds are never built in. A caller supplies thresholds together with a
provenance string, and :func:`evaluate_thresholds` reports only whether a
metric exceeds the supplied value. Validating thresholds needs labelled
swings and is a deferred-validation item.

Frames: ``events.address`` is the address frame, ``events.top`` the top of the
backswing and ``events.peak`` the impact proxy (see ``swing_events``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.analysis.swing_events import SwingEventFrames
from src.shared.python.core.contracts import ensure, require

METRIC_NAMES: tuple[str, ...] = (
    "pelvis_toward_ball_shift_m",
    "pelvis_lateral_shift_backswing_m",
    "pelvis_lateral_shift_downswing_m",
    "trunk_forward_tilt_change_deg",
)

AXIS_TOLERANCE = 1e-6


@dataclass(frozen=True)
class SwingPostureMetrics:
    """Continuous posture metrics.

    Attributes:
        pelvis_toward_ball_shift_m: Pelvis displacement along ``toward_ball``,
            address to impact (m).
        pelvis_lateral_shift_backswing_m: Displacement along ``target_line``,
            address to top (m).
        pelvis_lateral_shift_downswing_m: Displacement along ``target_line``,
            top to impact (m).
        trunk_forward_tilt_change_deg: Change in trunk forward tilt, address
            to impact (deg). Tilt is the angle of the trunk axis from vertical
            toward the ball.
    """

    pelvis_toward_ball_shift_m: float
    pelvis_lateral_shift_backswing_m: float
    pelvis_lateral_shift_downswing_m: float
    trunk_forward_tilt_change_deg: float

    def __post_init__(self) -> None:
        for f in fields(self):
            ensure(
                math.isfinite(getattr(self, f.name)),
                f"{f.name} must be finite",
                getattr(self, f.name),
            )


def _as_vector3(value: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    require(arr.shape == (3,), f"{name} must have shape (3,), got {arr.shape}", arr)
    require(bool(np.all(np.isfinite(arr))), f"{name} must be finite", arr)
    return arr


def _as_series(value: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    require(
        arr.ndim == 2 and arr.shape[1] == 3,
        f"{name} must have shape (N, 3), got {arr.shape}",
        arr,
    )
    require(bool(np.all(np.isfinite(arr))), f"{name} must be finite", arr)
    return arr


def _validated_frame(
    target_line: ArrayLike, toward_ball: ArrayLike, tolerance: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (target, toward_ball, vertical) as an orthonormal triad."""
    require(
        math.isfinite(tolerance) and tolerance > 0, "tolerance must be > 0", tolerance
    )
    target = _as_vector3(target_line, "target_line")
    ball = _as_vector3(toward_ball, "toward_ball")
    require(
        abs(float(np.linalg.norm(target)) - 1.0) <= tolerance,
        "target_line must be a unit vector",
        target,
    )
    require(
        abs(float(np.linalg.norm(ball)) - 1.0) <= tolerance,
        "toward_ball must be a unit vector",
        ball,
    )
    require(
        abs(float(target @ ball)) <= tolerance,
        "target_line and toward_ball must be orthogonal",
        float(target @ ball),
    )
    return target, ball, np.cross(target, ball)


def _forward_tilt_deg(
    axis: np.ndarray, ball: np.ndarray, vertical: np.ndarray
) -> float:
    """Angle of the trunk axis from vertical, positive toward the ball."""
    return math.degrees(math.atan2(float(axis @ ball), abs(float(axis @ vertical))))


def compute_swing_posture_metrics(
    pelvis_positions: ArrayLike,
    trunk_axes: ArrayLike,
    events: SwingEventFrames,
    target_line: ArrayLike,
    toward_ball: ArrayLike,
    *,
    axis_tolerance: float = AXIS_TOLERANCE,
) -> SwingPostureMetrics:
    """Compute the four neutral posture metrics.

    Args:
        pelvis_positions: (N, 3) world-frame pelvis translation (m).
        trunk_axes: (N, 3) trunk long-axis direction, pelvis toward neck.
            Need not be unit length but must be non-zero at the used frames.
        events: Event frames; ``events.peak`` is used as impact.
        target_line: Unit vector toward the target.
        toward_ball: Unit vector toward the ball, orthogonal to ``target_line``.
        axis_tolerance: Allowed deviation from orthonormality.

    Returns:
        Frozen :class:`SwingPostureMetrics`.

    Raises:
        ValueError: Non-orthonormal axes, bad shapes, mismatched lengths,
            non-finite values, event frames out of range, or a zero trunk axis.

    Invariance: results are unchanged by translating the capture or by
    rotating it, with both axes, about the vertical.
    """
    target, ball, vertical = _validated_frame(target_line, toward_ball, axis_tolerance)
    pelvis = _as_series(pelvis_positions, "pelvis_positions")
    trunk = _as_series(trunk_axes, "trunk_axes")
    require(
        len(pelvis) == len(trunk),
        "pelvis_positions and trunk_axes must have equal length",
        (len(pelvis), len(trunk)),
    )
    require(
        events.finish < len(pelvis),
        f"event frames out of range for {len(pelvis)} samples (finish={events.finish})",
        events,
    )
    a, top, impact = events.address, events.top, events.peak
    for idx in (a, impact):
        require(
            float(np.linalg.norm(trunk[idx])) > 0.0,
            f"trunk_axes[{idx}] must be non-zero",
            trunk[idx],
        )

    shift_ab = pelvis[impact] - pelvis[a]
    tilt_a = _forward_tilt_deg(trunk[a] / np.linalg.norm(trunk[a]), ball, vertical)
    tilt_i = _forward_tilt_deg(
        trunk[impact] / np.linalg.norm(trunk[impact]), ball, vertical
    )
    return SwingPostureMetrics(
        pelvis_toward_ball_shift_m=float(shift_ab @ ball),
        pelvis_lateral_shift_backswing_m=float((pelvis[top] - pelvis[a]) @ target),
        pelvis_lateral_shift_downswing_m=float((pelvis[impact] - pelvis[top]) @ target),
        trunk_forward_tilt_change_deg=tilt_i - tilt_a,
    )


def evaluate_thresholds(
    metrics: SwingPostureMetrics,
    thresholds: Mapping[str, float],
    provenance: str | None,
) -> dict[str, bool]:
    """Report which metrics exceed caller-supplied thresholds.

    A metric exceeds its threshold when its magnitude is strictly greater
    than the threshold (equality does not exceed). Thresholds are limits on
    magnitude, so they must be finite and non-negative. Only the metrics named
    in ``thresholds`` are reported.

    Args:
        metrics: Computed metrics.
        thresholds: Metric name to magnitude limit, in the metric's unit.
        provenance: Non-empty description of where the thresholds come from.

    Raises:
        ValueError: Missing provenance, empty mapping, unknown metric name, or
            a negative or non-finite threshold.
    """
    require(
        isinstance(provenance, str) and bool(provenance.strip()),
        "thresholds require a non-empty provenance string",
        provenance,
    )
    require(len(thresholds) > 0, "thresholds must not be empty", thresholds)
    unknown = sorted(set(thresholds) - set(METRIC_NAMES))
    require(not unknown, f"unknown metric names: {unknown}", unknown)
    result: dict[str, bool] = {}
    for name, limit in thresholds.items():
        require(
            math.isfinite(limit) and limit >= 0,
            f"threshold for {name} must be finite and >= 0",
            limit,
        )
        result[name] = abs(getattr(metrics, name)) > limit
    return result
