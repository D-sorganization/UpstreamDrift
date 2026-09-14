"""Alternating marker-placement calibration for body-fixed marker sets.

Marker offsets in body frames are unknown for the tour capture (no static
trial). This module alternates two injected operations until the marker fit
stops improving: (1) place each marker by expressing its observed world
position in its body frame at the current pose, averaged over the calibration
frames; (2) solve inverse kinematics with those placements. Forward
kinematics and IK are supplied as callables so the algorithm is engine-agnostic
and reusable across Pinocchio, MuJoCo, Drake, and OpenSim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.body_part_viz.fitters._kabsch import kabsch_rotation
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

Array = NDArray[np.float64]
Pose = tuple[Array, Array]
Offsets = dict[str, tuple[str, tuple[float, float, float]]]
PoseFn = Callable[[Array], Mapping[str, Pose]]
IkFn = Callable[[Offsets, TourCapture], Array]


def rigid_pose_from_markers(body_points: Array, world_points: Array) -> Pose:
    """Return (R, t) with ``world ≈ body @ R.T + t`` from >=3 non-collinear pairs."""
    p, q = np.asarray(body_points, float), np.asarray(world_points, float)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 3:
        raise ValueError("Rigid pose needs matching (N>=3, 3) point sets")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Rigid pose needs finite points")
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    rotation = kabsch_rotation(p - pc, q - qc)
    return rotation, qc - rotation @ pc


def express_in_body(points_world: Array, rotation: Array, translation: Array) -> Array:
    """Map world points into the body frame given its world pose (R, t)."""
    return (np.asarray(points_world, float) - translation) @ rotation


@dataclass(frozen=True)
class CalibrationResult:
    """Calibrated offsets, the final IK trajectory and the fit history."""

    offsets: Offsets
    q: Array
    rms_per_iteration_m: tuple[float, ...]
    iterations: int
    best_iteration: int
    per_marker_rms_m: dict[str, float]


def _placements(
    capture: TourCapture, bodies: Mapping[str, str], poses: list[Mapping[str, Pose]]
) -> Offsets:
    offsets: Offsets = {}
    for i, label in enumerate(capture.labels):
        body = bodies[label]
        local = [
            express_in_body(capture.points_m[f, i], *poses[f][body])
            for f in range(capture.frames)
            if capture.valid[f, i]
        ]
        if not local:
            raise ValueError(f"Marker {label} has no valid calibration frame")
        mean = np.mean(np.asarray(local), axis=0)
        offsets[label] = (body, (float(mean[0]), float(mean[1]), float(mean[2])))
    return offsets


def _marker_rms(
    capture: TourCapture, offsets: Offsets, poses: list[Mapping[str, Pose]]
) -> float:
    rms, _ = _marker_rms_and_per_marker(capture, offsets, poses)
    return rms


def _marker_rms_and_per_marker(
    capture: TourCapture, offsets: Offsets, poses: list[Mapping[str, Pose]]
) -> tuple[float, dict[str, float]]:
    all_errors: list[float] = []
    per_marker: dict[str, float] = {}
    for i, label in enumerate(capture.labels):
        body, offset = offsets[label]
        marker_errors: list[float] = []
        for f in range(capture.frames):
            if capture.valid[f, i]:
                r, t = poses[f][body]
                err = float(
                    np.linalg.norm(r @ np.asarray(offset) + t - capture.points_m[f, i])
                )
                marker_errors.append(err)
                all_errors.append(err)
        per_marker[label] = (
            float(np.sqrt(np.mean(np.square(marker_errors)))) if marker_errors else 0.0
        )
    total_rms = float(np.sqrt(np.mean(np.square(all_errors)))) if all_errors else 0.0
    return total_rms, per_marker


def calibrate_marker_offsets(
    capture: TourCapture,
    bodies: Mapping[str, str],
    pose_fn: PoseFn,
    ik_fn: IkFn,
    *,
    initial_q: Array,
    iterations: int,
) -> CalibrationResult:
    """Alternate placement and IK; return offsets, final q and RMS history.

    Preconditions: every capture label has a body; iterations >= 1; pose_fn
    maps a coordinate vector to world poses of every referenced body; ik_fn
    returns one coordinate row per capture frame. Postcondition: the RMS
    history has one entry per iteration, each measured after that iteration's
    IK with that iteration's placements. The returned offsets, q and
    per_marker_rms_m correspond to the best iteration (lowest total RMS).
    """
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < 1
    ):
        raise ValueError("iterations must be a positive integer")
    missing = [label for label in capture.labels if label not in bodies]
    if missing:
        raise ValueError(f"Capture labels without a body: {missing}")
    q0 = np.asarray(initial_q, float)
    if q0.ndim != 1 or not np.isfinite(q0).all():
        raise ValueError("initial_q must be a finite vector")
    q = np.tile(q0, (capture.frames, 1))
    history: list[float] = []

    best_iteration = 1
    best_rms = float("inf")
    best_offsets: Offsets = {}
    best_q: Array = np.empty(0)
    best_per_marker: dict[str, float] = {}

    for iter_idx in range(1, iterations + 1):
        poses = [pose_fn(q[f]) for f in range(capture.frames)]
        offsets = _placements(capture, bodies, poses)
        q = np.asarray(ik_fn(offsets, capture), float)
        if q.shape[0] != capture.frames or q.ndim != 2 or not np.isfinite(q).all():
            raise ValueError("IK must return a finite (frames, coordinates) array")
        poses = [pose_fn(q[f]) for f in range(capture.frames)]
        rms, per_marker = _marker_rms_and_per_marker(capture, offsets, poses)
        history.append(rms)
        if rms < best_rms:
            best_rms = rms
            best_iteration = iter_idx
            best_offsets = offsets
            best_q = q.copy()
            best_per_marker = per_marker

    return CalibrationResult(
        best_offsets,
        best_q,
        tuple(history),
        iterations,
        best_iteration,
        best_per_marker,
    )
