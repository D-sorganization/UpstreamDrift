"""Reference trajectory, smoothing, and consistency re-solve for full-body pipeline."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imageio
import mujoco
import numpy as np
from scipy.signal import butter, filtfilt

from src.shared.python.motion_matching.pipeline.constants import (
    CONSISTENCY_PRIOR,
    PLAYBACK_STRIDE,
    RATE_HZ,
    SHOULDER_GIMBALS,
)

if TYPE_CHECKING:
    from src.engines.physics_engines.mujoco.python.full_body_markers import (
        FullBodyMarkerKinematics,
    )
    from src.shared.python.motion_matching.pipeline.lane import Lane


def smooth_reference(q: np.ndarray, rate_hz: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase Butterworth low-pass of every coordinate (edge-padded).

    Args:
        q: (N, nq) array of coordinate trajectories.
        rate_hz: Sampling rate in Hz (> 0).
        cutoff_hz: Filter cutoff frequency in Hz (0 < cutoff_hz < 0.5 * rate_hz).

    Returns:
        (N, nq) array of smoothed coordinate trajectories.
    """
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz}")
    if cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}")
    nyquist = 0.5 * rate_hz
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"cutoff_hz ({cutoff_hz}) must be strictly less than Nyquist ({nyquist})"
        )
    if q.ndim != 2:
        raise ValueError(f"q must be a 2D array, got shape {q.shape}")
    if q.shape[0] < 2:
        return q.copy()

    b, a = butter(4, cutoff_hz / nyquist)
    padlen = min(60, q.shape[0] - 1)
    return filtfilt(b, a, q, axis=0, padlen=padlen)


def marker_errors(
    kin: FullBodyMarkerKinematics, q: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Compute Euclidean marker tracking errors per frame and marker.

    Args:
        kin: Marker kinematics model.
        q: (N, nq) coordinate array.
        points: (N, M, 3) target marker positions.

    Returns:
        (N, M) Euclidean errors in metres.
    """
    if q.ndim != 2:
        raise ValueError(f"q must be a 2D array, got shape {q.shape}")
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(
            f"points must be a 3D array of shape (N, M, 3), got {points.shape}"
        )
    if q.shape[0] != points.shape[0]:
        raise ValueError(
            f"Frame count mismatch: q has {q.shape[0]}, points has {points.shape[0]}"
        )

    return np.array(
        [
            np.linalg.norm(kin.marker_positions(row) - target, axis=1)
            for row, target in zip(q, points, strict=True)
        ]
    )


def full_capture_ik(
    lane: Lane,
    kin: FullBodyMarkerKinematics,
    q_start: np.ndarray,
) -> tuple[np.ndarray, list[Any]]:
    """Solve full-capture inverse kinematics and unwrap shoulder gimbals.

    Args:
        lane: Coordination lane providing capture data, bounds, and stance.
        kin: Marker kinematics model.
        q_start: Initial coordinate vector (nq,).

    Returns:
        (q_ik, fits): unwrapped coordinate trajectory (N, nq) and fit results list.
    """
    if q_start.ndim != 1 or len(q_start) != kin.nq:
        raise ValueError(f"q_start must have shape ({kin.nq},), got {q_start.shape}")

    q_ik, fits = lane.trajectory(kin, q_start)
    # Continuous coordinates: unwrap and keep each shoulder gimbal on Euler branch
    # nearest previous frame before smoothing, if gimbals are present in the model.
    active_gimbals = [
        triple
        for triple in SHOULDER_GIMBALS
        if all(name in kin.coordinate_order for name in triple)
    ]
    if active_gimbals:
        from src.engines.physics_engines.mujoco.python.full_body_markers import (
            continuous_branches,
        )

        q_ik = continuous_branches(q_ik, kin.coordinate_order, active_gimbals)
    return q_ik, fits


def consistency_resolve(
    lane: Lane,
    kin: FullBodyMarkerKinematics,
    q_smooth: np.ndarray,
    *,
    prior_weight: float = CONSISTENCY_PRIOR,
    iterations: int = 30,
) -> tuple[np.ndarray, list[Any]]:
    """Re-solve trajectory to keep smoothed reference on the ground with grip closed.

    Args:
        lane: Coordination lane providing capture data, bounds, stance, and ground.
        kin: Marker kinematics model.
        q_smooth: Smoothed coordinate trajectory (N, nq).
        prior_weight: Weight pulling the re-solve toward smoothed reference.
        iterations: Solver iterations per frame.

    Returns:
        (q_ref, ref_fits): re-solved trajectory and fit summaries.
    """
    if q_smooth.ndim != 2 or q_smooth.shape[0] != lane.frames:
        raise ValueError(
            f"q_smooth must have shape ({lane.frames}, {kin.nq}), got {q_smooth.shape}"
        )
    if prior_weight < 0:
        raise ValueError(f"prior_weight must be non-negative, got {prior_weight}")
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")

    q_ref, ref_fits = kin.solve_trajectory(
        lane.points,
        lane.valid,
        q_smooth[0],
        ground=lane.ground,
        prior_weight=prior_weight,
        iterations=iterations,
        flat_feet_per_frame=lane.stance,
        plant_stance=True,
        prior_trajectory=q_smooth,
        bounds=lane.bounds,
    )
    return q_ref, ref_fits


def render_playback(
    spec_bytes: bytes,
    names: Sequence[str],
    q: np.ndarray,
    lookat: np.ndarray,
    path: Path,
    show_com: bool = True,
) -> None:
    """Render animated GIF of motion from spec and joint trajectory."""
    from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter
    from src.engines.physics_engines.mujoco.python.visual_layer import add_com_markers

    xml, _ = exporter.export_full_body_mjcf(spec_bytes, visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    addresses = [model.joint(n).qposadr[0] for n in names]
    ground_height = float(
        json.loads(spec_bytes)["contact"].get("ground_height_m") or 0.0
    )
    renderer = mujoco.Renderer(model, 240, 320)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = 3.2, 135.0, -12.0
    frames_out = []
    for k in range(0, q.shape[0], PLAYBACK_STRIDE):
        data.qpos[addresses] = q[k]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        if show_com:
            add_com_markers(renderer.scene, model, data, ground_height)
        frames_out.append(renderer.render().copy())
    imageio.mimsave(path, frames_out, duration=1000 * PLAYBACK_STRIDE / RATE_HZ, loop=0)
