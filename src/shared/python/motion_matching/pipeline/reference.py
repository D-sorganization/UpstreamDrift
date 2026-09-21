from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.signal import butter, filtfilt

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.full_body_spec import canonical_sha256
from src.shared.python.motion_matching.tour_capture_contract import (
    MARKER_VALIDITY_POLICY,
)
from src.shared.python.motion_matching.pipeline.constants import (
    BOUND_WIDENING,
    CALIBRATION_PRIOR_FRAMES,
    CALIBRATION_STRIDE,
    CONSISTENCY_PRIOR,
    LEG_SEEDS,
    LOWER_LIMB_RANGES_DEG,
    PLAYBACK_STRIDE,
    RATE_HZ,
    REFERENCE_CUTOFF_HZ,
    SCALE_GRID,
    SHOULDER_GIMBALS,
)

if TYPE_CHECKING:
    from src.engines.physics_engines.mujoco.python.full_body_ik import (
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


@precondition(lambda label, is_valid=True: isinstance(label, str), "label must be str")
@postcondition(lambda r: r >= 0.0, "weight must be non-negative")
def marker_weight(label: str, is_valid: bool = True) -> float:
    """Return marker tracking weight according to the canonical validity policy."""
    return MARKER_VALIDITY_POLICY.weight_for(label, is_valid=is_valid)


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
        from src.shared.python.motion_matching.full_body_ik import (
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
    from src.engines.physics_engines.mujoco.python.visual_layer import (
        render_playback as _render,
    )

    _render(
        spec_bytes=spec_bytes,
        names=names,
        q=q,
        lookat=lookat,
        path=path,
        show_com=show_com,
        playback_stride=PLAYBACK_STRIDE,
        rate_hz=RATE_HZ,
    )


@dataclass(frozen=True)
class IKReportInputs:
    """Inputs required to construct the full IK pipeline report."""

    lane: Lane
    kin: FullBodyMarkerKinematics
    adapter: Any
    q_ik: np.ndarray
    fits: list[Any]
    q_smooth: np.ndarray
    q_ref: np.ndarray
    ref_fits: list[Any]
    labels: tuple[str, ...]
    attachments: Mapping[str, tuple[str, Sequence[float]]]
    calibration: Any
    calibration2: Any
    femur_scale: float
    tibia_scale: float
    scale_table: list[dict[str, Any]]
    scaled_spec: dict[str, Any]
    offsets: Mapping[str, tuple[str, Sequence[float]]] | None = None
    errors: np.ndarray | None = None
    ref_errors: np.ndarray | None = None
    constrained_ik: dict[str, Any] | None = None


def _build_reference_stage_report(
    inputs: IKReportInputs,
    kin: FullBodyMarkerKinematics,
    lane: Lane,
    ref_errors: np.ndarray,
    ref_heights: np.ndarray,
) -> dict[str, Any]:
    from src.shared.python.motion_matching.pipeline.dynamics import segment_rms

    return {
        "cutoff_hz": REFERENCE_CUTOFF_HZ,
        "consistency_prior": CONSISTENCY_PRIOR,
        "marker_rms_m": float(np.sqrt(np.mean(ref_errors[lane.valid] ** 2))),
        "segment_rms_m": segment_rms(inputs.labels, ref_errors, lane.valid),
        "closure_error_max_m": float(max(f.closure_error_m for f in inputs.ref_fits)),
        "lowest_sphere_height_min_m": float(ref_heights.min()),
        "lowest_sphere_height_max_m": float(ref_heights.max()),
        "max_deviation_from_smoothed_rad": float(
            np.abs(inputs.q_ref[:, 6:] - inputs.q_smooth[:, 6:]).max()
        ),
        "max_joint_speed_rad_s": float(
            np.abs(np.gradient(inputs.q_ref[:, 6:], lane.times, axis=0)).max()
        ),
        "stance_sphere_drift_max_m": float(
            max(
                [
                    float(
                        np.linalg.norm(
                            kin.sphere_ground_points(inputs.q_ref[k], lane.ground)[name]
                            - kin.sphere_ground_points(inputs.q_ref[0], lane.ground)[
                                name
                            ]
                        )
                    )
                    for k in range(lane.frames)
                    for name in lane.stance[k]
                    if all(name in lane.stance[j] for j in range(k + 1))
                ]
            )
        ),
    }


def _build_calibration_stage_report(
    inputs: IKReportInputs,
    lane: Lane,
) -> dict[str, Any]:
    custom_offsets = inputs.offsets
    cal2 = inputs.calibration2
    cal2_offsets = getattr(cal2, "offsets", None)
    if custom_offsets is not None:
        source_offsets = custom_offsets
    elif cal2_offsets is not None:
        source_offsets = cal2_offsets
    else:
        source_offsets = inputs.attachments

    return {
        "stride": CALIBRATION_STRIDE,
        "frames": len(lane.calibration_frames),
        "prior_frames": CALIBRATION_PRIOR_FRAMES,
        "prior_offsets_m": {k: list(v[1]) for k, v in LEG_SEEDS.items()},
        "rms_per_iteration_m": list(inputs.calibration.rms_per_iteration_m),
        "rms_per_iteration_after_scaling_m": list(cal2.rms_per_iteration_m),
        "per_marker_rms_m": cal2.per_marker_rms_m,
        "offsets_m": {
            k: {"body": b, "offset_m": list(o)} for k, (b, o) in source_offsets.items()
        },
    }


def build_ik_report(inputs: IKReportInputs) -> dict[str, Any]:
    """Build the structured IK and reference trajectory report dictionary.

    Preconditions:
    - ``inputs.q_ik`` and ``inputs.q_ref`` must be 2D arrays with frame count matching ``inputs.lane``.

    Postconditions:
    - Returns dictionary matching the full-body IK receipt specification.
    """
    from src.shared.python.motion_matching.pipeline.dynamics import (
        rom_flags,
        segment_rms,
    )

    lane = inputs.lane
    kin = inputs.kin
    adapter = inputs.adapter
    errors = (
        inputs.errors
        if inputs.errors is not None
        else marker_errors(kin, inputs.q_ik, lane.points)
    )
    ref_errors = (
        inputs.ref_errors
        if inputs.ref_errors is not None
        else marker_errors(kin, inputs.q_ref, lane.points)
    )
    heights = np.array(
        [min(kin.sphere_heights(q, lane.ground).values()) for q in inputs.q_ik]
    )
    ref_heights = np.array(
        [min(kin.sphere_heights(q, lane.ground).values()) for q in inputs.q_ref]
    )

    report = {
        "frames": lane.frames,
        "marker_rms_m": float(np.sqrt(np.mean(errors[lane.valid] ** 2))),
        "segment_rms_m": segment_rms(inputs.labels, errors, lane.valid),
        "closure_error_max_m": float(max(f.closure_error_m for f in inputs.fits)),
        "lowest_sphere_height_min_m": float(heights.min()),
        "lowest_sphere_height_max_m": float(heights.max()),
        "attachments_m": {
            label: {"body": body, "offset_m": [float(v) for v in offset]}
            for label, (body, offset) in inputs.attachments.items()
        },
        "reference": _build_reference_stage_report(
            inputs, kin, lane, ref_errors, ref_heights
        ),
        "calibration": _build_calibration_stage_report(inputs, lane),
        "segment_scaling": {
            "grid": SCALE_GRID,
            "table": inputs.scale_table,
            "femur_scale": inputs.femur_scale,
            "tibia_scale": inputs.tibia_scale,
            "spec_sha256": canonical_sha256(inputs.scaled_spec),
        },
        "joint_ranges_deg": LOWER_LIMB_RANGES_DEG,
        "range_of_motion_flags": rom_flags(inputs.q_ref, kin.coordinate_order),
        "bound_widening": BOUND_WIDENING,
        "leg_angle_ranges_deg": {
            name: [
                float(
                    np.degrees(inputs.q_ref[:, kin.coordinate_order.index(name)].min())
                ),
                float(
                    np.degrees(inputs.q_ref[:, kin.coordinate_order.index(name)].max())
                ),
            ]
            for name in kin.coordinate_order[adapter.upper_body_coordinates :]
        },
    }
    if inputs.constrained_ik is not None:
        report["constrained_ik"] = inputs.constrained_ik
    return report
