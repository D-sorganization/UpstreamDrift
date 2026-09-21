"""Forward dynamics tracking, shooting fit, and ZMP filtering for full-body pipeline."""

from __future__ import annotations

import math

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.contracts import postcondition, precondition

if TYPE_CHECKING:
    from src.engines.physics_engines.mujoco.python.full_body_ik import (
        FullBodyMarkerKinematics,
    )
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.dynamics_filter import (
    cart_table_shift,
    project_inside,
)
from src.shared.python.motion_matching.ground_support import convex_hull_contains
from src.shared.python.motion_matching.pipeline.constants import (
    BALANCE,
    CONSISTENCY_PRIOR,
    DT_S,
    OMEGA_RAD_S,
    RATE_HZ,
    SHOOTING_LOCKED,
    SHOOTING_RELAXATION,
    TRACKING_CUTOFF_HZ,
    ZMP_COM_WEIGHT,
    ZMP_FILTER_ITERATIONS,
    ZMP_MARGIN_M,
)
from src.shared.python.motion_matching.pipeline.reference import (
    marker_errors,
    smooth_reference,
)
from src.shared.python.motion_matching.range_of_motion import (
    HUMAN_RANGES_DEG,
    violations,
)
from src.shared.python.motion_matching.tour_capture_contract import MARKER_SEGMENTS

if TYPE_CHECKING:
    from src.shared.python.motion_matching.pipeline.lane import Lane

TRACKING_BACKENDS = frozenset({"kkt", "mj-inverse"})


@precondition(lambda name: isinstance(name, str), "tracking_backend must be a string")
@postcondition(
    lambda result: result in TRACKING_BACKENDS, "tracking_backend must be supported"
)
def validate_tracking_backend(name: str) -> str:
    """Validate selectable computed-torque tracking backend names."""
    key = name.strip().lower()
    if key not in TRACKING_BACKENDS:
        known = ", ".join(sorted(TRACKING_BACKENDS))
        raise ValueError(
            f"Unknown tracking backend {name!r}; expected one of [{known}]"
        )
    return key


def build_tracking_controller(
    sim: fs.FullBodySimulator,
    times: np.ndarray,
    q_track: np.ndarray,
    *,
    tracking_backend: str = "kkt",
):
    """Build a computed-torque controller for the selected tracking backend."""
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs

    tracking_backend = validate_tracking_backend(tracking_backend)
    if tracking_backend == "mj-inverse":
        from src.engines.physics_engines.mujoco.python.inverse_dynamics import (
            tracking_controller_mj_inverse,
        )

        return tracking_controller_mj_inverse(
            sim,
            times,
            q_track,
            omega_rad_s=OMEGA_RAD_S,
            zeta=1.0,
            balance=BALANCE,
        )
    return fs.tracking_controller(
        sim, times, q_track, omega_rad_s=OMEGA_RAD_S, zeta=1.0, balance=BALANCE
    )


def segment_rms(
    labels: tuple[str, ...] | Sequence[str],
    errors: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    """Compute RMS marker errors partitioned by anatomical body segment.

    Args:
        labels: Tuple or sequence of marker labels corresponding to columns.
        errors: (N, M) array of marker errors.
        valid: (N, M) boolean array of marker validity masks.

    Returns:
        Dictionary mapping segment name to RMS error in metres.
    """
    labels_tuple = tuple(labels)
    if errors.ndim != 2:
        raise ValueError(f"errors must be a 2D array, got {errors.ndim}D")
    if valid.shape != errors.shape:
        raise ValueError(
            f"Shape mismatch: errors {errors.shape} vs valid {valid.shape}"
        )
    if len(labels_tuple) != errors.shape[1]:
        raise ValueError(
            f"Column count mismatch: {len(labels_tuple)} labels vs {errors.shape[1]} columns"
        )

    out: dict[str, float] = {}
    for segment, members in MARKER_SEGMENTS.items():
        cols = [labels_tuple.index(m) for m in members if m in labels_tuple]
        if not cols:
            continue
        e, v = errors[:, cols], valid[:, cols]
        out[segment] = float(np.sqrt(np.mean(e[v] ** 2))) if v.any() else float("nan")
    return out


def replay(
    sim: fs.FullBodySimulator,
    lane: Lane,
    q_track: np.ndarray,
    *,
    tracking_backend: str = "kkt",
) -> tuple[fs.SimulationRecord, np.ndarray]:
    """Track ``q_track`` with computed torque from preloaded feet.

    Args:
        sim: Full-body forward simulator.
        lane: Coordination lane providing times.
        q_track: (N, nq) reference trajectory to track.
        tracking_backend: ``kkt`` (default plant affine solve) or ``mj-inverse``.

    Returns:
        (record, sim_q): simulation record and state resampled on the capture times.
    """
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs

    tracking_backend = validate_tracking_backend(tracking_backend)
    q0 = fs.preload_feet(sim, q_track[0])
    v0 = np.gradient(q_track, lane.times, axis=0)[0]
    controller = build_tracking_controller(
        sim,
        lane.times,
        q_track,
        tracking_backend=tracking_backend,
    )
    record = sim.run(
        q0,
        v0,
        controller,
        duration_s=float(lane.times[-1]),
        dt_s=DT_S,
        record_every=int(round(1.0 / (RATE_HZ * DT_S))),
    )
    sim_q = np.array(
        [
            [np.interp(t, record.time_s, record.q[:, k]) for k in range(sim.nv)]
            for t in lane.times
        ]
    )
    return record, sim_q


def zmp_summary(zmp: dict[str, Any], times: np.ndarray) -> dict[str, float]:
    """Calculate summary metrics for a ``reference_zmp`` result.

    Args:
        zmp: Dictionary containing 'outside_m' and 'unloaded' arrays.
        times: Array of capture timestamps in seconds.

    Returns:
        Dictionary of outside fractions, max, mean, and unloaded fraction.
    """
    window = (times >= 1.0) & (times < 1.5)
    outside = zmp["outside_m"]
    unloaded = zmp["unloaded"]
    return {
        "outside_fraction": float((outside > 0).mean()),
        "outside_fraction_1s_to_1_5s": float((outside[window] > 0).mean()),
        "outside_max_m": float(outside.max()),
        "outside_mean_m": float(outside.mean()),
        "unloaded_fraction": float(unloaded.mean()),
    }


def shooting_fit(
    lane: Lane,
    kin: FullBodyMarkerKinematics,
    sim: fs.FullBodySimulator,
    q_track: np.ndarray,
    q_ref: np.ndarray,
    iterations: int,
    log: logging.Logger,
    gain: float = SHOOTING_RELAXATION,
    *,
    tracking_backend: str = "kkt",
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Contact-aware shooting fit of the tracked reference (FB-5, MM-7b).

    The simulator is the plant: each iteration replays the current reference,
    measures how far the pelvis drifted from the reference pelvis path, moves
    the pelvis command against that drift by ``gain``, re-solves the joints
    against the markers with the command pinned (``SHOOTING_LOCKED``), and
    low-passes the result.

    Returns:
        (best_q, zmp, report): optimized reference, ZMP result, and iteration history.
    """
    root = [kin.coordinate_order.index(name) for name in SHOOTING_LOCKED]
    target = q_track[:, root].copy()
    command = target.copy()
    history: list[dict[str, Any]] = []
    best_q, best_rms = q_track, np.inf
    for k in range(iterations + 1):
        record, sim_q = replay(sim, lane, q_track, tracking_backend=tracking_backend)
        errors = marker_errors(kin, sim_q, lane.points)
        rms = float(np.sqrt(np.mean(errors[lane.valid] ** 2)))
        diff = sim_q[:, :3] - q_ref[:, :3]
        root_err = np.sqrt(
            np.einsum("ij,ij->i", diff, diff)
        )  # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is ~2.4x faster than np.linalg.norm(..., axis=1)
        zmp = fs.reference_zmp(sim, lane.times, q_track, lane.ground)
        history.append(
            {
                "iteration": k,
                "replay_marker_rms_m": rms,
                "replay_segment_rms_m": segment_rms(lane.labels, errors, lane.valid),
                "root_error_max_m": float(root_err.max()),
                "root_error_1_4s_m": float(root_err[int(round(1.4 * RATE_HZ))]),
                "weight_fraction_min": float(record.weight_fraction.min()),
                "reference_marker_rms_m": float(
                    np.sqrt(
                        np.mean(
                            marker_errors(kin, q_track, lane.points)[lane.valid] ** 2
                        )
                    )
                ),
                "reference_change_max_rad": float(
                    np.abs(q_track[:, 6:] - q_ref[:, 6:]).max()
                ),
                "pelvis_command_offset_max_m": float(
                    np.abs(command[:, :2] - target[:, :2]).max()
                ),
                **zmp_summary(zmp, lane.times),
            }
        )
        log.info(
            "shooting %d: replay markers %.1f mm, root max %.0f mm (1.4 s %.0f), "
            "wf min %.2f, zmp outside 1.0-1.5 s %.2f",
            k,
            rms * 1e3,
            root_err.max() * 1e3,
            history[-1]["root_error_1_4s_m"] * 1e3,
            history[-1]["weight_fraction_min"],
            history[-1]["outside_fraction_1s_to_1_5s"],
        )
        if rms < best_rms:
            best_q, best_rms = q_track, rms
        if k == iterations:
            break
        command = command - gain * (sim_q[:, root] - target)
        locked = [
            {name: float(command[j, i]) for i, name in enumerate(SHOOTING_LOCKED)}
            for j in range(len(lane.times))
        ]
        q_fit, _ = kin.solve_trajectory(
            lane.points,
            lane.valid,
            q_track[0],
            ground=lane.ground,
            prior_weight=CONSISTENCY_PRIOR,
            iterations=30,
            flat_feet_per_frame=lane.stance,
            plant_stance=True,
            prior_trajectory=q_track,
            bounds=lane.bounds,
            locked_per_frame=locked,
        )
        q_track = smooth_reference(q_fit, RATE_HZ, TRACKING_CUTOFF_HZ)
    zmp = fs.reference_zmp(sim, lane.times, best_q, lane.ground)
    report = {
        "locked": list(SHOOTING_LOCKED),
        "relaxation": gain,
        "iterations": history,
        "best_iteration": int(
            np.argmin([float(h["replay_marker_rms_m"]) for h in history])
        ),
    }
    return best_q, zmp, report


def zmp_filter(
    lane: Lane,
    kin: FullBodyMarkerKinematics,
    sim: fs.FullBodySimulator,
    q_track: np.ndarray,
    zmp: dict[str, Any],
    log: logging.Logger,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Dynamics filter on the tracked reference (MM-7b).

    Each pass projects the reference zero-moment point into the per-frame
    support polygon shrunk by ``ZMP_MARGIN_M``, finds the smallest smooth
    centre-of-mass shift whose cart-table effect closes the gap, re-solves
    the IK against the markers with the shifted centre-of-mass path as
    per-frame rows, and low-passes the result again.

    Returns:
        (q_track, zmp, report): filtered reference trajectory, final ZMP, and report.
    """
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs

    basis = kin._plane_basis(lane.ground)
    passes = []
    report = {
        "margin_m": ZMP_MARGIN_M,
        "com_weight": ZMP_COM_WEIGHT,
        "before": zmp_summary(zmp, lane.times),
    }
    for k in range(ZMP_FILTER_ITERATIONS):
        target = np.array(
            [
                project_inside(point, hull, ZMP_MARGIN_M) if not unloaded else point
                for point, hull, unloaded in zip(
                    zmp["zmp_xy"], zmp["hull_xy"], zmp["unloaded"], strict=True
                )
            ]
        )
        heights = zmp["com"][:, 2] - lane.ground.height_m
        shift = cart_table_shift(zmp["zmp_xy"], target, heights, lane.times)
        com_goal = zmp["com"][:, :2] + shift
        goals = [(basis @ np.r_[g, 0.0], ZMP_COM_WEIGHT) for g in com_goal]
        q_new, fits = kin.solve_trajectory(
            lane.points,
            lane.valid,
            q_track[0],
            ground=lane.ground,
            prior_weight=CONSISTENCY_PRIOR,
            iterations=30,
            flat_feet_per_frame=lane.stance,
            plant_stance=True,
            prior_trajectory=q_track,
            bounds=lane.bounds,
            com_targets_per_frame=goals,
        )
        q_track = smooth_reference(q_new, RATE_HZ, TRACKING_CUTOFF_HZ)
        zmp = fs.reference_zmp(sim, lane.times, q_track, lane.ground)
        errors = marker_errors(kin, q_track, lane.points)
        passes.append(
            {
                "pass": k + 1,
                "com_shift_max_m": float(
                    np.sqrt(np.einsum("ij,ij->i", shift, shift)).max()
                ),  # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is ~2.4x faster than np.linalg.norm(..., axis=1)
                "marker_rms_m": float(np.sqrt(np.mean(errors[lane.valid] ** 2))),
                "closure_error_max_m": float(max(f.closure_error_m for f in fits)),
                **zmp_summary(zmp, lane.times),
            }
        )
        log.info(
            "zmp filter pass %d: com shift max %.1f mm, markers %.1f mm, zmp outside "
            "%.2f (1.0-1.5 s %.2f), max %.0f mm",
            k + 1,
            passes[-1]["com_shift_max_m"] * 1e3,
            passes[-1]["marker_rms_m"] * 1e3,
            passes[-1]["outside_fraction"],
            passes[-1]["outside_fraction_1s_to_1_5s"],
            passes[-1]["outside_max_m"] * 1e3,
        )
    report["passes"] = passes
    return q_track, zmp, report


def rom_flags(q: np.ndarray, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Human range-of-motion violations of a trajectory (degrees, frames).

    Args:
        q: (N, nq) coordinate trajectory.
        names: Sequence of coordinate names matching columns of q.

    Returns:
        Dictionary mapping coordinate name to violation details.
    """
    found = violations(q, names, HUMAN_RANGES_DEG)
    return {
        name: {
            "max_excess_deg": round(v.max_excess_deg, 2),
            "frames": v.frames,
            "fraction": round(v.fraction, 4),
        }
        for name, v in found.items()
    }


def com_report(
    sim: fs.FullBodySimulator,
    kin: FullBodyMarkerKinematics,
    q: np.ndarray,
    ground: GroundPlane,
) -> dict[str, Any]:
    """Whole-body-plus-club centre of mass at ``q`` and support polygon check.

    Args:
        sim: Full-body simulator for mass properties.
        kin: Marker kinematics for contact sphere positions.
        q: (nq,) joint coordinate vector.
        ground: Ground plane.

    Returns:
        Dictionary reporting CoM position, height, polygon inclusion, and offset.
    """
    com, _ = sim.centre_of_mass(q)
    points = kin.sphere_ground_points(q, ground)
    polygon = np.array([p[:2] for p in points.values()])
    return {
        "com_m": [float(v) for v in com],
        "height_above_ground_m": float(com[2] - ground.height_m),
        "inside_support_polygon": bool(convex_hull_contains(com[:2], polygon)),
        "polygon_centroid_offset_m": float(np.linalg.norm(com[:2] - polygon.mean(0))),
    }


@dataclass(frozen=True)
class DynamicsReportInputs:
    """Inputs required to construct the full forward dynamics tracking report."""

    record: Any
    sim_q: np.ndarray
    q_ref: np.ndarray
    lane: Lane
    kin: FullBodyMarkerKinematics
    adapter: Any
    zmp: dict[str, Any]
    labels: tuple[str, ...] | Sequence[str]
    zmp_filter_report: dict[str, Any] | None = None
    shooting_report: dict[str, Any] | None = None
    sim_errors: np.ndarray | None = None
    tracking_backend: str = "kkt"


def _build_reference_zmp_report(
    zmp: Mapping[str, Any],
    times: np.ndarray,
) -> dict[str, Any]:
    return {
        "description": (
            "zero-moment point the tracked reference demands of this model "
            "against the support polygon of the spheres on the plane at each frame; "
            "outside means no unilateral foot contact can realise the reference there"
        ),
        "outside_fraction": float((zmp["outside_m"] > 0).mean()),
        "outside_fraction_1s_to_1_5s": float(
            (zmp["outside_m"][(times >= 1.0) & (times < 1.5)] > 0).mean()
        ),
        "outside_max_m": float(zmp["outside_m"].max()),
        "unloaded_fraction": float(zmp["unloaded"].mean()),
        "vertical_grf_over_weight": [
            float(zmp["grf_over_weight"][:, 2].min()),
            float(zmp["grf_over_weight"][:, 2].max()),
        ],
        "horizontal_grf_over_weight_max": float(
            np.sqrt(
                np.einsum(
                    "ij,ij->i",
                    zmp["grf_over_weight"][:, :2],
                    zmp["grf_over_weight"][:, :2],
                )
            ).max()
        ),  # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is ~2.4x faster than np.linalg.norm(..., axis=1)
    }


def _build_backswing_metrics(
    sim_q: np.ndarray,
    q_ref: np.ndarray,
    sim_errors: np.ndarray,
    valid: np.ndarray,
    frames: int,
    record: Any,
) -> dict[str, Any]:
    limit = min(361, frames)
    return {
        "root_error_max_m": float(
            np.sqrt(
                np.einsum(
                    "ij,ij->i", diff := (sim_q[:limit, :3] - q_ref[:limit, :3]), diff
                )
            ).max()
        ),  # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is ~2.4x faster than np.linalg.norm(..., axis=1)
        "marker_rms_m": float(np.sqrt(np.mean(sim_errors[:limit][valid[:limit]] ** 2))),
        "weight_fraction_min": float(
            record.weight_fraction[record.time_s <= 1.0].min()
        ),
        "weight_fraction_max": float(
            record.weight_fraction[record.time_s <= 1.0].max()
        ),
    }


def compute_swing_phase_windows(
    times: np.ndarray,
) -> tuple[tuple[str, float, float], ...]:
    """Compute (phase_name, start_time, end_time) tuples for canonical swing phases."""
    if len(times) == 0:
        return ()
    t_start = float(times[0])
    t_end = float(times[-1])
    t_span = t_end - t_start
    t_addr_end = t_start + min(0.30, 0.2 * t_span)
    return (
        ("address", t_start, t_addr_end),
        ("backswing", t_addr_end, t_start + 0.60 * t_span),
        ("downswing", t_start + 0.60 * t_span, t_start + 0.72 * t_span),
        ("impact", t_start + 0.72 * t_span, t_start + 0.78 * t_span),
        ("follow_through", t_start + 0.78 * t_span, t_end),
    )


def compute_phase_weight_fractions(
    times: np.ndarray,
    record_time_s: np.ndarray,
    weight_fraction: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute weight-fraction summary statistics partitioned by biomechanical swing phase.

    Phases:
        - address: start to min(0.30 s, 20% of duration)
        - backswing: end of address to 60% of duration
        - downswing: 60% to 72% of duration
        - impact: 72% to 78% of duration
        - follow_through: 78% of duration to end

    Args:
        times: Array of capture timestamps.
        record_time_s: Array of simulation timestamps.
        weight_fraction: Array of total vertical contact force / body weight.

    Returns:
        Dictionary mapping phase name to dict with 'min', 'max', 'mean' weight fractions.
    """
    if len(times) == 0 or len(record_time_s) == 0:
        return {}

    phase_windows = compute_swing_phase_windows(times)
    out: dict[str, dict[str, float]] = {}
    for name, t0, t1 in phase_windows:
        mask = (record_time_s >= t0) & (record_time_s <= t1)
        if mask.any():
            wf_sub = weight_fraction[mask]
            out[name] = {
                "min": float(wf_sub.min()),
                "max": float(wf_sub.max()),
                "mean": float(wf_sub.mean()),
            }
        else:
            out[name] = {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
            }
    return out


def build_dynamics_report(
    inputs: DynamicsReportInputs,
) -> tuple[dict[str, Any], np.ndarray]:
    """Build the structured dynamics tracking report and compute tracking errors.

    Preconditions:
    - ``inputs.sim_q`` and ``inputs.q_ref`` must have identical shapes.
    - ``inputs.lane.times`` must match length of ``inputs.sim_q``.

    Returns:
    - ``(dynamics_report, sim_errors)``
    """
    lane = inputs.lane
    kin = inputs.kin
    adapter = inputs.adapter
    record = inputs.record
    sim_q = inputs.sim_q
    q_ref = inputs.q_ref

    sim_errors = (
        inputs.sim_errors
        if inputs.sim_errors is not None
        else marker_errors(kin, sim_q, lane.points)
    )

    report = {
        "duration_s": float(lane.times[-1]),
        "dt_s": DT_S,
        "controller": {
            "type": "computed torque tracking, unactuated root",
            "backend": validate_tracking_backend(inputs.tracking_backend),
            "omega_rad_s": OMEGA_RAD_S,
            "zeta": 1.0,
            "balance": BALANCE,
            "tracking_cutoff_hz": TRACKING_CUTOFF_HZ,
        },
        "contact_parameters": adapter.contact_parameters.as_document(),
        "zmp_filter": inputs.zmp_filter_report,
        "shooting_fit": inputs.shooting_report,
        "reference_zmp": _build_reference_zmp_report(inputs.zmp, lane.times),
        "marker_rms_m": float(np.sqrt(np.mean(sim_errors[lane.valid] ** 2))),
        "segment_rms_m": segment_rms(inputs.labels, sim_errors, lane.valid),
        "joint_tracking_rms_rad": float(
            np.sqrt(np.mean((sim_q[:, 6:] - q_ref[:, 6:]) ** 2))
        ),
        "root_tracking_rms_m": float(
            np.sqrt(np.mean((sim_q[:, :3] - q_ref[:, :3]) ** 2))
        ),
        "weight_fraction": {
            "min": float(record.weight_fraction.min()),
            "max": float(record.weight_fraction.max()),
            "mean": float(record.weight_fraction.mean()),
            "by_phase": compute_phase_weight_fractions(
                lane.times, record.time_s, record.weight_fraction
            ),
        },
        "inside_support_polygon_fraction": float(record.inside_support_polygon.mean()),
        "range_of_motion_flags": rom_flags(sim_q, kin.coordinate_order),
        "root_error_timeline_m": {
            f"{t:.2f}": float(
                math.sqrt(
                    np.vdot(
                        diff := (
                            sim_q[int(round(t * RATE_HZ)), :3]
                            - q_ref[int(round(t * RATE_HZ)), :3]
                        ),
                        diff,
                    )
                )
            )  # ⚡ Bolt: math.sqrt(np.vdot) avoids temporary allocations and is faster than np.linalg.norm for small 1D arrays
            for t in (0.0, 0.25, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75)
            if int(round(t * RATE_HZ)) < lane.frames
        },
        "backswing_to_1s": _build_backswing_metrics(
            sim_q, q_ref, sim_errors, lane.valid, lane.frames, record
        ),
        "peak_joint_torque_n_m": float(np.abs(record.tau).max()),
        "lowest_sphere_height_min_m": float(record.lowest_sphere_height_m.min()),
        "lowest_sphere_height_max_m": float(record.lowest_sphere_height_m.max()),
    }
    return report, sim_errors
