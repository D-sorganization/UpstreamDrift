"""Forward dynamics tracking, shooting fit, and ZMP filtering for full-body pipeline."""

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from src.engines.physics_engines.mujoco.python import full_body_simulation as fs
from src.engines.physics_engines.mujoco.python.full_body_markers import (
    FullBodyMarkerKinematics,
)
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
) -> tuple[fs.SimulationRecord, np.ndarray]:
    """Track ``q_track`` with computed torque from preloaded feet.

    Args:
        sim: Full-body forward simulator.
        lane: Coordination lane providing times.
        q_track: (N, nq) reference trajectory to track.

    Returns:
        (record, sim_q): simulation record and state resampled on the capture times.
    """
    from src.engines.physics_engines.mujoco.python import full_body_simulation as fs

    q0 = fs.preload_feet(sim, q_track[0])
    v0 = np.gradient(q_track, lane.times, axis=0)[0]
    controller = fs.tracking_controller(
        sim, lane.times, q_track, omega_rad_s=OMEGA_RAD_S, zeta=1.0, balance=BALANCE
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
        record, sim_q = replay(sim, lane, q_track)
        errors = marker_errors(kin, sim_q, lane.points)
        rms = float(np.sqrt(np.mean(errors[lane.valid] ** 2)))
        root_err = np.linalg.norm(sim_q[:, :3] - q_ref[:, :3], axis=1)
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
    from src.engines.physics_engines.mujoco.python import full_body_simulation as fs

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
                "com_shift_max_m": float(np.linalg.norm(shift, axis=1).max()),
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
