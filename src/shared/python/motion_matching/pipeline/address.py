from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.engines.physics_engines.mujoco.python.full_body_ik import (
        FullBodyMarkerKinematics,
    )

from src.shared.python.motion_matching.anthropometric_candidate import (
    anthropometric_candidate,
)
from src.shared.python.motion_matching.full_body_spec import (
    canonical_sha256,
    validate_full_body_spec,
)
from src.shared.python.motion_matching.hip_calibration import (
    apply_hip_calibration,
    functional_hip_calibration,
    hip_rotation_zero,
)
from src.shared.python.motion_matching.marker_calibration import (
    calibrate_marker_offsets,
    static_marker_offsets,
)
from src.shared.python.motion_matching import posture_metrics as post
from src.shared.python.motion_matching.closure_fit import (
    closure_residual,
    fit_closure_placement,
)
from src.shared.python.motion_matching.pipeline.constants import (
    ADDRESS_BALANCE_WEIGHT,
    ADDRESS_ELBOW_BOUNDS_DEG,
    ADDRESS_RESTART_SPREAD_RAD,
    ADDRESS_RESTARTS,
    ADDRESS_SEEDS_DEG,
    CALIBRATION_ITERATIONS,
    CALIBRATION_PRIOR_FRAMES,
    ELBOW_PIT_WEIGHT,
    ELBOW_PIT_WEIGHTS_NEUTRAL,
    FORWARD_AXIS,
    LEG_SEEDS,
    NEUTRAL_BOUNDS_DEG,
    NEUTRAL_LOCKS,
    PRIOR,
    RIGHT_AXIS,
    SCALE_GRID,
    STATIC_FRAMES,
    TRAIL_WRIST_ADDRESS_DEG,
    UP_AXIS,
)
from src.shared.python.motion_matching.pipeline.lane import (
    add_toe_spheres,
    wrist_bounds,
)
from src.shared.python.motion_matching.segment_scaling import scale_segments
from src.shared.python.motion_matching.tour_capture_contract import (
    MARKER_SEGMENTS,
    TourCapture,
)

if TYPE_CHECKING:
    from src.shared.python.motion_matching.pipeline.lane import Lane


def scaled_offsets(
    offsets: Mapping[str, tuple[str, Sequence[float]]], femur: float, tibia: float
) -> dict[str, tuple[str, tuple[float, float, float]]]:
    """Leg marker offsets carried onto scaled femur and tibia bodies.

    Precondition: ``femur`` and ``tibia`` must be strictly positive.
    Postcondition: returned offsets preserve body names with scaled coordinate offsets.
    """
    if femur <= 0 or tibia <= 0:
        raise ValueError("femur and tibia must have positive scale factors")
    out: dict[str, tuple[str, tuple[float, float, float]]] = {}
    for label, (body, offset) in offsets.items():
        if body.startswith("femur"):
            factor = femur
        elif body.startswith("tibia"):
            factor = tibia
        else:
            factor = 1.0
        scaled = factor * np.asarray(offset, dtype=float)
        out[label] = (body, (float(scaled[0]), float(scaled[1]), float(scaled[2])))
    return out


def posture_summary(kin: Any, q: np.ndarray) -> dict[str, Any]:
    """Spine bend and clavicle-link angles of the model at one pose."""
    if hasattr(kin, "posture_summary"):
        return kin.posture_summary(q)
    adapter = kin.adapter
    m, d = adapter.model, adapter.data
    kin.marker_positions(q)

    def site(frame: str) -> np.ndarray:
        return d.site_xpos[m.site(adapter.metadata["frame_sites"][frame]).id].copy()

    hips = (
        d.xanchor[m.joint("hip_flexion_r").id] + d.xanchor[m.joint("hip_flexion_l").id]
    ) / 2
    spine, hub = site("Spine"), site("Hub")
    bend = post.spine_bend(spine - hips, hub - spine, UP_AXIS, FORWARD_AXIS, RIGHT_AXIS)
    links = {}
    for side in ("L", "R"):
        v = site(f"{side}S") - hub
        links[side] = float(np.degrees(np.arcsin(-v[2] / np.linalg.norm(v))))
    return {
        "spine_bend_deg": bend.__dict__,
        "clavicle_link_below_horizontal_deg": links,
        "hips_to_shoulder_centre_m": float(
            np.linalg.norm((site("LS") + site("RS")) / 2 - hips)
        ),
    }


def best_address(
    lane: Lane,
    kin: FullBodyMarkerKinematics,
    base: np.ndarray,
    *,
    neutral: bool = False,
    pit_weight: float = ELBOW_PIT_WEIGHT,
) -> Any:
    """Best address fit over the leg seeds; ``neutral`` locks the scapulae
    and bounds the spine to a straight-torso address.
    """
    bounds = dict(lane.bounds)
    locked = None
    axis_targets = None
    if neutral:
        locked = dict(NEUTRAL_LOCKS)
        bounds |= {
            name: (float(np.radians(lo)), float(np.radians(hi)))
            for name, (lo, hi) in NEUTRAL_BOUNDS_DEG.items()
        }
    balance = ADDRESS_BALANCE_WEIGHT if lane.anthropometric else 0.0
    if lane.anthropometric:
        bounds |= {
            name: (float(np.radians(lo)), float(np.radians(hi)))
            for name, (lo, hi) in ADDRESS_ELBOW_BOUNDS_DEG.items()
        }
        axis_targets = lane.pit_targets_for(list(range(STATIC_FRAMES)), pit_weight)

    best = None
    rng = np.random.default_rng(0)
    for seed in ADDRESS_SEEDS_DEG:
        start = base.copy()
        for joint, value in seed.items():
            for side in ("r", "l"):
                coord = f"{joint}_{side}"
                if coord in kin.coordinate_order:
                    start[kin.coordinate_order.index(coord)] = float(np.radians(value))
        starts = [start] + [
            np.concatenate(
                [
                    start[:6],
                    start[6:]
                    + rng.uniform(
                        -ADDRESS_RESTART_SPREAD_RAD,
                        ADDRESS_RESTART_SPREAD_RAD,
                        len(start) - 6,
                    ),
                ]
            )
            for _ in range(ADDRESS_RESTARTS)
        ]
        for q0 in starts:
            fit = kin.solve_pose(
                lane.points[0],
                lane.valid[0],
                q0,
                ground=lane.ground,
                prior_weight=PRIOR,
                iterations=100,
                flat_feet=lane.stance[0],
                bounds=bounds,
                locked=locked,
                marker_weights=lane.marker_weights,
                prior_weights=lane.prior_weights,
                axis_targets=axis_targets,
                balance_weight=balance,
            )
            if best is None or fit.marker_rms_m < best.marker_rms_m:
                best = fit
    if best is None:
        raise RuntimeError("No viable address fit found across multi-start seeds")
    return best


def static_offsets(
    lane: Lane, kin: FullBodyMarkerKinematics, q: np.ndarray, seeds: dict
) -> dict:
    """Static-trial placement of every seed marker at pose ``q`` over the
    first ``STATIC_FRAMES`` frames (bodies as in ``seeds``)."""
    frames = list(range(STATIC_FRAMES))
    seen = {
        label for label in seeds if lane.valid[frames, lane.labels.index(label)].any()
    }
    cols = [lane.labels.index(m) for m in seen]
    capture = TourCapture(
        time_s=lane.times[frames],
        labels=tuple(seen),
        points_m=lane.points[np.ix_(frames, cols)],
        valid=lane.valid[np.ix_(frames, cols)],
    )
    bodies = {label: seeds[label][0] for label in seen}
    poses = kin.body_poses(q, sorted(set(bodies.values())))
    placed = static_marker_offsets(capture, bodies, [poses] * len(frames))
    return {label: placed.get(label, seed) for label, seed in seeds.items()}


def static_trial(
    lane: Lane, spec_bytes: bytes, seeds: dict, q_seed: np.ndarray
) -> tuple[dict, Any, FullBodyMarkerKinematics]:
    """Alternating static trial: neutral address fit, placement of every
    marker at that pose, and again with the placed markers."""
    placed = dict(seeds)
    q = q_seed
    neutral = None
    kin = None
    for weight in ELBOW_PIT_WEIGHTS_NEUTRAL:
        adapter, kin = lane.kinematics(spec_bytes, placed)
        neutral = best_address(lane, kin, q, neutral=True, pit_weight=weight)
        placed = static_offsets(lane, kin, neutral.q, placed)
        q = neutral.q
    adapter, kin = lane.kinematics(spec_bytes, placed)
    if neutral is None or kin is None:
        raise RuntimeError("Static trial failed to converge")
    return placed, neutral, kin


def fit_closure_from_address(
    lane: Lane,
    kin: FullBodyMarkerKinematics,
    document: dict[str, Any],
    q_start: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the dual-grip weld at the address: keep both hands on the grip
    point but release the weld's orientation, hold the trail wrist at
    anatomical values, bound the lead wrist to human ranges, fit the markers,
    then rewrite the closure so the weld holds exactly there.
    """
    bounds = dict(lane.bounds) | wrist_bounds()
    locked = {k: float(np.radians(v)) for k, v in TRAIL_WRIST_ADDRESS_DEG.items()}
    fit = kin.solve_pose(
        lane.points[0],
        lane.valid[0],
        q_start,
        ground=lane.ground,
        prior_weight=PRIOR,
        iterations=150,
        flat_feet=lane.stance[0],
        bounds=bounds,
        locked=locked,
        marker_weights=lane.marker_weights,
        prior_weights=lane.prior_weights,
        axis_targets=lane.pit_targets_for(list(range(STATIC_FRAMES)), ELBOW_PIT_WEIGHT),
        closure_rotation_weight=0.0,
    )
    kin._set(fit.q)
    # LoD: Use delegating property closure_sites
    site_a = kin.closure_sites[0]
    hand_world = (
        kin.data.site_xmat[site_a].reshape(3, 3).copy(),
        kin.data.site_xpos[site_a].copy(),
    )
    club_body = document["closure"]["body_b"]
    club_world = kin.body_poses(fit.q, [club_body])[club_body]
    fitted = fit_closure_placement(document, hand_world, club_world)
    residual = closure_residual(
        hand_world, club_world, fitted["closure"]["placement_b"]
    )
    return fitted, {
        **fitted["closure_fit"],
        "open_chain_marker_rms_m": fit.marker_rms_m,
        "trail_wrist_locked_deg": dict(TRAIL_WRIST_ADDRESS_DEG),
        "residual_at_fit_m_rad": list(residual),
    }


def elbow_pit_targets(
    lane: Lane,
    kin: FullBodyMarkerKinematics,
    q: np.ndarray,
    weight: float,
) -> dict[str, tuple[Sequence[float], Sequence[float], float]]:
    """Axis targets pulling upper arms toward address marker pit directions."""
    return lane.pit_targets_for(list(range(STATIC_FRAMES)), weight)


def calibrate_legs(
    lane: Lane,
    spec_bytes: bytes,
    upper: Mapping[str, tuple[str, Sequence[float]]],
    seeds: Mapping[str, tuple[str, Sequence[float]]],
    q_start: np.ndarray,
) -> tuple[dict[str, tuple[str, tuple[float, float, float]]], Any]:
    """Alternating marker calibration of ``seeds`` (others fixed) with stance pins."""
    frames = lane.calibration_frames
    calibrated = tuple(seeds)
    cols = [lane.labels.index(m) for m in calibrated]
    leg_capture = TourCapture(
        time_s=lane.times[frames],
        labels=calibrated,
        points_m=lane.points[np.ix_(frames, cols)],
        valid=lane.valid[np.ix_(frames, cols)],
    )
    leg_bodies = {label: body for label, (body, _) in seeds.items()}
    adapter, kin0 = lane.kinematics(spec_bytes, {**upper, **seeds})
    state = {"kin": kin0}

    def pose_fn(q: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return state["kin"].body_poses(q, sorted(set(leg_bodies.values())))

    def ik_fn(
        offsets: dict[str, tuple[str, tuple[float, float, float]]],
        cap: TourCapture,
    ) -> np.ndarray:
        merged = {**upper, **offsets}
        attachments = {
            label: (
                merged[label][0],
                (
                    float(merged[label][1][0]),
                    float(merged[label][1][1]),
                    float(merged[label][1][2]),
                ),
            )
            for label in lane.labels
        }
        if lane.plant is not None:
            state["kin"] = lane.plant.create_ik(attachments)
        else:
            from src.engines.physics_engines.mujoco.python.full_body_ik import (
                FullBodyMarkerKinematics,
            )

            state["kin"] = FullBodyMarkerKinematics(adapter, attachments)
        return lane.trajectory(state["kin"], q_start, frames=frames)[0]

    result = calibrate_marker_offsets(
        leg_capture,
        leg_bodies,
        pose_fn,
        ik_fn,
        initial_q=q_start,
        iterations=CALIBRATION_ITERATIONS,
        prior_offsets={label: offset for label, (_, offset) in seeds.items()},
        prior_weight=CALIBRATION_PRIOR_FRAMES,
    )
    offsets = {
        label: (
            body,
            (float(offset[0]), float(offset[1]), float(offset[2])),
        )
        for label, (body, offset) in result.offsets.items()
    }
    return offsets, result


def search_segment_scales(
    lane: Lane,
    hip_spec: Mapping[str, Any],
    fixed: Mapping[str, tuple[str, Sequence[float]]],
    offsets: Mapping[str, tuple[str, Sequence[float]]],
    address_q: np.ndarray,
    grid: Sequence[float] = SCALE_GRID,
    log: logging.Logger | None = None,
) -> tuple[dict[str, Any], float, float, list[dict[str, Any]]]:
    """Grid search over femur and tibia scales minimizing pinned stance RMS.

    Preconditions:
    - ``grid`` must not be empty and must contain strictly positive scale factors.
    - ``address_q`` must be a 1D array of generalized coordinates.

    Postconditions:
    - Returns best scaled specification, femur scale, tibia scale, and table.
    """
    if not grid:
        raise ValueError("grid must not be empty")
    for s in grid:
        if s <= 0:
            raise ValueError(f"grid scale values must be positive, got {s}")
    if address_q.ndim != 1:
        raise ValueError(f"address_q must be a 1D array, got shape {address_q.shape}")

    scale_table: list[dict[str, Any]] = []
    best = (float("inf"), 1.0, 1.0, dict(hip_spec))
    for femur in grid:
        for tibia in grid:
            scales = {f"femur_{s}": femur for s in "rl"} | {
                f"tibia_{s}": tibia for s in "rl"
            }
            doc = scale_segments(hip_spec, scales)
            doc_bytes = json.dumps(doc, sort_keys=True).encode()
            rms = lane.pinned_rms(
                doc_bytes, {**fixed, **scaled_offsets(offsets, femur, tibia)}, address_q
            )
            scale_table.append({"femur": femur, "tibia": tibia, "pinned_rms_m": rms})
            if log:
                log.info(
                    "scale femur %.2f tibia %.2f: pinned RMS %.1f mm",
                    femur,
                    tibia,
                    rms * 1e3,
                )
            if rms < best[0]:
                best = (rms, femur, tibia, doc)
    _, femur_scale, tibia_scale, scaled_spec = best
    return scaled_spec, femur_scale, tibia_scale, scale_table


@dataclass(frozen=True)
class HipCalibrationOptions:
    """Options governing functional hip calibration and candidate preparation."""

    skip_hip_calibration: bool = False
    anthropometric: tuple[float, float] | None = None
    recalibrate_upper: bool = False


def prepare_hip_spec(
    lane: Lane,
    base_spec: Mapping[str, Any],
    upper_base: Mapping[str, Any],
    labels: tuple[str, ...],
    upper: Mapping[str, tuple[str, Sequence[float]]],
    alignment_old: Sequence[float] | None = None,
    options: HipCalibrationOptions | None = None,
) -> tuple[
    dict[str, Any],
    str,
    dict[str, Any],
    dict[str, tuple[str, Sequence[float]]],
    dict[str, tuple[str, Sequence[float]]],
]:
    """Execute functional hip calibration and configure candidate spec.

    Preconditions:
    - ``base_spec`` must contain valid marker attachments.
    - ``labels`` must match capture markers.

    Returns:
    - ``(hip_spec, qualification_note, hip_report, fixed, seeds_all)``
    """
    opts = options or HipCalibrationOptions()
    waist_offsets = {
        label: base_spec["marker_attachments"][label]["offset_m"]
        for label in MARKER_SEGMENTS["pelvis"]
    }
    hip_cal = functional_hip_calibration(lane.points, lane.valid, labels, waist_offsets)
    zero_twist = hip_rotation_zero(
        lane.points, lane.valid, labels, waist_offsets, calibration=hip_cal
    )
    if opts.skip_hip_calibration:
        hip_spec = add_toe_spheres(dict(base_spec))
    else:
        if alignment_old is None:
            raise ValueError(
                "alignment_old required when skip_hip_calibration is False"
            )
        hip_spec = add_toe_spheres(
            apply_hip_calibration(
                base_spec, hip_cal, alignment_old, zero_twist_deg=zero_twist
            )
        )
    unqualified = "unqualified" in str(base_spec.get("upper_body_qualification", ""))
    if opts.anthropometric:
        stature_m, mass_kg = opts.anthropometric
        hip_spec = anthropometric_candidate(
            hip_spec, stature_m=stature_m, mass_kg=mass_kg
        )
        qualification_note = (
            f"anthropometric candidate ({stature_m:.3f} m, {mass_kg:.1f} kg), "
            "unqualified: upper-body lengths, masses and inertias changed"
        )
    elif unqualified:
        qualification_note = str(base_spec.get("upper_body_qualification", ""))
    else:
        validate_full_body_spec(hip_spec, upper_base)
        qualification_note = "qualified upper body with functional hips"

    if opts.recalibrate_upper:
        seeds_all = {**upper, **LEG_SEEDS}
        fixed = {}
    else:
        seeds_all = dict(LEG_SEEDS)
        fixed = dict(upper)

    hip_report = {
        "spec_sha256": canonical_sha256(hip_spec),
        "centre_r_hip_frame_m": hip_cal.centre_r,
        "centre_l_hip_frame_m": hip_cal.centre_l,
        "radius_r_m": hip_cal.radius_r_m,
        "radius_l_m": hip_cal.radius_l_m,
        "sphere_sd_r_m": hip_cal.residual_sd_r_m,
        "sphere_sd_l_m": hip_cal.residual_sd_l_m,
        "frames": hip_cal.frames,
        "pelvis_axes_in_hip_frame": hip_cal.pelvis_axes,
        "waist_fit_max_residual_m": hip_cal.waist_fit_max_residual_m,
        "hip_zero_twist_deg": zero_twist.to_dict(),
    }
    return hip_spec, qualification_note, hip_report, fixed, seeds_all


def calibrated_address_summary(
    sim: Any,
    kin: FullBodyMarkerKinematics,
    address: Any,
    lane: Lane,
    labels: tuple[str, ...],
    adapter: Any,
) -> dict[str, Any]:
    """Summary of the calibrated address pose, closure, CoM, and angles.

    Preconditions:
    - ``address.q`` must match the degrees of freedom of ``kin``.
    """
    if len(address.q) != len(kin.coordinate_order):
        raise ValueError(
            f"address.q size {len(address.q)} does not match {len(kin.coordinate_order)}"
        )
    from src.shared.python.motion_matching.pipeline.dynamics import (
        com_report,
        segment_rms,
    )
    from src.shared.python.motion_matching.pipeline.reference import marker_errors

    return {
        "marker_rms_m": address.marker_rms_m,
        "segment_rms_m": segment_rms(
            labels,
            marker_errors(kin, address.q[None, :], lane.points[:1]),
            lane.valid[:1],
        ),
        "closure_error_m": address.closure_error_m,
        "lowest_sphere_height_m": address.lowest_sphere_height_m,
        "support_offset_m": kin.support_offset(address.q, lane.ground),
        "centre_of_mass": com_report(sim, kin, address.q, lane.ground),
        "leg_angles_deg": {
            name: float(np.degrees(address.q[kin.coordinate_order.index(name)]))
            for name in kin.coordinate_order[adapter.upper_body_coordinates :]
        },
        "posture": posture_summary(kin, address.q),
    }


@dataclass(frozen=True)
class AddressStageInputs:
    """Inputs to solve the initial address pose and optional static trial/closure."""

    lane: Lane
    base_spec: Mapping[str, Any]
    hip_spec: Mapping[str, Any]
    hip_bytes: bytes
    upper: Mapping[str, tuple[str, Sequence[float]]]
    fixed: Mapping[str, tuple[str, Sequence[float]]]
    seeds_all: Mapping[str, tuple[str, Sequence[float]]]
    labels: tuple[str, ...]
    hipcal_path: Path | None = None
    static_seeds: bool = False
    fit_closure: bool = False
    log: logging.Logger | None = None


@dataclass(frozen=True)
class AddressStageResult:
    """Result of solving the address pose stage."""

    address: Any
    address_report: dict[str, Any]
    adapter: Any
    kin: FullBodyMarkerKinematics
    fixed: dict[str, tuple[str, Sequence[float]]]
    seeds_all: dict[str, tuple[str, Sequence[float]]]
    hip_spec: dict[str, Any]
    hip_bytes: bytes


def solve_address_stage(inputs: AddressStageInputs) -> AddressStageResult:
    """Solve the address stage: seed offsets, static trial, and closure fitting.

    Preconditions:
    - ``fit_closure`` requires ``static_seeds``.
    """
    if inputs.fit_closure and not inputs.static_seeds:
        raise ValueError("--fit-closure needs --static-seeds")

    from src.shared.python.motion_matching.pipeline.dynamics import segment_rms
    from src.shared.python.motion_matching.pipeline.lane import document_seed
    from src.shared.python.motion_matching.pipeline.reference import marker_errors

    lane = inputs.lane
    base_spec = inputs.base_spec
    hip_spec = dict(inputs.hip_spec)
    hip_bytes = inputs.hip_bytes
    fixed = dict(inputs.fixed)
    seeds_all = dict(inputs.seeds_all)
    labels = inputs.labels
    log = inputs.log

    adapter, kin = lane.kinematics(hip_bytes, {**inputs.upper, **LEG_SEEDS})
    q_seed = document_seed(dict(base_spec), kin)
    address = lane.best_address(kin, q_seed)
    address_report: dict[str, Any] = {
        "seed_offsets": {
            "marker_rms_m": address.marker_rms_m,
            "segment_rms_m": segment_rms(
                labels,
                marker_errors(kin, address.q[None, :], lane.points[:1]),
                lane.valid[:1],
            ),
        },
        "stance_spheres": lane.stance[0],
    }

    if inputs.static_seeds:
        placed, neutral, kin = lane.static_trial(
            hip_bytes, {**fixed, **seeds_all}, q_seed
        )
        adapter = kin.adapter
        fixed = {label: placed[label] for label in fixed}
        seeds_all = {label: placed[label] for label in seeds_all}
        address = lane.best_address(kin, neutral.q)
        address_report["static_trial"] = {
            "frames": STATIC_FRAMES,
            "neutral_fit_rms_m": neutral.marker_rms_m,
            "neutral_posture": posture_summary(kin, neutral.q),
            "marker_rms_m": address.marker_rms_m,
            "posture": posture_summary(kin, address.q),
        }
        if log:
            log.info(
                "static trial: neutral fit %.1f mm, address with static seeds %.1f mm",
                neutral.marker_rms_m * 1e3,
                address.marker_rms_m * 1e3,
            )

    if inputs.fit_closure:
        hip_spec, closure_report = fit_closure_from_address(
            lane, kin, hip_spec, address.q
        )
        if inputs.hipcal_path:
            inputs.hipcal_path.write_text(
                json.dumps(hip_spec, indent=2, sort_keys=True) + "\n"
            )
            hip_bytes = inputs.hipcal_path.read_bytes()
        else:
            hip_bytes = json.dumps(hip_spec, sort_keys=True).encode()
        lane.bounds |= wrist_bounds()
        adapter, kin = lane.kinematics(hip_bytes, {**fixed, **seeds_all})
        address = lane.best_address(kin, address.q)
        closure_report["address_rms_after_m"] = address.marker_rms_m
        address_report["closure_fit"] = closure_report
        if log:
            log.info(
                "closure fit: weld turned %.1f deg, moved %.1f mm; address %.1f mm with "
                "wrists bounded",
                closure_report["rotation_change_deg"],
                closure_report["translation_change_m"] * 1e3,
                address.marker_rms_m * 1e3,
            )

    return AddressStageResult(
        address=address,
        address_report=address_report,
        adapter=adapter,
        kin=kin,
        fixed=fixed,
        seeds_all=seeds_all,
        hip_spec=hip_spec,
        hip_bytes=hip_bytes,
    )
