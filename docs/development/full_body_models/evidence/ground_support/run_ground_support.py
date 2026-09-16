"""Ground-supported full-body pipeline on the MuJoCo full-body model (GS-0 to GS-5).

Stages, each receipted in ``receipt.json`` beside this script:

0. Functional hip calibration: both hip joints of the v2 specification are
   relocated to the functional centres estimated from the knee markers in the
   pelvis frame (``full_body_spec_hipcal.json``, its own hash).
1. Ground calibration from the lowest toe markers of the tour capture mapped
   into the native world (``(x, y, z) -> (x, -z, y)``), and per-frame stance
   detection (which contact spheres are on the ground) from marker heights
   relative to address.
2. Address pose: 41-coordinate IK on frame 0 from several leg seeds with the
   25 qualified upper-body markers, the 8 lower-limb markers (anatomical seed
   offsets), the grip closure and the stance spheres pinned to the plane.
3. Lower-limb marker calibration with the shared alternating algorithm on a
   decimated frame set with the stance pins; then a grid search over femur
   and tibia length scales (``segment_scaling``) judged by the same pinned IK;
   recalibration on the scaled document (``full_body_spec_hipcal_scaled.json``).
4. Full 654-frame IK, zero-phase low-pass, and a consistency re-solve that
   keeps the smoothed reference on the ground and the grip closed.
5. Forward dynamics with the shared contact law: computed-torque tracking of
   the reference with an unactuated root, the feet carrying the golfer.
   Marker RMS of the simulated motion, ground reaction, centre of pressure
   and joint torques are reported; playback GIFs are rendered.

Every number here is a milestone of a stated candidate, not acceptance.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from src.engines.physics_engines.mujoco.python import (
    full_body_simulation as fs,
)  # noqa: E402
from src.shared.python.motion_matching.anthropometric_candidate import (  # noqa: E402
    anthropometric_candidate,
)
from src.shared.python.motion_matching.full_body_spec import (  # noqa: E402
    canonical_sha256,
    validate_full_body_spec,
)
from src.shared.python.motion_matching.hip_calibration import (  # noqa: E402
    apply_hip_calibration,
    functional_hip_calibration,
    hip_rotation_zero,
)
from src.shared.python.motion_matching.segment_scaling import (
    scale_segments,
)  # noqa: E402
from src.shared.python.motion_matching.tour_capture_contract import (  # noqa: E402
    MARKER_SEGMENTS,
)
from src.shared.python.motion_matching.pipeline import (  # noqa: E402
    GroundSupportReceiptInputs,
    Lane,
    add_toe_spheres,
    best_address,
    build_ground_support_receipt,
    com_report,
    configure_lane,
    consistency_resolve,
    document_bounds,
    document_seed,
    fit_closure_from_address,
    fitted_grip,
    full_capture_ik,
    marker_errors,
    posture_summary,
    render_playback,
    replay,
    rom_flags,
    scaled_offsets,
    segment_rms,
    shooting_fit,
    smooth_reference,
    stance_spheres,
    static_offsets,
    static_trial,
    wrist_bounds,
    zmp_filter,
    zmp_summary,
)
from src.shared.python.motion_matching.pipeline.constants import (  # noqa: E402
    ADDRESS_BALANCE_WEIGHT,
    ADDRESS_ELBOW_BOUNDS_DEG,
    ADDRESS_RESTART_SPREAD_RAD,
    ADDRESS_RESTARTS,
    ADDRESS_SEEDS_DEG,
    BALANCE,
    BOUND_WIDENING,
    CALIBRATION_ITERATIONS,
    CALIBRATION_PRIOR_FRAMES,
    CALIBRATION_STRIDE,
    CONSISTENCY_PRIOR,
    CONTACT_STIFFNESS_N_M,
    DT_S,
    ELBOW_PIT_MARKERS,
    ELBOW_PIT_WEIGHT,
    ELBOW_PIT_WEIGHT_SWING,
    ELBOW_PIT_WEIGHTS_NEUTRAL,
    FORWARD_AXIS,
    HEAD_MARKER_WEIGHT,
    IK_UNBOUNDED,
    LEG_LABELS,
    LEG_SEEDS,
    LOWER_LIMB_RANGES_DEG,
    NEUTRAL_BOUNDS_DEG,
    NEUTRAL_LOCKS,
    OMEGA_RAD_S,
    PLAYBACK_STRIDE,
    PRIOR,
    RATE_HZ,
    REFERENCE_CUTOFF_HZ,
    RIGHT_AXIS,
    SCALE_GRID,
    SHOOTING_LOCKED,
    SHOOTING_RELAXATION,
    SHOULDER_GIMBALS,
    SPIN_COORDINATES,
    SPIN_PRIOR,
    STANCE_TOLERANCE_M,
    STATIC_FRAMES,
    TOE_SPHERES,
    TOE_STANDOFF_M,
    TRACKING_CUTOFF_HZ,
    TRAIL_WRIST_ADDRESS_DEG,
    TRAJECTORY_RESTART_MARGIN_M,
    TRAJECTORY_RESTART_THRESHOLD_M,
    TRAJECTORY_RESTARTS,
    UP_AXIS,
    WRIST_COORDINATES,
    ZMP_COM_WEIGHT,
    ZMP_FILTER_ITERATIONS,
    ZMP_MARGIN_M,
)

# Re-exported public constants and path defaults
HERE = Path(__file__).resolve().parent
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v2.json"
BUILD_RECEIPT = ROOT / "docs/development/full_body_models/build_receipt_v2.json"
UPPER_SPEC = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
CANDIDATE = (
    ROOT
    / "docs/development/full_body_models/evidence/native_candidates/returned81_candidate.json"
)
CAPTURES = {
    "driver": ROOT / "data/C3D_TA_Driver.c3d",
    "iron": ROOT / "data/C3D_TA_Iron.c3d",
}
C3D = CAPTURES["driver"]
OUT = HERE

__all__ = [
    "ADDRESS_BALANCE_WEIGHT",
    "ADDRESS_ELBOW_BOUNDS_DEG",
    "ADDRESS_RESTART_SPREAD_RAD",
    "ADDRESS_RESTARTS",
    "ADDRESS_SEEDS_DEG",
    "BALANCE",
    "BOUND_WIDENING",
    "BUILD_RECEIPT",
    "C3D",
    "CALIBRATION_ITERATIONS",
    "CALIBRATION_PRIOR_FRAMES",
    "CALIBRATION_STRIDE",
    "CANDIDATE",
    "CAPTURES",
    "CONSISTENCY_PRIOR",
    "CONTACT_STIFFNESS_N_M",
    "DT_S",
    "ELBOW_PIT_MARKERS",
    "ELBOW_PIT_WEIGHT",
    "ELBOW_PIT_WEIGHTS_NEUTRAL",
    "ELBOW_PIT_WEIGHT_SWING",
    "FORWARD_AXIS",
    "HEAD_MARKER_WEIGHT",
    "HERE",
    "IK_UNBOUNDED",
    "LEG_LABELS",
    "LEG_SEEDS",
    "LOWER_LIMB_RANGES_DEG",
    "Lane",
    "NEUTRAL_BOUNDS_DEG",
    "NEUTRAL_LOCKS",
    "OMEGA_RAD_S",
    "OUT",
    "PLAYBACK_STRIDE",
    "PRIOR",
    "RATE_HZ",
    "REFERENCE_CUTOFF_HZ",
    "RIGHT_AXIS",
    "SCALE_GRID",
    "SHOOTING_LOCKED",
    "SHOOTING_RELAXATION",
    "SHOULDER_GIMBALS",
    "SPEC",
    "SPIN_COORDINATES",
    "SPIN_PRIOR",
    "STANCE_TOLERANCE_M",
    "STATIC_FRAMES",
    "TOE_SPHERES",
    "TOE_STANDOFF_M",
    "TRACKING_CUTOFF_HZ",
    "TRAIL_WRIST_ADDRESS_DEG",
    "TRAJECTORY_RESTARTS",
    "TRAJECTORY_RESTART_MARGIN_M",
    "TRAJECTORY_RESTART_THRESHOLD_M",
    "UPPER_SPEC",
    "UP_AXIS",
    "WRIST_COORDINATES",
    "ZMP_COM_WEIGHT",
    "ZMP_FILTER_ITERATIONS",
    "ZMP_MARGIN_M",
    "add_toe_spheres",
    "best_address",
    "com_report",
    "configure_lane",
    "consistency_resolve",
    "document_bounds",
    "document_seed",
    "fit_closure_from_address",
    "fitted_grip",
    "full_capture_ik",
    "marker_errors",
    "posture_summary",
    "render_playback",
    "replay",
    "rom_flags",
    "scaled_offsets",
    "segment_rms",
    "shooting_fit",
    "smooth_reference",
    "stance_spheres",
    "static_offsets",
    "static_trial",
    "wrist_bounds",
    "zmp_filter",
    "zmp_summary",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument(
        "--spec",
        type=Path,
        default=SPEC,
        help="full-body document to start from (default: the qualified v2)",
    )
    parser.add_argument(
        "--skip-hip-calibration",
        action="store_true",
        help="keep the document's hip joints (anthropometric geometry places them)",
    )
    parser.add_argument(
        "--anthropometric",
        nargs=2,
        type=float,
        metavar=("STATURE_M", "MASS_KG"),
        help="build the de Leva candidate (unqualified) before calibration",
    )
    parser.add_argument(
        "--recalibrate-upper",
        action="store_true",
        help="calibrate the 25 upper-body offsets too (qualified offsets as prior)",
    )
    parser.add_argument(
        "--capture",
        choices=sorted(CAPTURES),
        default="driver",
        help="which canonical tour-average capture to match",
    )
    parser.add_argument(
        "--fit-closure",
        action="store_true",
        help="fit the two-hand closure weld from the address with anatomical "
        "wrists and bound the wrists to human ranges (needs --static-seeds)",
    )
    parser.add_argument(
        "--bound-wrists",
        action="store_true",
        help="impose the human wrist and forearm ranges in the IK even for a "
        "document without a fitted grip rotation",
    )
    parser.add_argument(
        "--free-wrists",
        action="store_true",
        help="leave the wrists and forearms unbounded (flagged only) even for a "
        "document with a fitted grip rotation, e.g. to feed fit_grip_rotation.py",
    )
    parser.add_argument(
        "--shooting-fit",
        type=int,
        default=0,
        metavar="N",
        help="contact-aware shooting fit of the tracked reference (FB-5): N "
        "replay/re-solve iterations with the replayed pelvis pinned; the best "
        "replay is kept",
    )
    parser.add_argument(
        "--shooting-gain",
        type=float,
        default=SHOOTING_RELAXATION,
        help="iterative-learning gain on the pelvis command per shooting pass",
    )
    parser.add_argument(
        "--zmp-filter",
        action="store_true",
        help="dynamics-filter the tracked reference (MM-7b): shift its "
        "centre-of-mass path so the zero-moment point stays inside the "
        "support polygon, re-solving the IK with centre-of-mass rows",
    )
    parser.add_argument(
        "--static-seeds",
        action="store_true",
        help="place every marker from a neutral-spine static trial (upper-body "
        "placements stay fixed unless --recalibrate-upper)",
    )
    args = parser.parse_args()
    global OUT, C3D
    OUT = args.out
    C3D = CAPTURES[args.capture]
    OUT.mkdir(parents=True, exist_ok=True)
    hipcal_path = OUT / "full_body_spec_hipcal.json"
    scaled_path = OUT / "full_body_spec_hipcal_scaled.json"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("ground_support")
    t_start = time.perf_counter()

    base_spec = json.loads(args.spec.read_text())
    unqualified = "unqualified" in str(base_spec.get("upper_body_qualification", ""))
    upper_base = json.loads(UPPER_SPEC.read_text())
    candidate_bytes = CANDIDATE.read_bytes()
    upper = {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in base_spec["marker_attachments"].items()
        if a["offset_m"] is not None
    }
    labels = tuple({**upper, **LEG_SEEDS})
    lane = Lane(labels, C3D)
    configure_lane(lane, base_spec)
    if args.bound_wrists and args.free_wrists:
        raise ValueError("--bound-wrists and --free-wrists exclude each other")
    if args.bound_wrists or (fitted_grip(base_spec) and not args.free_wrists):
        lane.bounds |= wrist_bounds()
        log.info("wrists and forearms bounded to the human ranges in the IK")

    # 0. functional hip calibration
    waist_offsets = {
        label: base_spec["marker_attachments"][label]["offset_m"]
        for label in MARKER_SEGMENTS["pelvis"]
    }
    hip_cal = functional_hip_calibration(lane.points, lane.valid, labels, waist_offsets)
    zero_twist = hip_rotation_zero(
        lane.points, lane.valid, labels, waist_offsets, calibration=hip_cal
    )
    if args.skip_hip_calibration:
        hip_spec = add_toe_spheres(base_spec)
    else:
        alignment_old = json.loads(BUILD_RECEIPT.read_text())["pelvis_alignment"][
            "hip_from_opensim_pelvis"
        ]
        hip_spec = add_toe_spheres(
            apply_hip_calibration(
                base_spec, hip_cal, alignment_old, zero_twist_deg=zero_twist
            )
        )
    if args.anthropometric:
        stature_m, mass_kg = args.anthropometric
        hip_spec = anthropometric_candidate(
            hip_spec, stature_m=stature_m, mass_kg=mass_kg
        )
        qualification_note = (
            f"anthropometric candidate ({stature_m:.3f} m, {mass_kg:.1f} kg), "
            "unqualified: upper-body lengths, masses and inertias changed"
        )
    elif unqualified:
        qualification_note = str(base_spec["upper_body_qualification"])
    else:
        validate_full_body_spec(hip_spec, upper_base)
        qualification_note = "qualified upper body with functional hips"
    hipcal_path.write_text(json.dumps(hip_spec, indent=2, sort_keys=True) + "\n")
    hip_bytes = hipcal_path.read_bytes()
    fixed: dict[str, tuple[str, Sequence[float]]]
    seeds_all: dict[str, tuple[str, Sequence[float]]]
    if args.recalibrate_upper:
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

    # 2. address pose with seed leg offsets
    adapter, kin = lane.kinematics(hip_bytes, {**upper, **LEG_SEEDS})
    q_seed = document_seed(base_spec, kin)
    address = lane.best_address(kin, q_seed)
    address_report = {
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
    if args.static_seeds:
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
        log.info(
            "static trial: neutral fit %.1f mm, address with static seeds %.1f mm",
            neutral.marker_rms_m * 1e3,
            address.marker_rms_m * 1e3,
        )

    if args.fit_closure:
        if not args.static_seeds:
            raise ValueError("--fit-closure needs --static-seeds")
        hip_spec, closure_report = fit_closure_from_address(
            lane, kin, hip_spec, address.q
        )
        hipcal_path.write_text(json.dumps(hip_spec, indent=2, sort_keys=True) + "\n")
        hip_bytes = hipcal_path.read_bytes()
        lane.bounds |= wrist_bounds()
        adapter, kin = lane.kinematics(hip_bytes, {**fixed, **seeds_all})
        address = lane.best_address(kin, address.q)
        closure_report["address_rms_after_m"] = address.marker_rms_m
        address_report["closure_fit"] = closure_report
        log.info(
            "closure fit: weld turned %.1f deg, moved %.1f mm; address %.1f mm with "
            "wrists bounded",
            closure_report["rotation_change_deg"],
            closure_report["translation_change_m"] * 1e3,
            address.marker_rms_m * 1e3,
        )

    # 3. leg calibration, segment scale search, recalibration
    offsets, calibration = lane.calibrate_legs(hip_bytes, fixed, seeds_all, address.q)
    scale_table = []
    best = (float("inf"), 1.0, 1.0, hip_spec)
    for femur in SCALE_GRID:
        for tibia in SCALE_GRID:
            scales = {f"femur_{s}": femur for s in "rl"} | {
                f"tibia_{s}": tibia for s in "rl"
            }
            doc = scale_segments(hip_spec, scales)
            doc_bytes = json.dumps(doc, sort_keys=True).encode()
            rms = lane.pinned_rms(
                doc_bytes, {**fixed, **scaled_offsets(offsets, femur, tibia)}, address.q
            )
            scale_table.append({"femur": femur, "tibia": tibia, "pinned_rms_m": rms})
            log.info(
                "scale femur %.2f tibia %.2f: pinned RMS %.1f mm",
                femur,
                tibia,
                rms * 1e3,
            )
            if rms < best[0]:
                best = (rms, femur, tibia, doc)
    _, femur_scale, tibia_scale, scaled_spec = best
    if not args.anthropometric and not unqualified:
        validate_full_body_spec(scaled_spec, upper_base)
    scaled_path.write_text(json.dumps(scaled_spec, indent=2, sort_keys=True) + "\n")
    spec_bytes = scaled_path.read_bytes()
    offsets, calibration2 = lane.calibrate_legs(
        spec_bytes, fixed, scaled_offsets(offsets, femur_scale, tibia_scale), address.q
    )
    attachments = {**fixed, **offsets}
    adapter, kin = lane.kinematics(spec_bytes, attachments)
    sim = fs.FullBodySimulator(adapter)
    address2 = lane.best_address(kin, address.q)
    address_report["calibrated"] = {
        "marker_rms_m": address2.marker_rms_m,
        "segment_rms_m": segment_rms(
            labels,
            marker_errors(kin, address2.q[None, :], lane.points[:1]),
            lane.valid[:1],
        ),
        "closure_error_m": address2.closure_error_m,
        "lowest_sphere_height_m": address2.lowest_sphere_height_m,
        "support_offset_m": kin.support_offset(address2.q, lane.ground),
        "centre_of_mass": com_report(sim, kin, address2.q, lane.ground),
        "leg_angles_deg": {
            name: float(np.degrees(address2.q[kin.coordinate_order.index(name)]))
            for name in kin.coordinate_order[adapter.upper_body_coordinates :]
        },
        "posture": posture_summary(kin, address2.q),
    }

    # 4. full IK, smoothing, consistency re-solve
    q_ik, fits = full_capture_ik(lane, kin, address2.q)
    errors = marker_errors(kin, q_ik, lane.points)
    q_smooth = smooth_reference(q_ik, RATE_HZ, REFERENCE_CUTOFF_HZ)
    q_ref, ref_fits = consistency_resolve(
        lane, kin, q_smooth, prior_weight=CONSISTENCY_PRIOR, iterations=30
    )
    ref_errors = marker_errors(kin, q_ref, lane.points)
    heights = np.array([min(kin.sphere_heights(q, lane.ground).values()) for q in q_ik])
    ref_heights = np.array(
        [min(kin.sphere_heights(q, lane.ground).values()) for q in q_ref]
    )
    ik_report = {
        "frames": lane.frames,
        "marker_rms_m": float(np.sqrt(np.mean(errors[lane.valid] ** 2))),
        "segment_rms_m": segment_rms(labels, errors, lane.valid),
        "closure_error_max_m": float(max(f.closure_error_m for f in fits)),
        "lowest_sphere_height_min_m": float(heights.min()),
        "lowest_sphere_height_max_m": float(heights.max()),
        "attachments_m": {
            label: {"body": body, "offset_m": [float(v) for v in offset]}
            for label, (body, offset) in attachments.items()
        },
        "reference": {
            "cutoff_hz": REFERENCE_CUTOFF_HZ,
            "consistency_prior": CONSISTENCY_PRIOR,
            "marker_rms_m": float(np.sqrt(np.mean(ref_errors[lane.valid] ** 2))),
            "segment_rms_m": segment_rms(labels, ref_errors, lane.valid),
            "closure_error_max_m": float(max(f.closure_error_m for f in ref_fits)),
            "lowest_sphere_height_min_m": float(ref_heights.min()),
            "lowest_sphere_height_max_m": float(ref_heights.max()),
            "max_deviation_from_smoothed_rad": float(
                np.abs(q_ref[:, 6:] - q_smooth[:, 6:]).max()
            ),
            "max_joint_speed_rad_s": float(
                np.abs(np.gradient(q_ref[:, 6:], lane.times, axis=0)).max()
            ),
            "stance_sphere_drift_max_m": float(
                max(
                    [
                        float(
                            np.linalg.norm(
                                kin.sphere_ground_points(q_ref[k], lane.ground)[name]
                                - kin.sphere_ground_points(q_ref[0], lane.ground)[name]
                            )
                        )
                        for k in range(lane.frames)
                        for name in lane.stance[k]
                        if all(name in lane.stance[j] for j in range(k + 1))
                    ]
                )
            ),
        },
        "calibration": {
            "stride": CALIBRATION_STRIDE,
            "frames": len(lane.calibration_frames),
            "prior_frames": CALIBRATION_PRIOR_FRAMES,
            "prior_offsets_m": {k: list(v[1]) for k, v in LEG_SEEDS.items()},
            "rms_per_iteration_m": list(calibration.rms_per_iteration_m),
            "rms_per_iteration_after_scaling_m": list(calibration2.rms_per_iteration_m),
            "per_marker_rms_m": calibration2.per_marker_rms_m,
            "offsets_m": {
                k: {"body": b, "offset_m": list(o)} for k, (b, o) in offsets.items()
            },
        },
        "segment_scaling": {
            "grid": SCALE_GRID,
            "table": scale_table,
            "femur_scale": femur_scale,
            "tibia_scale": tibia_scale,
            "spec_sha256": canonical_sha256(scaled_spec),
        },
        "joint_ranges_deg": LOWER_LIMB_RANGES_DEG,
        "range_of_motion_flags": rom_flags(q_ref, kin.coordinate_order),
        "bound_widening": BOUND_WIDENING,
        "leg_angle_ranges_deg": {
            name: [
                float(np.degrees(q_ref[:, kin.coordinate_order.index(name)].min())),
                float(np.degrees(q_ref[:, kin.coordinate_order.index(name)].max())),
            ]
            for name in kin.coordinate_order[adapter.upper_body_coordinates :]
        },
    }
    np.savez(
        OUT / "ik_trajectory.npz",
        time_s=lane.times,
        q=q_ik,
        q_ref=q_ref,
        errors_m=errors,
        ref_errors_m=ref_errors,
        valid=lane.valid,
    )

    # 5. forward dynamics tracking
    q_track = smooth_reference(q_ref, RATE_HZ, TRACKING_CUTOFF_HZ)
    zmp = fs.reference_zmp(sim, lane.times, q_track, lane.ground)
    zmp_filter_report: dict | None = None
    if args.zmp_filter:
        q_track, zmp, zmp_filter_report = zmp_filter(lane, kin, sim, q_track, zmp, log)
    shooting_report: dict | None = None
    if args.shooting_fit > 0:
        q_track, zmp, shooting_report = shooting_fit(
            lane, kin, sim, q_track, q_ref, args.shooting_fit, log, args.shooting_gain
        )
    record, sim_q = replay(sim, lane, q_track)
    sim_errors = marker_errors(kin, sim_q, lane.points)
    dynamics_report = {
        "duration_s": float(lane.times[-1]),
        "dt_s": DT_S,
        "controller": {
            "type": "computed torque tracking, unactuated root",
            "omega_rad_s": OMEGA_RAD_S,
            "zeta": 1.0,
            "balance": BALANCE,
            "tracking_cutoff_hz": TRACKING_CUTOFF_HZ,
        },
        "contact_parameters": adapter.contact_parameters.as_document(),
        "zmp_filter": zmp_filter_report,
        "shooting_fit": shooting_report,
        "reference_zmp": {
            "description": "zero-moment point the tracked reference demands "
            "of this model against the support polygon of the spheres on the "
            "plane at each frame; outside means no unilateral foot contact "
            "can realise the reference there",
            "outside_fraction": float((zmp["outside_m"] > 0).mean()),
            "outside_fraction_1s_to_1_5s": float(
                (zmp["outside_m"][(lane.times >= 1.0) & (lane.times < 1.5)] > 0).mean()
            ),
            "outside_max_m": float(zmp["outside_m"].max()),
            "unloaded_fraction": float(zmp["unloaded"].mean()),
            "vertical_grf_over_weight": [
                float(zmp["grf_over_weight"][:, 2].min()),
                float(zmp["grf_over_weight"][:, 2].max()),
            ],
            "horizontal_grf_over_weight_max": float(
                np.linalg.norm(zmp["grf_over_weight"][:, :2], axis=1).max()
            ),
        },
        "marker_rms_m": float(np.sqrt(np.mean(sim_errors[lane.valid] ** 2))),
        "segment_rms_m": segment_rms(labels, sim_errors, lane.valid),
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
        },
        "inside_support_polygon_fraction": float(record.inside_support_polygon.mean()),
        "range_of_motion_flags": rom_flags(sim_q, kin.coordinate_order),
        "root_error_timeline_m": {
            f"{t:.2f}": float(
                np.linalg.norm(
                    sim_q[int(round(t * RATE_HZ)), :3]
                    - q_ref[int(round(t * RATE_HZ)), :3]
                )
            )
            for t in (0.0, 0.25, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75)
        },
        "backswing_to_1s": {
            "root_error_max_m": float(
                np.linalg.norm(sim_q[:361, :3] - q_ref[:361, :3], axis=1).max()
            ),
            "marker_rms_m": float(
                np.sqrt(np.mean(sim_errors[:361][lane.valid[:361]] ** 2))
            ),
            "weight_fraction_min": float(
                record.weight_fraction[record.time_s <= 1.0].min()
            ),
            "weight_fraction_max": float(
                record.weight_fraction[record.time_s <= 1.0].max()
            ),
        },
        "peak_joint_torque_n_m": float(np.abs(record.tau).max()),
        "lowest_sphere_height_min_m": float(record.lowest_sphere_height_m.min()),
        "lowest_sphere_height_max_m": float(record.lowest_sphere_height_m.max()),
    }
    np.savez(
        OUT / "dynamics_record.npz",
        time_s=record.time_s,
        q=record.q,
        v=record.v,
        tau=record.tau,
        normal_force_n=record.normal_force_n,
        weight_fraction=record.weight_fraction,
        cop_m=record.centre_of_pressure_m,
        inside=record.inside_support_polygon,
        lowest_sphere_height_m=record.lowest_sphere_height_m,
        sim_errors_m=sim_errors,
    )
    lookat = np.nanmean(lane.points[0], axis=0)
    names = tuple(kin.coordinate_order)
    render_playback(spec_bytes, names, q_ref, lookat, OUT / "ik_playback.gif")
    render_playback(spec_bytes, names, sim_q, lookat, OUT / "tracking_playback.gif")

    receipt = build_ground_support_receipt(
        GroundSupportReceiptInputs(
            base_spec=base_spec,
            spec_path=args.spec,
            scaled_path=scaled_path,
            hipcal_path=hipcal_path,
            recalibrate_upper=args.recalibrate_upper,
            anthropometric=args.anthropometric,
            qualification_note=qualification_note,
            spec_bytes=spec_bytes,
            hip_report=hip_report,
            candidate_bytes=candidate_bytes,
            c3d_path=C3D,
            capture_name=args.capture,
            lane=lane,
            address_report=address_report,
            ik_report=ik_report,
            dynamics_report=dynamics_report,
            kin=kin,
            q_ref=q_ref,
            elapsed_s=time.perf_counter() - t_start,
        )
    )
    (OUT / "receipt.json").write_text(
        json.dumps(receipt, indent=2, default=float) + "\n"
    )
    log.info(
        json.dumps(
            {k: receipt[k] for k in ("address", "dynamics")}, indent=1, default=float
        )
    )
    log.info(
        "ik %s",
        json.dumps(
            {
                k: ik_report[k]
                for k in (
                    "marker_rms_m",
                    "segment_rms_m",
                    "reference",
                    "segment_scaling",
                    "leg_angle_ranges_deg",
                )
            },
            indent=1,
            default=float,
        ),
    )
    log.info(
        "calibration rms %s -> scaled %s",
        calibration.rms_per_iteration_m,
        calibration2.rms_per_iteration_m,
    )


if __name__ == "__main__":
    main()
