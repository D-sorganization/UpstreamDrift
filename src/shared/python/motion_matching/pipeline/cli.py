"""CLI runner for engine-independent motion-matching ground support pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np

from src.shared.python.contracts import precondition
from src.shared.python.motion_matching import (
    full_body_forward_dynamics as fs,
)
from src.shared.python.motion_matching.club_face_target import (
    FACE_FRAME,
    FACE_ORIENTATION_WEIGHT,
    HEAD_TRIAD_LABELS,
    face_fit_summary,
    model_face_normals,
    observe_capture_face,
)
from src.shared.python.motion_matching.full_body_spec import (
    validate_full_body_spec,
)
from src.shared.python.motion_matching.hip_calibration import (
    pelvis_alignment_from_spec,
)
from src.shared.python.motion_matching.pipeline.address import (
    AddressStageInputs,
    HipCalibrationOptions,
    calibrated_address_summary,
    prepare_hip_spec,
    scaled_offsets,
    search_segment_scales,
    solve_address_stage,
)
from src.shared.python.motion_matching.pipeline.address_feet import (
    build_foot_targets,
    foot_progression_series,
    model_feet_deg,
    record_foot_progression,
    seed_document_feet,
    split_address_coordinates,
)
from src.shared.python.motion_matching.pipeline.constants import (
    CANDIDATE,
    CAPTURE_NAMES,
    CONSISTENCY_PRIOR,
    DEFAULT_MJX_ITERATIONS,
    LEG_SEEDS,
    REFERENCE_CUTOFF_HZ,
    SHOOTING_RELAXATION,
    SPEC,
    TRACKING_CUTOFF_HZ,
    UPPER_SPEC,
    capture_path,
)
from src.shared.python.motion_matching.pipeline.trajectory_optimiser import (
    TRAJECTORY_OPTIMISERS,
    run_trajectory_optimiser,
    validate_trajectory_optimiser,
)
from src.shared.python.motion_matching.pipeline.dynamics import (
    DynamicsReportInputs,
    ShootingFitConfig,
    build_dynamics_report,
    replay,
    score_reference,
    shooting_fit,
    zmp_filter,
)
from src.shared.python.motion_matching.pipeline.centroidal_filter import (
    centroidal_filter,
)
from src.shared.python.motion_matching.pipeline.finish_feasibility import (
    finish_feasibility_report,
)
from src.shared.python.motion_matching.pipeline.gaze_residual import (
    head_gaze_receipt,
)
from src.shared.python.motion_matching.pipeline.lane import (
    Lane,
    configure_lane,
    document_seed,
    expand_stance_for_width,
    fitted_grip,
    wrist_bounds,
)
from src.shared.python.motion_matching.pipeline.turn_split import (
    DEFAULT_SHOULDER_GIRDLE_WEIGHT,
    DEFAULT_THORAX_WEIGHT,
    add_turn_split_arguments,
    turn_split_active,
    turn_split_report,
)
from src.shared.python.motion_matching.pipeline.receipt import (
    GroundSupportReceiptInputs,
    build_ground_support_receipt,
    log_pipeline_summary,
)
from src.shared.python.motion_matching.pipeline.reference import (
    IKReportInputs,
    build_ik_report,
    consistency_resolve,
    full_capture_ik,
    marker_errors,
    render_playback,
    smooth_reference,
)
from src.shared.python.motion_matching.pipeline.plant import (
    get_plant,
)

logger = logging.getLogger(__name__)


#: Receipt stride of the model toe-out time series (frames).
FOOT_SERIES_STRIDE = 6


def _positive_int(value: str) -> int:
    try:
        val = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid integer: {value!r}") from e
    if val <= 0:
        raise argparse.ArgumentTypeError(f"--mjx-iterations must be > 0, got {val}")
    return val


def _nonnegative_float(value: str) -> float:
    try:
        val = float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid number: {value!r}") from e
    if not np.isfinite(val) or val < 0:
        raise argparse.ArgumentTypeError(f"--gaze-weight must be >= 0, got {val}")
    return val


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ground support pipeline runner."""
    parser = argparse.ArgumentParser(
        description="Ground-supported full-body pipeline runner with pluggable plant engines."
    )
    parser.add_argument("--out", type=Path, default=Path.cwd())
    parser.add_argument(
        "--address-only",
        action="store_true",
        help="stop after the calibrated address and write address_report.json",
    )
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
        "--gaze-weight",
        type=_nonnegative_float,
        default=0.0,
        help=(
            "soft head-gaze residual weight (eyes on the ball until impact + "
            "0.03 s, then a 0.35 s release to the target line); 0 keeps the "
            "marker-faithful head (default)"
        ),
    )
    parser.add_argument(
        "--capture",
        choices=list(CAPTURE_NAMES),
        default="driver",
        help="which canonical tour-average capture to match",
    )
    parser.add_argument(
        "--fit-closure",
        action="store_true",
        help="fit two-hand closure weld from address (needs --static-seeds)",
    )
    parser.add_argument(
        "--bound-wrists",
        action="store_true",
        help="impose human wrist/forearm ranges in IK without fitted grip rotation",
    )
    parser.add_argument(
        "--free-wrists",
        action="store_true",
        help="leave wrists/forearms unbounded even with fitted grip rotation",
    )
    parser.add_argument(
        "--shooting-fit",
        type=int,
        default=0,
        metavar="N",
        help="contact-aware shooting fit of tracked reference (FB-5): N passes",
    )
    parser.add_argument(
        "--shooting-gain",
        type=float,
        default=SHOOTING_RELAXATION,
        help="iterative-learning gain on pelvis command per shooting pass",
    )
    parser.add_argument(
        "--zmp-filter",
        action="store_true",
        help="dynamics-filter reference to keep ZMP inside support polygon",
    )
    parser.add_argument(
        "--centroidal-filter",
        action="store_true",
        help="centroidal feasibility filter v2: full ZMP, vertical force and "
        "friction cone over the finish (runs after --zmp-filter)",
    )
    parser.add_argument(
        "--foot-half-width-m",
        type=float,
        default=None,
        help="add lateral heel/forefoot contact spheres at +/- this distance so "
        "the foot is a sole, not a centre line (#11671)",
    )
    parser.add_argument(
        "--torsional-patch-m",
        type=float,
        default=None,
        help="torsional (spin) friction patch radius applied at every loaded "
        "contact sphere (#11671)",
    )
    parser.add_argument(
        "--static-seeds",
        action="store_true",
        help="place every marker from a neutral-spine static trial",
    )
    parser.add_argument(
        "--engine",
        default="mujoco",
        help="plant physics engine (mujoco, drake, pinocchio)",
    )
    parser.add_argument(
        "--backend",
        choices=["mujoco", "pink"],
        default="mujoco",
        help="kinematic tracking backend engine (mujoco or pink)",
    )
    parser.add_argument(
        "--pink-step-mode",
        choices=["physical", "projection"],
        default="physical",
        help="integration step mode for Pink solver",
    )
    parser.add_argument(
        "--pink-solver",
        default="quadprog",
        help="QP solver backend for Pink",
    )
    parser.add_argument(
        "--pink-limit-policy",
        choices=["enforce", "ignore"],
        default="enforce",
        help="joint limit policy for Pink solver",
    )
    parser.add_argument(
        "--ik-backend",
        choices=["lm", "mujoco-minimize"],
        default="lm",
        help="MuJoCo marker IK backend (lm or mujoco-minimize)",
    )
    parser.add_argument(
        "--tracking",
        choices=["kkt", "mj-inverse", "wrench-qp"],
        default="kkt",
        help="computed-torque tracking backend (kkt, mj-inverse, or the opt-in "
        "contact-wrench QP wrench-qp, #11670)",
    )
    parser.add_argument(
        "--trajectory-optimiser",
        choices=sorted(TRAJECTORY_OPTIMISERS),
        default="none",
        help="trajectory optimisation backend (none, mjx-knots)",
    )
    parser.add_argument(
        "--mjx-iterations",
        type=_positive_int,
        default=DEFAULT_MJX_ITERATIONS,
        help="number of Adam iterations for MJX knot optimisation (default: 40)",
    )
    parser.add_argument(
        "--mjx-method",
        choices=["adam", "lbfgs"],
        default="adam",
        help="optimisation algorithm for MJX knot optimisation (adam or lbfgs)",
    )
    parser.add_argument(
        "--neural-mode",
        choices=["classical", "preview", "verified"],
        default="classical",
        help="neural-assisted matching mode (classical, preview, verified)",
    )
    parser.add_argument(
        "--neural-model",
        default=None,
        help="neural model identity from qualified roster (e.g. driven_double_pendulum)",
    )
    parser.add_argument(
        "--no-neural-fallback",
        action="store_true",
        help="disable classical fallback when neural proposal is rejected or unavailable",
    )
    parser.add_argument(
        "--neural-checkpoint",
        type=Path,
        default=None,
        help="optional explicit path to qualified neural model checkpoint",
    )
    parser.add_argument(
        "--foot-progression",
        choices=("off", "capture", "default"),
        default="off",
        help=(
            "address foot toe-out (OSV-4, #11730): 'capture' sets each foot to the "
            "capture's measured toe-out (flagged 20 deg default where the markers "
            "are unreliable), 'default' forces 20 deg per foot, 'off' (default) "
            "keeps the legacy multi-start result. Also squares the forefoot marker "
            "seeds and records model vs capture angles in the receipt."
        ),
    )
    parser.add_argument(
        "--face-weight",
        type=float,
        default=FACE_ORIENTATION_WEIGHT,
        help=(
            "weight of the club-face orientation residual (OSV-10, #11759): the "
            "IK pulls the rendered face normal onto the one the capture head "
            "triad implies; 0 restores the marker-only fit"
        ),
    )
    add_turn_split_arguments(parser)
    return parser


def _attach_face_report(
    ik_report: dict[str, Any],
    lane: Lane,
    cal_res: Any,
    q_pair: tuple[np.ndarray, np.ndarray],
    weight: float,
) -> None:
    """Add the face residual's receipt block when the residual was active."""
    if lane.face_targets is not None:
        ik_report["face_orientation"] = _face_orientation_report(
            lane, cal_res, *q_pair, weight
        )


def _face_orientation_report(
    lane: Lane, cal_res: Any, q_ik: np.ndarray, q_ref: np.ndarray, weight: float
) -> dict[str, Any]:
    """Receipt block of the face residual: weight and model-vs-capture fit."""
    report: dict[str, Any] = {
        "weight": weight,
        "frame": FACE_FRAME,
        "triad": list(HEAD_TRIAD_LABELS),
        "targeted_frames": sum(t is not None for t in lane.face_targets or ()),
    }
    if not hasattr(cal_res.kin, "body_poses"):
        report["reason"] = "IK provider has no body_poses; fit not measured"
        return report
    capture, _ = observe_capture_face(
        lane.points, lane.valid, lane.labels, cal_res.attachments, cal_res.scaled_spec
    )
    for name, q in (("ik", q_ik), ("reference", q_ref)):
        model = model_face_normals(cal_res.kin, q, cal_res.scaled_spec)
        report[name] = face_fit_summary(model, capture)
    return report


@dataclass(frozen=True)
class PipelineContext:
    """Working context and configuration for pipeline stages."""

    args: argparse.Namespace
    out_dir: Path
    c3d_path: Path
    engine: str
    log: logging.Logger
    t_start: float


def _init_pipeline(args: argparse.Namespace) -> PipelineContext:
    """Validate backend availability and initialize directories and logging."""
    if args.backend == "pink":
        from src.shared.python.motion_matching.pipeline.lane import (
            probe_pink_capability,
        )

        available, diag = probe_pink_capability()
        if not available:
            raise RuntimeError(
                f"Pink backend requested but not available: {diag['reason']}"
            )
    if args.backend == "pink" and getattr(args, "ik_backend", "lm") != "lm":
        raise RuntimeError(
            "Pink backend cannot be combined with --ik-backend mujoco-minimize"
        )
    if args.backend == "pink" and getattr(args, "tracking", "kkt") != "kkt":
        raise RuntimeError("Pink backend cannot be combined with --tracking mj-inverse")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return PipelineContext(
        args=args,
        out_dir=out_dir,
        c3d_path=capture_path(args.capture),
        engine=getattr(args, "engine", "mujoco"),
        log=logging.getLogger("ground_support"),
        t_start=time.perf_counter(),
    )


@dataclass(frozen=True)
class _CalibrateAndScaleResult:
    scaled_spec: dict[str, Any]
    spec_bytes: bytes
    address_report: dict[str, Any]
    adapter: Any
    kin: Any
    sim: Any
    address2: Any
    attachments: Any
    offsets: Any
    calibration: Any
    calibration2: Any
    hip_report: dict[str, Any]
    qualification_note: str


def _validate_scaled_spec(
    args: argparse.Namespace,
    base_spec: dict[str, Any],
    scaled_spec: dict[str, Any],
    upper_base: dict[str, Any],
) -> None:
    if not args.anthropometric and "unqualified" not in str(
        base_spec.get("upper_body_qualification", "")
    ):
        validate_full_body_spec(scaled_spec, upper_base)


def _pin_width_spheres(args: Any, lane: Lane, hip_spec: dict[str, Any]) -> None:
    """Pin lateral foot spheres in the IK stance when a foot width is set."""
    if getattr(args, "foot_half_width_m", None) is not None:
        lane.stance = expand_stance_for_width(
            lane.stance, [s["name"] for s in hip_spec["contact"]["spheres"]]
        )


def _calibrate_and_scale(
    ctx: PipelineContext,
    lane: Lane,
    base_spec: dict[str, Any],
    upper_base: dict[str, Any],
    upper: dict[str, tuple[str, tuple[float, float, float]]],
    labels: tuple[str, ...],
) -> _CalibrateAndScaleResult:
    """Execute hip calibration, address solve, segment scale search, and leg calibration."""
    args = ctx.args
    log = ctx.log
    hipcal_path = ctx.out_dir / "full_body_spec_hipcal.json"
    scaled_path = ctx.out_dir / "full_body_spec_hipcal_scaled.json"

    # The alignment of the spec being rewritten, never another spec's build
    # receipt: a foreign alignment rotates both hips (#12109).
    alignment_old = (
        pelvis_alignment_from_spec(base_spec) if not args.skip_hip_calibration else None
    )
    hip_spec, qualification_note, hip_report, fixed, seeds_all = prepare_hip_spec(
        lane,
        base_spec,
        upper_base,
        labels,
        upper,
        alignment_old,
        options=HipCalibrationOptions(
            skip_hip_calibration=args.skip_hip_calibration,
            anthropometric=tuple(args.anthropometric) if args.anthropometric else None,
            recalibrate_upper=args.recalibrate_upper,
            foot_half_width_m=getattr(args, "foot_half_width_m", None),
            torsional_patch_m=getattr(args, "torsional_patch_m", None),
        ),
    )
    hipcal_path.write_text(
        json.dumps(hip_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hip_bytes = hipcal_path.read_bytes()
    _pin_width_spheres(args, lane, hip_spec)
    lane.plant = get_plant(ctx.engine, hip_spec)

    stage2 = solve_address_stage(
        AddressStageInputs(
            lane=lane,
            base_spec=base_spec,
            hip_spec=hip_spec,
            hip_bytes=hip_bytes,
            upper=upper,
            fixed=fixed,
            seeds_all=seeds_all,
            labels=labels,
            hipcal_path=hipcal_path,
            static_seeds=args.static_seeds,
            fit_closure=args.fit_closure,
            log=log,
        )
    )
    address, address_report = stage2.address, stage2.address_report
    fixed, seeds_all, hip_spec = stage2.fixed, stage2.seeds_all, stage2.hip_spec

    offsets, calibration = lane.calibrate_legs(hip_bytes, fixed, seeds_all, address.q)
    scaled_spec, femur_scale, tibia_scale, scale_table = search_segment_scales(
        lane, hip_spec, fixed, offsets, address.q, log=log
    )
    _validate_scaled_spec(args, base_spec, scaled_spec, upper_base)
    scaled_path.write_text(
        json.dumps(scaled_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    spec_bytes = scaled_path.read_bytes()
    lane.plant = get_plant(ctx.engine, scaled_spec)
    offsets, calibration2 = lane.calibrate_legs(
        spec_bytes, fixed, scaled_offsets(offsets, femur_scale, tibia_scale), address.q
    )
    attachments = {**fixed, **offsets}
    adapter, kin = lane.kinematics(
        spec_bytes, attachments, ik_backend=getattr(args, "ik_backend", "lm")
    )
    sim = fs.FullBodySimulator(adapter)
    address2 = lane.best_address(kin, address.q)
    address_report["calibrated"] = calibrated_address_summary(
        sim, kin, address2, lane, labels, adapter
    )
    record_foot_progression(address_report, kin, address2.q, lane.feet)
    return _CalibrateAndScaleResult(
        scaled_spec,
        spec_bytes,
        address_report,
        adapter,
        kin,
        sim,
        address2,
        attachments,
        offsets,
        calibration,
        calibration2,
        hip_report,
        qualification_note,
    )


class _PinkFrameFit:
    def __init__(self, closure_error_m: float, marker_rms_m: float) -> None:
        self.closure_error_m = float(closure_error_m)
        self.marker_rms_m = float(marker_rms_m)


def _solve_pink_ik(
    ctx: PipelineContext,
    lane: Lane,
    kin: Any,
    scaled_spec: dict[str, Any],
    labels: tuple[str, ...],
    initial_q: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, list[Any], list[Any], np.ndarray, np.ndarray, dict[str, Any]
]:
    """Solve kinematic trajectory using Pink constrained IK."""
    from src.engines.physics_engines.pinocchio.python.pink_trajectory import (
        PinkTrajectoryService,
    )
    from src.shared.python.motion_matching.constrained_ik import (
        IKOptions,
        IKTrajectoryRequest,
    )

    args = ctx.args
    pink_service = PinkTrajectoryService(scaled_spec)
    time_s = np.asarray(lane.times, dtype=np.float64)
    points = np.asarray(lane.points, dtype=np.float64)
    valid = np.asarray(lane.valid, dtype=bool)

    solve_req = IKTrajectoryRequest(
        initial_q=initial_q,
        time_s=time_s,
        marker_targets=points,
        validity_mask=valid,
        labels=labels,
        model_name=str(scaled_spec.get("name", "golf_humanoid")),
        posture_target=initial_q,
    )
    ik_opts = IKOptions(
        step_mode=args.pink_step_mode,
        solver=args.pink_solver,
        limit_policy=args.pink_limit_policy,
    )
    pink_result = pink_service.solve_trajectory(solve_req, ik_opts)
    q_ik = pink_result.configurations
    errors = marker_errors(kin, q_ik, lane.points)

    fits = [
        _PinkFrameFit(
            closure_error_m=(
                float(r.weld_translation_error_m)
                if np.isfinite(r.weld_translation_error_m)
                else 0.0
            ),
            marker_rms_m=(
                float(np.mean(list(r.marker_errors_m.values())))
                if r.marker_errors_m
                and all(np.isfinite(list(r.marker_errors_m.values())))
                else 0.0
            ),
        )
        for r in pink_result.frame_residuals
    ]
    q_smooth = smooth_reference(q_ik, lane.rate_hz, REFERENCE_CUTOFF_HZ)
    audit_result = pink_service.audit_trajectory(q_smooth, solve_req, ik_opts)
    all_converged = bool(pink_result.passed)
    rate_limits_respected = bool(
        all(len(a.exceeded_joints) == 0 for a in audit_result.rate_audits)
    )
    max_ratio = float(
        max((a.max_velocity_ratio for a in audit_result.rate_audits), default=0.0)
    )
    is_qualified = bool(all_converged and audit_result.passed and rate_limits_respected)

    closure_residual_m = float(
        max(
            (
                float(r.weld_translation_error_m)
                for r in pink_result.frame_residuals
                if np.isfinite(r.weld_translation_error_m)
            ),
            default=float("nan"),
        )
    )
    if not np.isfinite(closure_residual_m):
        closure_residual_m = float("inf")
    closure_budget_m = 1.0e-4
    if is_qualified and closure_residual_m > closure_budget_m:
        is_qualified = False
    constrained_ik_dict = {
        "backend_name": "pink",
        "solver": args.pink_solver,
        "model_name": str(scaled_spec.get("name", "golf_humanoid")),
        "capture_name": args.capture,
        "step_mode": args.pink_step_mode,
        "limit_policy": args.pink_limit_policy,
        "task_policy": "dual_grip_hard_equality",
        "time_semantics": (
            "strict_physical_elapsed_dt"
            if args.pink_step_mode == "physical"
            else "uniform_fixed_dt"
        ),
        "frame_count": int(lane.frames),
        "frame_success_count": int(np.sum(pink_result.frame_success)),
        "all_frames_converged": all_converged,
        "first_failed_frame": pink_result.first_failed_frame,
        "per_frame_status": [bool(x) for x in pink_result.frame_success],
        "max_velocity_ratio": max_ratio,
        "closure_residual_m": closure_residual_m,
        "closure_residual_budget_m": closure_budget_m,
        "is_qualified": is_qualified,
        "qualification_state": "qualified" if is_qualified else "disqualified",
    }
    q_ref = q_smooth
    ref_fits = [
        _PinkFrameFit(
            closure_error_m=(
                float(r.weld_translation_error_m)
                if np.isfinite(r.weld_translation_error_m)
                else 0.0
            ),
            marker_rms_m=(
                float(np.mean(list(r.marker_errors_m.values())))
                if r.marker_errors_m
                and all(np.isfinite(list(r.marker_errors_m.values())))
                else 0.0
            ),
        )
        for r in audit_result.frame_residuals
    ]
    ref_errors = marker_errors(kin, q_ref, lane.points)
    return q_ik, q_ref, fits, ref_fits, errors, ref_errors, constrained_ik_dict


def _solve_trajectory_ik(
    ctx: PipelineContext,
    lane: Lane,
    kin: Any,
    scaled_spec: dict[str, Any],
    labels: tuple[str, ...],
    initial_q: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[Any],
    list[Any],
    np.ndarray,
    np.ndarray,
    dict[str, Any] | None,
]:
    """Solve full capture IK trajectory via selected kinematic backend."""
    if ctx.args.backend == "pink":
        return _solve_pink_ik(ctx, lane, kin, scaled_spec, labels, initial_q)
    q_ik, fits = full_capture_ik(lane, kin, initial_q)
    errors = marker_errors(kin, q_ik, lane.points)
    q_smooth = smooth_reference(q_ik, lane.rate_hz, REFERENCE_CUTOFF_HZ)
    q_ref, ref_fits = consistency_resolve(
        lane, kin, q_smooth, prior_weight=CONSISTENCY_PRIOR, iterations=30
    )
    ref_errors = marker_errors(kin, q_ref, lane.points)
    return q_ik, q_ref, fits, ref_fits, errors, ref_errors, None


def _save_dynamics_record(
    out_dir: Path,
    record: Any,
    sim_errors: np.ndarray,
    lane: Lane,
    q_track: np.ndarray,
) -> None:
    """Write ``dynamics_record.npz``.

    The NPZ also keeps the tracked reference (``q_track`` on ``track_time_s``)
    so same-input bundles can be rebuilt from a run directory (#11607).
    """
    np.savez(
        out_dir / "dynamics_record.npz",
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
        q_track=q_track,
        track_time_s=lane.times,
    )


def _render_playbacks(
    out_dir: Path,
    cal_res: _CalibrateAndScaleResult,
    kin: Any,
    lane: Lane,
    q_ref: np.ndarray,
    sim_q: np.ndarray,
) -> None:
    """Render IK and tracking playback GIFs.

    The playback looks at the capture's first-frame marker centroid and runs
    at the capture's own rate.
    """
    lookat = np.nanmean(lane.points[0], axis=0)
    names = tuple(kin.coordinate_order)
    for q, name in ((q_ref, "ik_playback.gif"), (sim_q, "tracking_playback.gif")):
        render_playback(
            cal_res.spec_bytes,
            names,
            q,
            lookat,
            out_dir / name,
            rate_hz=lane.rate_hz,
        )


def _write_receipt(out_dir: Path, receipt: dict[str, Any]) -> None:
    (out_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, default=float) + "\n", encoding="utf-8"
    )


def _apply_trajectory_optimiser(
    args: argparse.Namespace,
    out_dir: Path,
    receipt: dict[str, Any],
    *,
    lane: Lane,
    kin: Any,
    sim: Any,
) -> None:
    """Run the selected trajectory optimiser and record its summary in ``receipt``.

    An optimised reference is rescored through the shared ``sim`` so its
    numbers are comparable with the unoptimised and shooting runs; a stage
    that reports success without writing the reference is an error.
    """
    trajectory_optimiser = validate_trajectory_optimiser(
        getattr(args, "trajectory_optimiser", "none")
    )
    if trajectory_optimiser == "none":
        return
    # The MJX exporter reads receipt.json, so write it once before the stage;
    # the caller writes it again with the stage summary.
    _write_receipt(out_dir, receipt)
    raw_method = getattr(args, "mjx_method", "adam")
    method: Literal["adam", "lbfgs"] = "lbfgs" if raw_method == "lbfgs" else "adam"
    opt_summary = run_trajectory_optimiser(
        trajectory_optimiser,
        out_dir,
        iterations=getattr(args, "mjx_iterations", DEFAULT_MJX_ITERATIONS),
        method=method,
    )
    if opt_summary is not None:
        npz_path = out_dir / "mjx_optimised_reference.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"optimiser wrote no reference: {npz_path}")
        q_opt = np.load(npz_path)["q"]
        opt_summary["shared_simulator_replay"] = score_reference(
            lane, kin, sim, q_opt, tracking_backend=getattr(args, "tracking", "kkt")
        )
        receipt["trajectory_optimiser"] = opt_summary


def _feasibility_filters(
    args: argparse.Namespace,
    context: tuple[Lane, Any, Any, logging.Logger],
    q_track: np.ndarray,
    zmp: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Run the optional cart-table and centroidal feasibility filters in order."""
    lane, kin, sim, log = context
    zmp_report: dict[str, Any] | None = None
    centroidal_report: dict[str, Any] | None = None
    if args.zmp_filter:
        q_track, zmp, zmp_report = zmp_filter(lane, kin, sim, q_track, zmp, log)
    if getattr(args, "centroidal_filter", False):
        q_track, zmp, centroidal_report = centroidal_filter(
            lane, kin, sim, q_track, zmp, log
        )
    return q_track, zmp, zmp_report, centroidal_report


def _simulate_and_receipt(
    ctx: PipelineContext,
    lane: Lane,
    kin: Any,
    sim: Any,
    adapter: Any,
    labels: tuple[str, ...],
    q_ref: np.ndarray,
    cal_res: _CalibrateAndScaleResult,
    base_spec: dict[str, Any],
    ik_report: dict[str, Any],
) -> dict[str, Any]:
    """Execute forward dynamics tracking replay, renders, and receipt generation."""
    args = ctx.args
    out_dir = ctx.out_dir
    log = ctx.log
    tracking = getattr(args, "tracking", "kkt")
    q_track = smooth_reference(q_ref, lane.rate_hz, TRACKING_CUTOFF_HZ)
    zmp = fs.reference_zmp(sim, lane.times, q_track, lane.ground)
    q_track, zmp, zmp_filter_report, centroidal_report = _feasibility_filters(
        args, (lane, kin, sim, log), q_track, zmp
    )
    shooting_report: dict[str, Any] | None = None
    if args.shooting_fit > 0:
        q_track, zmp, shooting_report = shooting_fit(
            lane,
            kin,
            sim,
            q_track,
            q_ref,
            log,
            ShootingFitConfig(
                iterations=args.shooting_fit,
                gain=args.shooting_gain,
                tracking_backend=tracking,
            ),
        )
    record, sim_q = replay(sim, lane, q_track, tracking_backend=tracking)
    finish = finish_feasibility_report(
        sim,
        times_track=lane.times,
        q_track=q_track,
        record=record,
        zmp=zmp,
        ground=lane.ground,
        q_ik=q_ref,
    )
    dynamics_report, sim_errors = build_dynamics_report(
        DynamicsReportInputs(
            lane=lane,
            kin=kin,
            adapter=adapter,
            labels=labels,
            record=record,
            sim_q=sim_q,
            q_ref=q_ref,
            zmp=zmp,
            zmp_filter_report=zmp_filter_report,
            centroidal_filter_report=centroidal_report,
            shooting_report=shooting_report,
            tracking_backend=tracking,
            finish_feasibility=finish,
        )
    )
    _save_dynamics_record(out_dir, record, sim_errors, lane, q_track)
    _render_playbacks(out_dir, cal_res, kin, lane, q_ref, sim_q)
    receipt = build_ground_support_receipt(
        GroundSupportReceiptInputs(
            backend=args.backend,
            ik_backend=getattr(args, "ik_backend", "lm"),
            tracking_backend=tracking,
            base_spec=base_spec,
            spec_path=Path(args.spec),
            scaled_path=out_dir / "full_body_spec_hipcal_scaled.json",
            hipcal_path=out_dir / "full_body_spec_hipcal.json",
            recalibrate_upper=args.recalibrate_upper,
            anthropometric=args.anthropometric,
            qualification_note=cal_res.qualification_note,
            spec_bytes=cal_res.spec_bytes,
            hip_report=cal_res.hip_report,
            candidate_bytes=CANDIDATE.read_bytes(),
            c3d_path=ctx.c3d_path,
            capture_name=args.capture,
            lane=lane,
            address_report=cal_res.address_report,
            ik_report=ik_report,
            dynamics_report=dynamics_report,
            kin=kin,
            q_ref=q_ref,
            elapsed_s=time.perf_counter() - ctx.t_start,
        )
    )
    receipt["engine"] = ctx.engine
    receipt["head_gaze"] = head_gaze_receipt(lane, kin, q_ref)
    _apply_trajectory_optimiser(args, out_dir, receipt, lane=lane, kin=kin, sim=sim)
    _write_receipt(out_dir, receipt)
    log_pipeline_summary(
        log, receipt, ik_report, cal_res.calibration, cal_res.calibration2
    )
    return receipt


@dataclass(frozen=True)
class CalibratedRun:
    """Pipeline state after hip calibration, the address solve and leg scaling."""

    ctx: PipelineContext
    lane: Lane
    base_spec: dict[str, Any]
    labels: tuple[str, ...]
    cal_res: _CalibrateAndScaleResult


@precondition(lambda args: args is not None, "args must not be None")
def calibrate_run(args: argparse.Namespace) -> CalibratedRun:
    """Run the pipeline through the calibrated address (no trajectory or dynamics)."""
    ctx = _init_pipeline(args)
    base_spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    upper_base = json.loads(UPPER_SPEC.read_text(encoding="utf-8"))
    upper = {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in base_spec["marker_attachments"].items()
        if a["offset_m"] is not None
    }
    labels = tuple({**upper, **LEG_SEEDS})

    plant = get_plant(ctx.engine, base_spec)
    lane = Lane(labels, ctx.c3d_path, plant=plant)
    configure_lane(lane, base_spec)
    lane.feet = build_foot_targets(lane, getattr(args, "foot_progression", "off"))
    if lane.feet is not None:
        seed_kin = plant.create_ik(lane.leg_seeds())
        base_spec = seed_document_feet(
            base_spec, seed_kin, lane.feet, document_seed(base_spec, seed_kin)
        )
    if args.bound_wrists and args.free_wrists:
        raise ValueError("--bound-wrists and --free-wrists exclude each other")
    if args.bound_wrists or (fitted_grip(base_spec) and not args.free_wrists):
        lane.bounds |= wrist_bounds()
        ctx.log.info("wrists and forearms bounded to the human ranges in the IK")

    lane.gaze_weight = float(getattr(args, "gaze_weight", 0.0))

    cal_res = _calibrate_and_scale(ctx, lane, base_spec, upper_base, upper, labels)
    return CalibratedRun(ctx, lane, base_spec, labels, cal_res)


def run_address_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Return the calibrated address report (``foot_progression`` included).

    Adds ``hip_coordinates_deg``: the six hip coordinates at the calibrated
    address, so a coordinate at its range limit is visible in the report.
    """
    cal_res = calibrate_run(args).cal_res
    report = cal_res.address_report
    names = list(cal_res.kin.coordinate_order)
    report["hip_coordinates_deg"] = {
        name: float(np.degrees(cal_res.address2.q[names.index(name)]))
        for name in names
        if name.startswith("hip_")
    }
    angles, translations = split_address_coordinates(names, cal_res.address2.q)
    report["address_coordinates_deg"] = angles
    report["address_translations_m"] = translations
    out = Path(args.out) / "address_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    return report


@precondition(lambda args: args is not None, "args must not be None")
def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full-body ground support matching pipeline."""
    run = calibrate_run(args)
    ctx, lane, base_spec, labels, cal_res = (
        run.ctx,
        run.lane,
        run.base_spec,
        run.labels,
        run.cal_res,
    )
    lane.set_face_targets(cal_res.attachments, cal_res.scaled_spec, args.face_weight)
    lane.set_turn_split(
        cal_res.attachments,
        getattr(args, "thorax_weight", DEFAULT_THORAX_WEIGHT),
        getattr(args, "shoulder_girdle_weight", DEFAULT_SHOULDER_GIRDLE_WEIGHT),
    )

    (
        q_ik,
        q_ref,
        fits,
        ref_fits,
        errors,
        ref_errors,
        constrained_ik_dict,
    ) = _solve_trajectory_ik(
        ctx, lane, cal_res.kin, cal_res.scaled_spec, labels, cal_res.address2.q
    )

    if lane.feet is not None:
        cal_res.address_report["foot_progression"]["trajectory"] = {
            "stride_frames": FOOT_SERIES_STRIDE,
            "finish_model_deg": model_feet_deg(cal_res.kin, q_ik[-1], lane.feet),
            "series_model_deg": foot_progression_series(
                cal_res.kin, q_ik, lane.feet, FOOT_SERIES_STRIDE
            ),
        }

    ik_report = build_ik_report(
        IKReportInputs(
            lane=lane,
            kin=cal_res.kin,
            adapter=cal_res.adapter,
            labels=labels,
            attachments=cal_res.attachments,
            offsets=cal_res.offsets,
            calibration=cal_res.calibration,
            calibration2=cal_res.calibration2,
            scaled_spec=cal_res.scaled_spec,
            scale_table=[],
            femur_scale=1.0,
            tibia_scale=1.0,
            q_ik=q_ik,
            fits=fits,
            errors=errors,
            q_smooth=q_ref,
            q_ref=q_ref,
            ref_fits=ref_fits,
            ref_errors=ref_errors,
            constrained_ik=constrained_ik_dict,
        )
    )
    _attach_face_report(ik_report, lane, cal_res, (q_ik, q_ref), args.face_weight)
    if turn_split_active(lane):
        ik_report["turn_split"] = turn_split_report(lane)
    np.savez(
        ctx.out_dir / "ik_trajectory.npz",
        time_s=lane.times,
        q=q_ik,
        q_ref=q_ref,
        errors_m=errors,
        ref_errors_m=ref_errors,
        valid=lane.valid,
    )

    return _simulate_and_receipt(
        ctx,
        lane,
        cal_res.kin,
        cal_res.sim,
        cal_res.adapter,
        labels,
        q_ref,
        cal_res,
        base_spec,
        ik_report,
    )


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    if args.address_only:
        run_address_stage(args)
        return
    run_pipeline(args)


if __name__ == "__main__":
    main()
