"""CLI runner for engine-independent motion-matching ground support pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.shared.python.contracts import precondition
from src.shared.python.motion_matching import (
    full_body_forward_dynamics as fs,
)
from src.shared.python.motion_matching.full_body_spec import (
    validate_full_body_spec,
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
from src.shared.python.motion_matching.pipeline.constants import (
    BUILD_RECEIPT,
    CANDIDATE,
    CAPTURES,
    CONSISTENCY_PRIOR,
    LEG_SEEDS,
    RATE_HZ,
    REFERENCE_CUTOFF_HZ,
    SHOOTING_RELAXATION,
    SPEC,
    TRACKING_CUTOFF_HZ,
    UPPER_SPEC,
)
from src.shared.python.motion_matching.pipeline.dynamics import (
    DynamicsReportInputs,
    build_dynamics_report,
    replay,
    shooting_fit,
    zmp_filter,
)
from src.shared.python.motion_matching.pipeline.lane import (
    Lane,
    configure_lane,
    fitted_grip,
    wrist_bounds,
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


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ground support pipeline runner."""
    parser = argparse.ArgumentParser(
        description="Ground-supported full-body pipeline runner with pluggable plant engines."
    )
    parser.add_argument("--out", type=Path, default=Path.cwd())
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
        choices=["kkt", "mj-inverse"],
        default="kkt",
        help="computed-torque tracking backend (kkt or mj-inverse)",
    )
    return parser


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
        c3d_path=CAPTURES[args.capture],
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

    receipt_data = json.loads(BUILD_RECEIPT.read_text(encoding="utf-8"))
    pelvis_alignment = receipt_data.get("pelvis_alignment", {})
    alignment_old = (
        pelvis_alignment.get("hip_from_opensim_pelvis")
        if not args.skip_hip_calibration
        else None
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
        ),
    )
    hipcal_path.write_text(
        json.dumps(hip_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hip_bytes = hipcal_path.read_bytes()

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
    address = stage2.address
    address_report = stage2.address_report
    fixed = stage2.fixed
    seeds_all = stage2.seeds_all
    hip_spec = stage2.hip_spec

    offsets, calibration = lane.calibrate_legs(hip_bytes, fixed, seeds_all, address.q)
    scaled_spec, femur_scale, tibia_scale, scale_table = search_segment_scales(
        lane, hip_spec, fixed, offsets, address.q, log=log
    )
    unqualified = "unqualified" in str(base_spec.get("upper_body_qualification", ""))
    if not args.anthropometric and not unqualified:
        validate_full_body_spec(scaled_spec, upper_base)
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
    return _CalibrateAndScaleResult(
        scaled_spec=scaled_spec,
        spec_bytes=spec_bytes,
        address_report=address_report,
        adapter=adapter,
        kin=kin,
        sim=sim,
        address2=address2,
        attachments=attachments,
        offsets=offsets,
        calibration=calibration,
        calibration2=calibration2,
        hip_report=hip_report,
        qualification_note=qualification_note,
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
    q_smooth = smooth_reference(q_ik, RATE_HZ, REFERENCE_CUTOFF_HZ)
    audit_result = pink_service.audit_trajectory(q_smooth, solve_req, ik_opts)
    all_converged = bool(pink_result.passed)
    rate_limits_respected = bool(
        all(len(a.exceeded_joints) == 0 for a in audit_result.rate_audits)
    )
    max_ratio = float(
        max((a.max_velocity_ratio for a in audit_result.rate_audits), default=0.0)
    )
    is_qualified = bool(all_converged and audit_result.passed and rate_limits_respected)

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
    q_smooth = smooth_reference(q_ik, RATE_HZ, REFERENCE_CUTOFF_HZ)
    q_ref, ref_fits = consistency_resolve(
        lane, kin, q_smooth, prior_weight=CONSISTENCY_PRIOR, iterations=30
    )
    ref_errors = marker_errors(kin, q_ref, lane.points)
    return q_ik, q_ref, fits, ref_fits, errors, ref_errors, None


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
    q_track = smooth_reference(q_ref, RATE_HZ, TRACKING_CUTOFF_HZ)
    zmp = fs.reference_zmp(sim, lane.times, q_track, lane.ground)
    zmp_filter_report: dict[str, Any] | None = None
    if args.zmp_filter:
        q_track, zmp, zmp_filter_report = zmp_filter(lane, kin, sim, q_track, zmp, log)
    shooting_report: dict[str, Any] | None = None
    if args.shooting_fit > 0:
        q_track, zmp, shooting_report = shooting_fit(
            lane, kin, sim, q_track, q_ref, args.shooting_fit, log, args.shooting_gain
        )
    record, sim_q = replay(
        sim, lane, q_track, tracking_backend=getattr(args, "tracking", "kkt")
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
            shooting_report=shooting_report,
            tracking_backend=getattr(args, "tracking", "kkt"),
        )
    )
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
    )
    lookat = np.nanmean(lane.points[0], axis=0)
    names = tuple(kin.coordinate_order)
    render_playback(
        cal_res.spec_bytes, names, q_ref, lookat, out_dir / "ik_playback.gif"
    )
    render_playback(
        cal_res.spec_bytes, names, sim_q, lookat, out_dir / "tracking_playback.gif"
    )

    receipt = build_ground_support_receipt(
        GroundSupportReceiptInputs(
            backend=args.backend,
            ik_backend=getattr(args, "ik_backend", "lm"),
            tracking_backend=getattr(args, "tracking", "kkt"),
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
    (out_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, default=float) + "\n", encoding="utf-8"
    )
    log_pipeline_summary(
        log, receipt, ik_report, cal_res.calibration, cal_res.calibration2
    )
    return receipt


@precondition(lambda args: args is not None, "args must not be None")
def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full-body ground support matching pipeline."""
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
    if args.bound_wrists and args.free_wrists:
        raise ValueError("--bound-wrists and --free-wrists exclude each other")
    if args.bound_wrists or (fitted_grip(base_spec) and not args.free_wrists):
        lane.bounds |= wrist_bounds()
        ctx.log.info("wrists and forearms bounded to the human ranges in the IK")

    cal_res = _calibrate_and_scale(ctx, lane, base_spec, upper_base, upper, labels)

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
    run_pipeline(args)


if __name__ == "__main__":
    main()
