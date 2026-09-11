"""Exploratory first-prefix replay using the tested shared optimizer and native adapter."""

from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import logging
import hashlib
import subprocess
from pathlib import Path
import sys
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--run-dir", type=Path, required=True)
starts = parser.add_mutually_exclusive_group()
starts.add_argument("--checkpoint", type=Path)
starts.add_argument("--transfer-report", type=Path)
parser.add_argument(
    "--basis", default="constant", help="Bernstein degree name: constant through sextic"
)
parser.add_argument("--initial-state", type=Path, required=True)
parser.add_argument("--duration", type=float, default=0.1)
parser.add_argument("--max-nfev", type=int, default=10)
parser.add_argument("--finite-difference-step", type=float, default=0.001)
parser.add_argument(
    "--smoothness-weight",
    type=float,
    default=0.05,
    help="Curvature regularizer weight on Bernstein control points",
)
parser.add_argument(
    "--anatomical-weights",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use anatomical hierarchy weights for marker tracking",
)
parser.add_argument(
    "--terminal-weight",
    type=float,
    default=5.0,
    help="Weight penalty on terminal frame marker error",
)
parser.add_argument(
    "--time-weight-scale",
    type=float,
    default=4.0,
    help="Scale factor alpha for time weighting (1 + alpha*(t/T)^p)",
)
parser.add_argument(
    "--time-weight-power",
    type=float,
    default=2.0,
    help="Power p for time weighting (1 + alpha*(t/T)^p)",
)
parser.add_argument(
    "--pelvis-yaw-weight",
    type=float,
    default=25.0,
    help="Weight penalty on pelvis yaw error (enforcing < 5%% error)",
)
parser.add_argument(
    "--pelvis-yaw-max-error-pct",
    type=float,
    default=5.0,
    help="Maximum acceptable pelvis yaw error percentage (default 5.0%%)",
)
parser.add_argument(
    "--club-marker-weight",
    type=float,
    default=25.0,
    help="Weight penalty on clubhead markers (e.g. Marker_2:2:*, Marker_3:3:*, club)",
)
args = parser.parse_args()
if not np.isfinite(args.finite_difference_step) or args.finite_difference_step <= 0:
    parser.error("finite-difference-step must be finite and positive")
if not np.isfinite(args.club_marker_weight) or args.club_marker_weight <= 0:
    parser.error("club-marker-weight must be finite and positive")
if not np.isfinite(args.smoothness_weight) or args.smoothness_weight < 0:
    parser.error("smoothness-weight must be finite and non-negative")
if not np.isfinite(args.terminal_weight) or args.terminal_weight < 0:
    parser.error("terminal-weight must be finite and non-negative")
if not np.isfinite(args.pelvis_yaw_weight) or args.pelvis_yaw_weight < 0:
    parser.error("pelvis-yaw-weight must be finite and non-negative")
if not np.isfinite(args.pelvis_yaw_max_error_pct) or args.pelvis_yaw_max_error_pct <= 0:
    parser.error("pelvis-yaw-max-error-pct must be finite and positive")
sys.path.insert(0, str(args.repo))
from src.shared.python.motion_matching.prefix_fit import (
    MarkerTarget,
    PrefixFitOptions,
    PrefixStage,
    fit_prefixes,
    bernstein_to_simscape,
    build_anatomical_marker_weights,
    bernstein_curvature_regularizer,
)
from src.engines.Simscape_Multibody_Models.python.tour_checkpoints import (
    read_prefix_candidate,
    snapshot_prefix_report,
)

from src.engines.Simscape_Multibody_Models.python.tour_fit_state import (
    qualified_fit_identity,
    transfer_prefix_candidate,
    BASIS_CONTROLS,
    verify_native_initial_state,
)

root = args.run_dir
if (root / "first_prefix_fit.json").exists():
    raise FileExistsError("Use a fresh run directory; preserve earlier reports")
if args.max_nfev < 1:
    raise ValueError("max-nfev must be positive")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
capture = json.loads((root / "driver_marker_payload.json").read_text())
seed = json.loads(args.initial_state.read_text())
basis = f"{args.basis}-bernstein-6"
basis_duration = (
    float(capture["time_s"][-1]) if args.basis == "constant" else args.duration
)
identity = qualified_fit_identity(
    seed, capture["source_sha256"], args.duration, basis_duration, basis=basis
)
control_count = BASIS_CONTROLS[basis]
assignments = dict(zip(seed["labels"], seed["body_names"], strict=True))
labels = list(assignments)
indices = [capture["labels"].index(name) for name in labels]
points = np.asarray(capture["points_world_m"])[:, indices]
valid = np.asarray(capture["valid"])[:, indices]
points[~valid] = np.nan
assert np.isfinite(points[0]).all()
time_s = np.asarray(capture["time_s"])
mask = time_s <= args.duration + 1e-12
requested = time_s[mask]
observed = points[mask]
if len(requested) < 2 or abs(requested[-1] - args.duration) > 1e-10:
    raise ValueError("duration must select an exact capture sample after t0")
scales = np.array(
    [
        1500.0 if name.startswith("Translation") else 200.0
        for name in seed["coordinate_names"]
    ]
)
scales = np.repeat(scales, control_count)
report = {
    "status": "running",
    "qualification": "exploratory-first-prefix-only",
    "source_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True
    ).strip(),
    "source_patches": subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"], cwd=args.repo, text=True
    ).splitlines(),
    "source_patch_scope": "tracked differences from HEAD; copied runner identified separately by SHA256",
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "initial_state_sha256": hashlib.sha256(args.initial_state.read_bytes()).hexdigest(),
    "fit_identity": identity,
    "source_sha256": capture["source_sha256"],
    "duration_s": float(requested[-1]),
    "labels": labels,
    "body_assignments": assignments,
    "attachment_calibration": seed["qualification"],
    "initial_velocity": "qualified native tangent seed; SI qd held fixed",
    "parameter_mapping": f"effort=scale*(p-1), p in [0,2]; {control_count} Bernstein controls per joint in coordinate-major order; native A..G output",
    "effort_scales": scales.tolist(),
    "optimizer_options": {
        "finite_difference_step": args.finite_difference_step,
        "max_nfev": args.max_nfev,
    },
    "excluded_labels": [name for name in capture["labels"] if name not in labels],
    "evaluations": [],
}
initial_parameters = (
    np.asarray(read_prefix_candidate(args.checkpoint, report))
    if args.checkpoint
    else np.ones(len(scales))
)
if args.transfer_report:
    transfer_raw = args.transfer_report.read_bytes()
    initial_parameters = np.asarray(
        transfer_prefix_candidate(json.loads(transfer_raw), report)
    )
    report["candidate_transfer"] = {
        "source_report_sha256": hashlib.sha256(transfer_raw).hexdigest(),
        "source_path": str(args.transfer_report),
        "semantics": "torque candidate only; fresh objective and evaluation history",
    }
report["initial_parameters"] = initial_parameters.tolist()
import matlab
import matlab.engine

engine = matlab.engine.start_matlab("-nodesktop -nosplash")
try:
    assert engine.version("-release") == "2025b"
    report["matlab_root"] = engine.matlabroot()
    engine_dir = (
        args.repo / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab"
    )
    engine.addpath(str(engine_dir / "src/model"), nargout=0)
    engine.addpath(engine.genpath(str(engine_dir / "src/functions")), nargout=0)
    engine.addpath(str(engine_dir / "motion_matching/shared"), nargout=0)
    engine.workspace["fit_seed_path"] = str(args.initial_state)
    engine.workspace["fit_frame_names"] = list(assignments.values())
    engine.workspace["fit_duration"] = float(requested[-1])
    engine.eval(
        """
load_system('GolfSwing3D_Kinetic'); fit_priority_cleanup=configure_capture_velocity_targets();
fit_seed=jsondecode(fileread(fit_seed_path)); fit_ws=get_param('GolfSwing3D_Kinetic','ModelWorkspace');
fit_geometry_names={'UpperArmLength','LowerArmLength'};
for j=1:2; assignin(fit_ws,fit_geometry_names{j},fit_seed.geometry_in(j)); end
[fit_ks,fit_schema]=build_golf_kinematics();
assert(isequal(string(fit_seed.coordinate_names(:)),fit_schema.coordinate_names(:)),'Native joint order differs from qualified seed');
addTargetVariables(fit_ks,fit_schema.q_ids); addOutputVariables(fit_ks,fit_schema.frame_ids); addOutputVariables(fit_ks,fit_schema.rotation_ids);
fit_opts=capture_fit_sim_options(fit_duration);
fit_opts.sample_rate=360; fit_opts.fast_restart=true; fit_opts.retain_raw_output=true; fit_opts.verbosity='Silent'; fit_opts.joint_names=string(fit_seed.coordinate_names)';
for j=1:2; fit_opts.input_overrides.(fit_geometry_names{j})=fit_seed.geometry_in(j); end
for j=1:numel(fit_seed.q)
 name=fit_opts.joint_names(j); value=fit_seed.q(j); velocity=fit_seed.qd(j);
 if ~startsWith(name,'Translation'); value=rad2deg(value); velocity=rad2deg(velocity); end
 fit_opts.input_overrides.(replace(name,'Input','StartPosition'))=value;
 fit_opts.input_overrides.(replace(name,'Input','StartVelocity'))=velocity;
end
[found,fit_bodies]=ismember(string(fit_frame_names(:)),string({fit_schema.frames.name})); assert(all(found));
fit_offsets=fit_seed.offsets_m;
[fit_initial_values,fit_initial_flag,fit_initial_targets]=solve(fit_ks,fit_seed.q(:));
assert(fit_initial_flag==1 && all(fit_initial_targets),'Declared initial native pose must satisfy constraints');
fit_position_count=numel(fit_schema.frame_ids);
fit_origins=reshape(fit_initial_values(1:fit_position_count),3,[])';
fit_rotations=intrinsic_xyz_to_rotm(reshape(fit_initial_values(fit_position_count+1:end),3,[])');
fit_expected_initial=project_body_markers(fit_origins,fit_rotations,fit_bodies,fit_offsets);

""",
        nargout=0,
    )
    report["body_local_offsets_m"] = np.asarray(
        engine.workspace["fit_offsets"]
    ).tolist()

    def save_report() -> None:
        (root / "first_prefix_fit.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
        snapshot_prefix_report(
            root / "first_prefix_fit.json", root / "checkpoints/first-prefix"
        )

    def forward(parameters: np.ndarray, clock: np.ndarray) -> np.ndarray:
        effort = scales * (parameters - 1)
        controls = effort.reshape(len(seed["coordinate_names"]), control_count)
        theta = bernstein_to_simscape(controls, duration_s=basis_duration).ravel()
        engine.workspace["fit_theta"] = matlab.double(theta[:, None].tolist())
        engine.workspace["fit_time"] = matlab.double(clock[:, None].tolist())
        started = time.monotonic()
        engine.eval(
            "[fit_prediction,fit_last_replay]=simulate_golf_markers(fit_theta,fit_opts,fit_ks,fit_schema,fit_bodies,fit_offsets,fit_time);",
            nargout=0,
        )
        prediction = np.asarray(engine.workspace["fit_prediction"])
        assert (
            prediction.shape == (len(clock), len(labels), 3)
            and np.isfinite(prediction).all()
        )
        distance = np.linalg.norm(prediction - observed[: len(clock)], axis=2)
        report["evaluations"].append(
            {
                "number": len(report["evaluations"]) + 1,
                "elapsed_s": time.monotonic() - started,
                "rmse_m": float(np.sqrt(np.nanmean(distance**2))),
                "efforts": effort.tolist(),
            }
        )
        if len(report["evaluations"]) % 5 == 0:
            save_report()
            logger.info(
                "Evaluation %d: marker RMS %.9g m",
                report["evaluations"][-1]["number"],
                report["evaluations"][-1]["rmse_m"],
            )
        return prediction

    baseline = forward(np.ones(len(scales)), requested)
    report["baseline_prediction_m"] = baseline.tolist()
    initial_q = np.asarray(engine.eval("fit_last_replay.q(1,:);"))[0]
    initial_qd = np.asarray(engine.eval("fit_last_replay.qd(1,:);"))[0]
    report.update(
        verify_native_initial_state(
            seed,
            initial_q,
            initial_qd,
            baseline[0],
            np.asarray(engine.workspace["fit_expected_initial"]),
            observed[0],
        )
    )
    save_report()

    def checkpoint(stage: PrefixStage) -> None:
        report["stage"] = asdict(stage)
        report["stage"]["parameters"] = stage.parameters.tolist()
        save_report()

    custom_marker_weights = {}
    if args.club_marker_weight is not None:
        for lbl in labels:
            lower = lbl.lower()
            if "marker_2" in lower or "marker_3" in lower or "club" in lower:
                custom_marker_weights[lbl] = args.club_marker_weight

    marker_weights = (
        build_anatomical_marker_weights(labels, custom_weights=custom_marker_weights)
        if args.anatomical_weights
        else np.ones(len(labels))
    )
    regularizer = (
        bernstein_curvature_regularizer(
            control_count=control_count,
            weight=args.smoothness_weight,
            scales=scales,
        )
        if args.smoothness_weight > 0
        else None
    )
    wl_idx = labels.index("WaistLeft") if "WaistLeft" in labels else None
    wr_idx = labels.index("WaistRight") if "WaistRight" in labels else None
    pelvis_indices = (
        (wl_idx, wr_idx) if wl_idx is not None and wr_idx is not None else None
    )
    report["regularization"] = {
        "smoothness_weight": args.smoothness_weight,
        "anatomical_weights": args.anatomical_weights,
        "club_marker_weight": args.club_marker_weight,
        "marker_weights": marker_weights.tolist(),
        "terminal_weight": args.terminal_weight,
        "time_weight_scale": args.time_weight_scale,
        "time_weight_power": args.time_weight_power,
        "pelvis_yaw_weight": args.pelvis_yaw_weight,
        "pelvis_yaw_max_error_pct": args.pelvis_yaw_max_error_pct,
    }
    save_report()

    fit = fit_prefixes(
        MarkerTarget(requested, observed, marker_weights),
        forward,
        initial=initial_parameters,
        lower=np.zeros(len(scales)),
        upper=2 * np.ones(len(scales)),
        prefix_end_s=[float(requested[-1])],
        acceptance_rmse_m=0.025,
        options=PrefixFitOptions(
            max_nfev=args.max_nfev,
            finite_difference_step=args.finite_difference_step,
            checkpoint=checkpoint,
            regularization=regularizer,
            terminal_weight=args.terminal_weight,
            time_weight_scale=args.time_weight_scale,
            time_weight_power=args.time_weight_power,
            pelvis_indices=pelvis_indices,
            pelvis_yaw_weight=args.pelvis_yaw_weight,
            pelvis_yaw_max_error_pct=args.pelvis_yaw_max_error_pct,
            acceptance_terminal_rmse_m=0.035,
        ),
    )
    final_stage = fit.stages[-1]
    report["accepted_numerically"] = fit.accepted
    report["terminal_rmse_m"] = final_stage.terminal_rmse_m
    report["terminal_max_m"] = final_stage.terminal_max_m
    report["pelvis_yaw_diff_deg"] = final_stage.pelvis_yaw_diff_deg
    report["pelvis_yaw_error_pct"] = final_stage.pelvis_yaw_error_pct
    report["final_prediction_m"] = forward(fit.parameters, requested).tolist()
    report["status"] = "exploratory-fit-computed"
    engine.workspace["fit_replay_path"] = str(root / "final_native_replay.mat")
    engine.eval(
        "save(fit_replay_path,'fit_last_replay','fit_theta','fit_offsets','fit_seed','-v7.3');",
        nargout=0,
    )
    save_report()
except Exception as error:
    report["status"] = "failed"
    report["error"] = repr(error)
    (root / "first_prefix_fit.json").write_text(json.dumps(report, indent=2))
    raise
finally:
    engine.eval(
        "if exist('fit_priority_cleanup','var'); clear fit_priority_cleanup; end; if bdIsLoaded('GolfSwing3D_Kinetic'); set_param('GolfSwing3D_Kinetic','FastRestart','off'); close_system('GolfSwing3D_Kinetic',0); bdclose('all'); end",
        nargout=0,
    )
    engine.quit()
