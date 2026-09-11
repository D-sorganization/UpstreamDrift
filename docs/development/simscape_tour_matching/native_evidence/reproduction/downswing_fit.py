"""Downswing phase optimizer starting from qualified top-of-backswing state (#9921).

Matches the downswing-to-finish segment [t_top, t_final] using the qualified terminal
state S_top = (q(t_top), qd(t_top)) and enforces C^0 torque continuity with the backswing:
c_{0, down} = tau_{back}(t_top).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

parser = argparse.ArgumentParser(
    description="Fit downswing phase starting from qualified backswing"
)
parser.add_argument("--repo", type=Path, required=True, help="Repository root path")
parser.add_argument(
    "--run-dir", type=Path, required=True, help="Output directory for downswing fit"
)
parser.add_argument(
    "--backswing-run",
    type=Path,
    required=True,
    help="Directory containing qualified backswing candidate (first_prefix_fit.json and final_native_replay.mat)",
)
parser.add_argument(
    "--duration",
    type=float,
    default=1.814,
    help="End time of downswing in physical capture seconds",
)
parser.add_argument(
    "--basis", default="sextic", help="Bernstein basis: constant through sextic"
)
parser.add_argument(
    "--max-nfev", type=int, default=15, help="Maximum optimizer iterations"
)
parser.add_argument(
    "--finite-difference-step",
    type=float,
    default=0.00001,
    help="Relative finite difference step",
)
parser.add_argument(
    "--smoothness-weight",
    type=float,
    default=0.08,
    help="Curvature regularizer weight on Bernstein control points",
)
parser.add_argument(
    "--anatomical-weights",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use anatomical hierarchy weights for marker tracking",
)
parser.add_argument(
    "--enforce-c0",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enforce C^0 torque continuity at transition from backswing",
)
parser.add_argument(
    "--transfer-report",
    type=Path,
    default=None,
    help="Previous shorter downswing candidate report to warm start from",
)
args = parser.parse_args()

if not np.isfinite(args.finite_difference_step) or args.finite_difference_step <= 0:
    parser.error("finite-difference-step must be finite and positive")
if not np.isfinite(args.smoothness_weight) or args.smoothness_weight < 0:
    parser.error("smoothness-weight must be finite and non-negative")

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
    snapshot_prefix_report,
)
from src.engines.Simscape_Multibody_Models.python.tour_fit_state import (
    BASIS_CONTROLS,
)

root = args.run_dir
root.mkdir(parents=True, exist_ok=True)
if (root / "downswing_fit.json").exists():
    raise FileExistsError("Use a fresh run directory; preserve earlier reports")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Load backswing report and qualified replay
back_report = json.loads((args.backswing_run / "first_prefix_fit.json").read_text())
t_top = float(back_report["duration_s"])
if args.duration <= t_top:
    raise ValueError(
        f"Downswing duration ({args.duration}s) must be greater than t_top ({t_top}s)"
    )

# 2. Load driver capture payload
capture = json.loads((root / "driver_marker_payload.json").read_text())
time_s = np.asarray(capture["time_s"])

# Slicing for downswing: t in [t_top, args.duration]
mask = (time_s >= t_top - 1e-12) & (time_s <= args.duration + 1e-12)
requested_global = time_s[mask]
if len(requested_global) < 2:
    raise ValueError("Downswing window needs at least two capture samples")

# Downswing local clock starting at 0
requested_local = requested_global - requested_global[0]
down_duration = float(requested_local[-1])

seed = back_report["fit_identity"]
assignments = back_report["body_assignments"]
labels = back_report["labels"]
indices = [capture["labels"].index(name) for name in labels]
points = np.asarray(capture["points_world_m"])[:, indices]
valid = np.asarray(capture["valid"])[:, indices]
points[~valid] = np.nan
observed = points[mask]

basis = f"{args.basis}-bernstein-6"
control_count = BASIS_CONTROLS[basis]
n_coords = len(seed["coordinate_names"])

# Effort scales
scales = np.array(
    [
        1500.0 if name.startswith("Translation") else 200.0
        for name in seed["coordinate_names"]
    ]
)
scales = np.repeat(scales, control_count)

report: dict[str, Any] = {
    "status": "running",
    "qualification": "exploratory-downswing-fit",
    "source_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True
    ).strip(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "backswing_run": str(args.backswing_run),
    "t_top_s": t_top,
    "downswing_duration_s": down_duration,
    "total_swing_duration_s": float(requested_global[-1]),
    "labels": labels,
    "body_assignments": assignments,
    "effort_scales": scales.tolist(),
    "optimizer_options": {
        "finite_difference_step": args.finite_difference_step,
        "max_nfev": args.max_nfev,
    },
    "evaluations": [],
}

import h5py

# Load final native replay from backswing to get qualified terminal state q(t_top), qd(t_top)
with h5py.File(str(args.backswing_run / "final_native_replay.mat"), "r") as h5_file:
    # In HDF5, arrays are stored (27, N)
    q_top = np.asarray(h5_file["fit_last_replay"]["q"])[:, -1]
    qd_top = np.asarray(h5_file["fit_last_replay"]["qd"])[:, -1]
    theta_back = np.asarray(h5_file["fit_theta"]).ravel()

# Compute terminal backswing torque
# theta_back is [A..G] native coefficients per joint
theta_matrix = theta_back.reshape(n_coords, 7)
tau_top = np.array([np.polyval(row, t_top) for row in theta_matrix])
report["tau_top_Nm"] = tau_top.tolist()
report["q_top"] = q_top.tolist()
report["qd_top"] = qd_top.tolist()

# Start MATLAB engine
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

    engine.workspace["fit_frame_names"] = list(assignments.values())
    engine.workspace["fit_duration"] = down_duration
    engine.workspace["q_top"] = matlab.double(q_top.tolist())
    engine.workspace["qd_top"] = matlab.double(qd_top.tolist())
    engine.workspace["geom"] = matlab.double(seed["geometry_in"])
    engine.workspace["coord_names"] = list(seed["coordinate_names"])
    engine.workspace["fit_offsets"] = matlab.double(seed["offsets_m"])
    engine.workspace["fit_seed_path"] = str(
        args.backswing_run / "first_prefix_fit.json"
    )

    engine.eval(
        """
load_system('GolfSwing3D_Kinetic'); fit_priority_cleanup=configure_capture_velocity_targets();
fit_seed=jsondecode(fileread(fit_seed_path)); fit_ws=get_param('GolfSwing3D_Kinetic','ModelWorkspace');
fit_geometry_names={'UpperArmLength','LowerArmLength'};
for j=1:2; assignin(fit_ws,fit_geometry_names{j},geom(j)); end
[fit_ks,fit_schema]=build_golf_kinematics();
addTargetVariables(fit_ks,fit_schema.q_ids); addOutputVariables(fit_ks,fit_schema.frame_ids); addOutputVariables(fit_ks,fit_schema.rotation_ids);
fit_opts=capture_fit_sim_options(fit_duration + 0.010);
fit_opts.sample_rate=360; fit_opts.fast_restart=true; fit_opts.retain_raw_output=true; fit_opts.verbosity='Silent';
fit_opts.joint_names=string(coord_names)';
for j=1:2; fit_opts.input_overrides.(fit_geometry_names{j})=geom(j); end
for j=1:numel(q_top)
 name=fit_opts.joint_names(j); value=q_top(j); velocity=qd_top(j);
 if ~startsWith(name,'Translation'); value=rad2deg(value); velocity=rad2deg(velocity); end
 fit_opts.input_overrides.(replace(name,'Input','StartPosition'))=value;
 fit_opts.input_overrides.(replace(name,'Input','StartVelocity'))=velocity;
end
[found,fit_bodies]=ismember(string(fit_frame_names(:)),string({fit_schema.frames.name})); assert(all(found));
""",
        nargout=0,
    )

    # Initial parameter vector
    initial_parameters = np.ones(len(scales))
    lower_bounds = np.zeros(len(scales))
    upper_bounds = 2.0 * np.ones(len(scales))

    if args.transfer_report is not None and args.transfer_report.exists():
        transfer_source = json.loads(args.transfer_report.read_text())
        old_params = np.asarray(
            transfer_source.get("final_parameters", []), dtype=float
        )
        old_dur = float(transfer_source.get("downswing_duration_s", 0.0))
        if len(old_params) == len(scales) and old_dur > 0 and down_duration >= old_dur:
            ratio = down_duration / old_dur
            values = old_params.reshape(n_coords, control_count)
            result = np.empty_like(values)
            for level in range(control_count):
                result[:, level] = values[:, 0]
                values = (1 - ratio) * values[:, :-1] + ratio * values[:, 1:]
            initial_parameters = np.clip(result.ravel(), 0.0, 2.0)
            logger.info(
                "Transferred warm-start candidate from %s (ratio %.4f)",
                args.transfer_report,
                ratio,
            )
            report["transferred_from"] = str(args.transfer_report)

    # If enforcing C0 continuity:
    # effort = scales * (p - 1) => p = 1 + effort / scale
    # For k=0 (first control point of each coordinate):
    if args.enforce_c0:
        for j in range(n_coords):
            p0 = 1.0 + tau_top[j] / scales[j * control_count]
            p0_clamped = float(np.clip(p0, 0.0, 2.0))
            idx = j * control_count
            initial_parameters[idx] = p0_clamped
            # Lock or tightly bound first control point
            lower_bounds[idx] = max(0.0, p0_clamped - 1e-6)
            upper_bounds[idx] = min(2.0, p0_clamped + 1e-6)

    def save_report() -> None:
        (root / "downswing_fit.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )

    def forward(parameters: np.ndarray, clock: np.ndarray) -> np.ndarray:
        effort = scales * (parameters - 1)
        controls = effort.reshape(n_coords, control_count)
        theta = bernstein_to_simscape(controls, duration_s=down_duration).ravel()
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

    marker_weights = (
        build_anatomical_marker_weights(labels)
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

    fit = fit_prefixes(
        MarkerTarget(requested_local, observed, marker_weights),
        forward,
        initial=initial_parameters,
        lower=lower_bounds,
        upper=upper_bounds,
        prefix_end_s=[down_duration],
        acceptance_rmse_m=0.005,
        options=PrefixFitOptions(
            max_nfev=args.max_nfev,
            finite_difference_step=args.finite_difference_step,
            regularization=regularizer,
        ),
    )

    report["accepted_numerically"] = fit.accepted
    report["final_prediction_m"] = forward(fit.parameters, requested_local).tolist()
    final_effort = scales * (fit.parameters - 1)
    report["final_effort"] = final_effort.tolist()
    report["final_parameters"] = fit.parameters.tolist()
    report["final_rmse_m"] = report["evaluations"][-1]["rmse_m"]
    report["status"] = "downswing-fit-computed"
    engine.workspace["fit_replay_path"] = str(root / "final_native_replay.mat")
    engine.eval(
        "save(fit_replay_path,'fit_last_replay','fit_theta','fit_offsets','fit_seed','-v7.3');",
        nargout=0,
    )
    save_report()

except Exception as error:
    report["status"] = "failed"
    report["error"] = repr(error)
    (root / "downswing_fit.json").write_text(json.dumps(report, indent=2))
    raise
finally:
    engine.eval(
        "if exist('fit_priority_cleanup','var'); clear fit_priority_cleanup; end; if bdIsLoaded('GolfSwing3D_Kinetic'); set_param('GolfSwing3D_Kinetic','FastRestart','off'); close_system('GolfSwing3D_Kinetic',0); bdclose('all'); end",
        nargout=0,
    )
    engine.quit()
