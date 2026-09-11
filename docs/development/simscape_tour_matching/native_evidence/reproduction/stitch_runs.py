"""CLI tool to stitch backswing and downswing polynomial fits into a composite trajectory (#9921)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import scipy.io as sio

parser = argparse.ArgumentParser(
    description="Stitch backswing and downswing fits into composite swing trajectory"
)
parser.add_argument("--repo", type=Path, required=True, help="Repository root path")
parser.add_argument(
    "--backswing-run",
    type=Path,
    required=True,
    help="Qualified backswing run directory",
)
parser.add_argument(
    "--downswing-run",
    type=Path,
    required=True,
    help="Qualified downswing run directory",
)
parser.add_argument(
    "--output-dir", type=Path, required=True, help="Output directory for stitched swing"
)
parser.add_argument(
    "--global-degree",
    type=int,
    default=6,
    help="Degree of single global polynomial approximation",
)
args = parser.parse_args()

sys.path.insert(0, str(args.repo))
from src.shared.python.motion_matching.piecewise_polynomial import (
    PolynomialSegment,
    PiecewisePolynomialTorque,
    stitch_two_phase_trajectories,
)

out_dir = args.output_dir
out_dir.mkdir(parents=True, exist_ok=True)

import h5py

# 1. Load backswing and downswing data
back_fit = json.loads((args.backswing_run / "first_prefix_fit.json").read_text())
down_fit = json.loads((args.downswing_run / "downswing_fit.json").read_text())

with h5py.File(str(args.backswing_run / "final_native_replay.mat"), "r") as f_back:
    theta_back = np.asarray(f_back["fit_theta"]).reshape(-1, 7)

with h5py.File(str(args.downswing_run / "final_native_replay.mat"), "r") as f_down:
    theta_down = np.asarray(f_down["fit_theta"]).reshape(-1, 7)

t_top = float(back_fit["duration_s"])
t_final = float(down_fit["total_swing_duration_s"])
dur_down = float(down_fit["downswing_duration_s"])

# Extract Bernstein control points (efforts)
controls_back = np.asarray(back_fit["evaluations"][-1]["efforts"]).reshape(-1, 7)
controls_down = np.asarray(down_fit["evaluations"][-1]["efforts"]).reshape(-1, 7)

# Build segments in Bernstein basis
seg_back = PolynomialSegment(0.0, t_top, controls_back, is_bernstein=True)
seg_down = PolynomialSegment(t_top, t_final, controls_down, is_bernstein=True)

# Stitch segments
stitched = stitch_two_phase_trajectories(
    seg_back, seg_down, enforce_c0=True, enforce_c1=False
)
continuity = stitched.check_continuity(tol=1e-4)

# Global polynomial fit
global_coeffs = stitched.project_to_global_polynomial(
    degree=args.global_degree, pin_endpoints=True
)

from src.shared.python.motion_matching.prefix_fit import normalized_to_simscape

global_theta = normalized_to_simscape(global_coeffs, duration_s=t_final).ravel()

report: dict[str, Any] = {
    "status": "stitched-composite-ready",
    "backswing_run": str(args.backswing_run),
    "downswing_run": str(args.downswing_run),
    "t_top_s": t_top,
    "t_final_s": t_final,
    "n_channels": stitched.n_channels,
    "continuity": continuity,
    "global_degree": args.global_degree,
    "global_coeffs_ascending_power": global_coeffs.tolist(),
    "global_theta": global_theta.tolist(),
}

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

(out_dir / "stitched_swing.json").write_text(json.dumps(report, indent=2))
sio.savemat(
    str(out_dir / "stitched_swing.mat"),
    {
        "stitched_report": report,
        "global_coeffs": global_coeffs,
        "global_theta": global_theta,
        "theta_back": theta_back,
        "theta_down": theta_down,
        "controls_back": controls_back,
        "controls_down": controls_down,
        "t_top": t_top,
        "t_final": t_final,
    },
)
logger.info(
    "Stitched composite written to %s: C0 continuous=%s (max jump: %.4e Nm)",
    out_dir,
    continuity["c0_continuous"],
    continuity["max_c0_jump"],
)
