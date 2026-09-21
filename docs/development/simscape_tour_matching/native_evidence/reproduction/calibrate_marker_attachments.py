"""Reproducible calibration and decomposition script for marker attachments under held-out validation.

Performs fixed 3D marker attachment calibration on the training interval
(t in [0.0, 0.85] s, frames 0..306 @ 360 Hz) and evaluates generalization
performance on the held-out validation interval (t in (0.85, 1.814] s,
frames 307..653) on unchanged model topology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import tarfile

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, default=Path("."), help="Repository root path")
parser.add_argument(
    "--output",
    type=Path,
    default=Path(
        "docs/development/simscape_tour_matching/native_evidence/calibration_training_validation_receipt.json"
    ),
    help="Output receipt JSON path",
)
args = parser.parse_args()

sys.path.insert(0, str(args.repo))
from src.shared.python.body_part_viz.fitters._kabsch import kabsch_rotation
from src.shared.python.motion_matching.rigidity import rigid_attachment_residuals

base = args.repo / "docs/development/simscape_tour_matching/native_evidence"
candidate_path = base / "two_window_fit_9967_102/returned-candidate.json"
candidate_raw = candidate_path.read_bytes()
cand = json.loads(candidate_raw)

with tarfile.open(base / "run102_original_inputs.tar.gz") as archive:
    capture_raw = archive.extractfile("driver_marker_payload_9967.json").read()
cap = json.loads(capture_raw)

assert cand["capture_sha256"] == cap["source_sha256"]

labels = cand["marker_labels"]
bodies = np.array(cand["marker_bodies"])
orig_offsets = np.array(cand["marker_offsets_m"])

idx = [cap["labels"].index(label) for label in labels]
points = np.array(cap["points_world_m"], dtype=float)[:, idx]
valid = np.array(cap["valid"], dtype=bool)[:, idx]
points[~valid] = np.nan

clock = np.array(cap["time_s"])
assert len(clock) == 654

train_frames = (0, 307)  # 0..306 (t <= 0.85 s)
val_frames = (307, 654)  # 307..653 (t > 0.85 s)


def calc_rms(err_arr: NDArray[np.float64]) -> float:
    finite_vals = err_arr[np.isfinite(err_arr)]
    if not len(finite_vals):
        return 0.0
    return float(np.sqrt(np.mean(finite_vals**2)) * 1000.0)


def per_body_rms(err_arr: NDArray[np.float64]) -> dict[str, float]:
    out: dict[str, float] = {}
    for b in np.unique(bodies):
        b_mask = bodies == b
        out[str(b)] = calc_rms(err_arr[:, b_mask])
    return out


def per_marker_rms(err_arr: NDArray[np.float64]) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, label in enumerate(labels):
        out[str(label)] = calc_rms(err_arr[:, i : i + 1])
    return out


# Baseline errors with original offsets
orig_errors = rigid_attachment_residuals(orig_offsets, points, bodies)

# Alternating calibration on training frames
fitted_offsets = orig_offsets.copy()
for _iteration in range(10):
    new_offsets = fitted_offsets.copy()
    for b in np.unique(bodies):
        b_idx = np.flatnonzero(bodies == b)
        if len(b_idx) < 3:
            continue
        b_locals: list[list[NDArray[np.float64]]] = [[] for _ in range(len(b_idx))]
        for f in range(train_frames[0], train_frames[1]):
            obs = points[f, b_idx]
            obs_valid = ~np.isnan(obs).any(axis=1)
            if np.count_nonzero(obs_valid) < 3:
                continue
            cur_b_pts = fitted_offsets[b_idx[obs_valid]]
            pc = cur_b_pts.mean(axis=0)
            qc = obs[obs_valid].mean(axis=0)
            rot = kabsch_rotation(cur_b_pts - pc, obs[obs_valid] - qc)
            trans = qc - rot @ pc
            for k in range(len(b_idx)):
                if obs_valid[k]:
                    p_body = (obs[k] - trans) @ rot
                    b_locals[k].append(p_body)
        for k, marker_i in enumerate(b_idx):
            if b_locals[k]:
                new_offsets[marker_i] = np.mean(b_locals[k], axis=0)
    fitted_offsets = new_offsets

fitted_errors = rigid_attachment_residuals(fitted_offsets, points, bodies)

receipt = {
    "qualification": "fixed-offset calibration on training frames (0..306, t <= 0.85s) evaluated under held-out validation (307..653, t > 0.85s) on unchanged model topology",
    "inputs": {
        "candidate_sha256": hashlib.sha256(candidate_raw).hexdigest(),
        "payload_sha256": hashlib.sha256(capture_raw).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    },
    "training_interval": {
        "start_frame": train_frames[0],
        "end_frame": train_frames[1] - 1,
        "start_time_s": float(clock[train_frames[0]]),
        "end_time_s": float(clock[train_frames[1] - 1]),
        "frame_count": train_frames[1] - train_frames[0],
    },
    "validation_interval": {
        "start_frame": val_frames[0],
        "end_frame": val_frames[1] - 1,
        "start_time_s": float(clock[val_frames[0]]),
        "end_time_s": float(clock[val_frames[1] - 1]),
        "frame_count": val_frames[1] - val_frames[0],
    },
    "full_interval": {
        "start_frame": 0,
        "end_frame": len(clock) - 1,
        "start_time_s": float(clock[0]),
        "end_time_s": float(clock[-1]),
        "frame_count": len(clock),
    },
    "metrics_baseline": {
        "training_aggregate_rms_mm": calc_rms(
            orig_errors[train_frames[0] : train_frames[1]]
        ),
        "validation_aggregate_rms_mm": calc_rms(
            orig_errors[val_frames[0] : val_frames[1]]
        ),
        "full_aggregate_rms_mm": calc_rms(orig_errors),
        "training_body_rms_mm": per_body_rms(
            orig_errors[train_frames[0] : train_frames[1]]
        ),
        "validation_body_rms_mm": per_body_rms(
            orig_errors[val_frames[0] : val_frames[1]]
        ),
        "full_body_rms_mm": per_body_rms(orig_errors),
        "training_marker_rms_mm": per_marker_rms(
            orig_errors[train_frames[0] : train_frames[1]]
        ),
        "validation_marker_rms_mm": per_marker_rms(
            orig_errors[val_frames[0] : val_frames[1]]
        ),
        "full_marker_rms_mm": per_marker_rms(orig_errors),
    },
    "metrics_calibrated": {
        "training_aggregate_rms_mm": calc_rms(
            fitted_errors[train_frames[0] : train_frames[1]]
        ),
        "validation_aggregate_rms_mm": calc_rms(
            fitted_errors[val_frames[0] : val_frames[1]]
        ),
        "full_aggregate_rms_mm": calc_rms(fitted_errors),
        "training_body_rms_mm": per_body_rms(
            fitted_errors[train_frames[0] : train_frames[1]]
        ),
        "validation_body_rms_mm": per_body_rms(
            fitted_errors[val_frames[0] : val_frames[1]]
        ),
        "full_body_rms_mm": per_body_rms(fitted_errors),
        "training_marker_rms_mm": per_marker_rms(
            fitted_errors[train_frames[0] : train_frames[1]]
        ),
        "validation_marker_rms_mm": per_marker_rms(
            fitted_errors[val_frames[0] : val_frames[1]]
        ),
        "full_marker_rms_mm": per_marker_rms(fitted_errors),
    },
    "marker_labels": labels,
    "marker_bodies": bodies.tolist(),
    "original_offsets_m": orig_offsets.tolist(),
    "fitted_offsets_m": fitted_offsets.tolist(),
    "offset_deltas_mm": ((fitted_offsets - orig_offsets) * 1000.0).tolist(),
}

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
logger.info("Calibration receipt successfully written to %s", args.output)
logger.info(
    "  Training Aggregate RMS: %.2f mm -> %.2f mm",
    receipt["metrics_baseline"]["training_aggregate_rms_mm"],
    receipt["metrics_calibrated"]["training_aggregate_rms_mm"],
)
logger.info(
    "  Validation Aggregate RMS: %.2f mm -> %.2f mm",
    receipt["metrics_baseline"]["validation_aggregate_rms_mm"],
    receipt["metrics_calibrated"]["validation_aggregate_rms_mm"],
)
logger.info(
    "  Hub Validation RMS: %.2f mm -> %.2f mm",
    receipt["metrics_baseline"]["validation_body_rms_mm"]["Hub"],
    receipt["metrics_calibrated"]["validation_body_rms_mm"]["Hub"],
)
