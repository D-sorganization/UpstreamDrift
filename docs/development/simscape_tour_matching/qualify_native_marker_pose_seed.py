"""Fit one observed C3D frame as a native closure-constrained marker pose."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_constrained_pose import (
    NativeConstrainedPoseOracle,
)
from src.engines.physics_engines.pinocchio.python.native_model import NativePinocchioModel
from src.shared.python.motion_matching.constrained_marker_pose import fit_marker_pose
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--bound", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not np.isfinite(args.bound) or args.bound <= 0:
        raise ValueError("Output must be new and coordinate bound must be positive")
    raw = args.model.read_bytes()
    spec = json.loads(raw)
    names = tuple(spec["coordinate_order"])
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_text()), names, hashlib.sha256(raw).hexdigest()
    )
    payload = json.loads(args.payload.read_text())
    frame = args.frame
    labels = tuple(candidate.document["marker_labels"])
    indices = [payload["labels"].index(label) for label in labels]
    points = np.asarray(payload["points_world_m"], dtype=float)
    valid = np.asarray(payload["valid"], dtype=bool)
    if frame < 0 or frame >= points.shape[0]:
        raise ValueError("Frame is outside target payload")
    target = points[frame, indices]
    observed = valid[frame, indices]
    model = NativePinocchioModel(spec)
    oracle = NativeConstrainedPoseOracle(
        model,
        names,
        candidate.document["marker_bodies"],
        np.asarray(candidate.document["marker_offsets_m"], dtype=float),
    )
    initial = np.asarray(candidate.document["q0"], dtype=float)
    result = fit_marker_pose(
        initial,
        initial - args.bound,
        initial + args.bound,
        target,
        observed,
        oracle.forward,
        oracle.closure,
        max_iterations=100,
    )
    output = {
        "model_sha256": hashlib.sha256(raw).hexdigest(),
        "candidate_sha256": candidate.sha256,
        "capture_sha256": payload["source_sha256"],
        "frame": frame,
        "time_s": payload["time_s"][frame],
        "observed_markers": int(observed.sum()),
        "coordinate_bound": args.bound,
        "marker_rms_m": result.marker_rms_m,
        "closure_max_abs": result.closure_max_abs,
        "closure_satisfied": result.closure_satisfied,
        "optimizer_converged": result.optimizer_converged,
        "message": result.message,
        "iterations": result.iterations,
        "coordinates": result.coordinates.tolist(),
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
