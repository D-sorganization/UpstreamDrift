"""Build auditable, closure-constrained static native pose seeds by continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_constrained_pose import (
    NativeConstrainedPoseOracle,
)
from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.shared.python.motion_matching.constrained_marker_pose import fit_marker_pose
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate
from src.shared.python.motion_matching.gimbal_branch import gimbal_branch_interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--bound", type=float, required=True)
    parser.add_argument("--use-derivatives", action="store_true")
    parser.add_argument("--gimbal-margin", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not np.isfinite(args.bound) or args.bound <= 0:
        raise ValueError("Output must be new and coordinate bound must be positive")
    if args.frames != sorted(set(args.frames)):
        raise ValueError("Frames must be unique and strictly ascending")
    raw = args.model.read_bytes()
    specification = json.loads(raw)
    names = tuple(specification["coordinate_order"])
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_text()), names, hashlib.sha256(raw).hexdigest()
    )
    payload = json.loads(args.payload.read_text())
    labels = tuple(candidate.document["marker_labels"])
    indices = [payload["labels"].index(label) for label in labels]
    points = np.asarray(payload["points_world_m"], dtype=float)
    valid = np.asarray(payload["valid"], dtype=bool)
    if args.frames[0] < 0 or args.frames[-1] >= points.shape[0]:
        raise ValueError("Frame is outside target payload")
    model = NativePinocchioModel(specification)
    oracle = NativeConstrainedPoseOracle(
        model,
        names,
        candidate.document["marker_bodies"],
        np.asarray(candidate.document["marker_offsets_m"], dtype=float),
    )

    def marker_jacobian(q: np.ndarray) -> np.ndarray:
        return model.marker_derivatives(
            dict(zip(names, q.tolist(), strict=True)),
            candidate.document["marker_bodies"],
            np.asarray(candidate.document["marker_offsets_m"], dtype=float),
        ).dposition_dq

    def closure_jacobian(q: np.ndarray) -> np.ndarray:
        return model.closure_position_linearization(
            dict(zip(names, q.tolist(), strict=True))
        ).jacobian

    current = np.asarray(candidate.document["q0"], dtype=float)
    branch_bounds = {}
    if args.gimbal_margin is not None:
        for joint in specification["joints"]:
            rotations = [
                p for p in joint["primitives"] if p["primitive"] in ("Rx", "Ry", "Rz")
            ]
            if len(rotations) == 3 and len({p["primitive"] for p in rotations}) == 3:
                name = rotations[1]["coordinate"]
                branch_bounds[name] = gimbal_branch_interval(
                    current[names.index(name)], args.gimbal_margin
                )
    records = []
    for frame in args.frames:
        observed = valid[frame, indices]
        lower, upper = current - args.bound, current + args.bound
        for name, (lo, hi) in branch_bounds.items():
            index = names.index(name)
            lower[index], upper[index] = max(lower[index], lo), min(upper[index], hi)
        result = fit_marker_pose(
            current,
            lower,
            upper,
            points[frame, indices],
            observed,
            oracle.forward,
            oracle.closure,
            forward_jacobian=marker_jacobian if args.use_derivatives else None,
            closure_jacobian=closure_jacobian if args.use_derivatives else None,
            max_iterations=100,
        )
        records.append(
            {
                "frame": frame,
                "time_s": payload["time_s"][frame],
                "observed_markers": int(observed.sum()),
                "marker_rms_m": result.marker_rms_m,
                "closure_max_abs": result.closure_max_abs,
                "closure_satisfied": result.closure_satisfied,
                "optimizer_converged": result.optimizer_converged,
                "message": result.message,
                "iterations": result.iterations,
                "coordinates": result.coordinates.tolist(),
            }
        )
        if np.any(result.coordinates < lower - 1e-8) or np.any(
            result.coordinates > upper + 1e-8
        ):
            raise ValueError("Returned pose violates numerical path-search bounds")
        if not result.closure_satisfied:
            raise ValueError("Continuation produced a closure-invalid static pose")
        current = result.coordinates
    args.output.write_text(
        json.dumps(
            {
                "model_sha256": hashlib.sha256(raw).hexdigest(),
                "candidate_sha256": candidate.sha256,
                "capture_sha256": payload["source_sha256"],
                "coordinate_bound": args.bound,
                "supplied_derivatives": args.use_derivatives,
                "numerical_gimbal_branch_bounds": branch_bounds,
                "gimbal_margin_rad": args.gimbal_margin,
                "runner_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "records": records,
                "scope": "Static closure-constrained poses only; not a smooth trajectory or dynamic fit.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
