"""Probe local static reachability of observed markers with native grip closure."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.constrained_marker_pose import fit_marker_pose
from src.shared.python.motion_matching.marker_projection import project_markers
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate


def pose_schedule(
    capture_time: list[float], requested: list[float], duration: float, direction: str
) -> np.ndarray:
    """Return exact capture indices; never replace a missing sample by a neighbor."""
    clock, times = np.asarray(capture_time), np.asarray(requested)
    if (
        direction not in ("independent", "forward", "backward")
        or clock.ndim != 1
        or times.ndim != 1
        or not times.size
        or not np.isfinite(clock).all()
        or not np.isfinite(times).all()
        or np.any(np.diff(clock) <= 0)
        or np.any(np.diff(times) <= 0)
        or times[0] <= 0
        or times[-1] != duration
    ):
        raise ValueError("Invalid capture clock or pose schedule")
    indices = []
    for time in times:
        matches = np.flatnonzero(np.isclose(clock, time, atol=1e-12, rtol=0))
        if len(matches) != 1:
            raise ValueError("Exact target sample is missing or ambiguous")
        indices.append(int(matches[0]))
    return np.asarray(indices[::-1] if direction == "backward" else indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "candidate", "target", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--translation-radius", type=float, default=0.2)
    parser.add_argument("--rotation-radius", type=float, default=0.5)
    parser.add_argument("--seed-report", type=Path)
    parser.add_argument("--times", type=float, nargs="+", default=[0.6, 0.7, 0.8])
    parser.add_argument(
        "--direction",
        choices=["independent", "forward", "backward"],
        default="independent",
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.model.read_bytes()
    spec = json.loads(raw)
    names = spec["coordinate_order"]
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_text()), names, hashlib.sha256(raw).hexdigest()
    )
    doc = candidate.document
    seeds = None
    if args.seed_report is not None:
        prior = json.loads(args.seed_report.read_text())
        if prior["candidate_sha256"] != candidate.sha256:
            raise ValueError("Seed report has a different candidate")
        seeds = {
            row["time_s"]: np.asarray(row["coordinates"]) for row in prior["poses"]
        }
    payload = json.loads(args.target.read_text())
    if payload["source_sha256"] != doc["capture_sha256"]:
        raise ValueError("Capture mismatch")
    schedule = pose_schedule(
        payload["time_s"], args.times, doc["duration_s"], args.direction
    )
    clock = np.r_[0.0, np.asarray(payload["time_s"])[np.sort(schedule)]]
    replay = replay_candidate(
        raw, candidate, clock, rtol=1e-11, atol=1e-13, max_step=0.00025
    )
    engine = NativePinocchioModel(spec)
    indices = [payload["labels"].index(label) for label in doc["marker_labels"]]
    zero = dict.fromkeys(names, 0.0)

    def mapping(q: np.ndarray) -> dict:
        return dict(zip(names, map(float, q), strict=True))

    def forward(q: np.ndarray) -> np.ndarray:
        return project_markers(
            engine.frame_poses(mapping(q)),
            doc["marker_bodies"],
            doc["marker_offsets_m"],
        )

    def closure(q: np.ndarray) -> np.ndarray:
        # Public adapter contract: acceleration refreshes the native closure data.
        # Zero rates/efforts here serve static pose diagnosis only.
        engine.accelerations(mapping(q), zero, zero)
        return engine.closure_errors()[0]

    report = {
        "qualification": "local static pose diagnosis; not forward fit or global error floor",
        "candidate_sha256": candidate.sha256,
        "input_sha256": {
            k: hashlib.sha256(getattr(args, k).read_bytes()).hexdigest()
            for k in ("model", "candidate", "target")
        },
        "local_translation_bound_m": args.translation_radius,
        "local_rotation_bound_rad": args.rotation_radius,
        "seed_report_sha256": hashlib.sha256(args.seed_report.read_bytes()).hexdigest()
        if args.seed_report
        else None,
        "poses": [],
        "direction": args.direction,
        "requested_times_s": args.times,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    previous = None
    for k in schedule:
        time = payload["time_s"][k]
        i = int(np.flatnonzero(clock == time)[0])
        target = np.asarray(payload["points_world_m"])[k, indices]
        valid = np.asarray(payload["valid"], dtype=bool)[k, indices]
        initial = replay.integration.state[i, : len(names)]
        radius = np.full(len(names), args.rotation_radius)
        radius[:3] = args.translation_radius
        start = initial
        if previous is not None and args.direction != "independent":
            start = previous
        elif seeds is not None:
            start = seeds[float(time)]
        fitted = fit_marker_pose(
            start,
            initial - radius,
            initial + radius,
            target,
            valid,
            forward,
            closure,
            max_iterations=args.max_iterations,
        )
        row = asdict(fitted)
        row["coordinates"] = fitted.coordinates.tolist()
        row["time_s"] = float(time)
        row["near_bound_coordinates"] = [
            name
            for name, delta, limit in zip(
                names, abs(fitted.coordinates - initial), radius, strict=True
            )
            if limit - delta < 1e-5
        ]
        row["marker_labels"] = doc["marker_labels"]
        row["marker_errors_m"] = np.where(
            valid, np.linalg.norm(forward(fitted.coordinates) - target, axis=1), np.nan
        ).tolist()
        row["initial_marker_rms_m"] = float(
            np.sqrt(
                np.mean(np.sum((forward(initial)[valid] - target[valid]) ** 2, axis=1))
            )
        )
        report["poses"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if args.direction != "independent":
            if not fitted.closure_satisfied or not fitted.optimizer_converged:
                raise ValueError(
                    "Continuation stopped at an unqualified pose; partial receipt saved"
                )
            previous = fitted.coordinates


if __name__ == "__main__":
    main()
