"""Measure full native adapter evaluation against observed capture markers."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "candidate", "target", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.model.read_bytes()
    spec = json.loads(raw)
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_text()),
        spec["coordinate_order"],
        hashlib.sha256(raw).hexdigest(),
    )
    doc = candidate.document
    target = json.loads(args.target.read_text())
    if target["source_sha256"] != doc["capture_sha256"]:
        raise ValueError("Capture identity differs from candidate")
    indices = [target["labels"].index(label) for label in doc["marker_labels"]]
    time = np.asarray(target["time_s"])
    mask = time <= doc["duration_s"]
    time = time[mask]
    points = np.asarray(target["points_world_m"])[mask][:, indices]
    valid = np.asarray(target["valid"], dtype=bool)[mask][:, indices]
    valid &= np.isfinite(points).all(axis=2)
    if not valid.any() or time[-1] != doc["duration_s"]:
        raise ValueError("Insufficient target coverage")
    start = perf_counter()
    result = replay_candidate(
        raw, candidate, time, rtol=1e-11, atol=1e-13, max_step=0.00025
    )
    squared = np.sum((result.markers_m - points) ** 2, axis=2)
    elapsed = perf_counter() - start

    def rms(selected: np.ndarray) -> float:
        if not selected.any():
            raise ValueError("Requested metric has no observations")
        return float(np.sqrt(np.mean(squared[selected])))

    early = valid & (time[:, None] <= 0.6)
    terminal = np.zeros_like(valid)
    terminal[-1] = valid[-1]
    club = (
        terminal
        & np.asarray(
            [
                label.lower().startswith(("marker_2", "marker_3"))
                for label in doc["marker_labels"]
            ]
        )[None, :]
    )
    receipt = {
        "qualification": "observed-marker baseline only; no optimizer run or fit acceptance",
        "candidate_sha256": candidate.sha256,
        "duration_s": float(time[-1]),
        "modeled_marker_count": len(indices),
        "capture_marker_count": len(target["labels"]),
        "observed_marker_samples": int(valid.sum()),
        "sample_count": len(time),
        "whole_rms_m": rms(valid),
        "early_rms_m": rms(early),
        "terminal_rms_m": rms(terminal),
        "terminal_club_cluster_rms_m": rms(club),
        "adapter_and_residual_elapsed_s": elapsed,
        "integration_elapsed_s": result.integration.elapsed_s,
        "closure_pose_max_abs": result.closure_pose_max_abs,
        "closure_velocity_max_abs": result.closure_velocity_max_abs,
        "input_sha256": {
            name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
            for name in ("model", "candidate", "target")
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
