"""Check reusable native sensitivity adapter against the audited full block."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_sensitivity import (
    replay_marker_sensitivities,
)
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "model",
        "candidate",
        "reference-report",
        "reference-jacobian",
        "output",
    ):
        parser.add_argument(f"--{key}", type=Path, required=True)
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
    receipt = json.loads(args.reference_report.read_text())
    if (
        not receipt["accepted_audit"]
        or receipt["candidate_sha256"] != candidate.sha256
        or receipt["jacobian_artifact_sha256"]
        != hashlib.sha256(args.reference_jacobian.read_bytes()).hexdigest()
    ):
        raise ValueError("Sensitivity reference identity or qualification differs")
    with np.load(args.reference_jacobian) as data:
        time = data["time"]
        expected = data["marker_jacobian"]
    started = perf_counter()
    result = replay_marker_sensitivities(raw, candidate, time)
    elapsed = perf_counter() - started
    relative = np.sqrt(
        np.sum((result.marker_jacobian - expected) ** 2, axis=(0, 1, 2))
    ) / np.maximum(np.sqrt(np.sum(expected**2, axis=(0, 1, 2))), 1e-12)
    accepted = bool(np.max(relative) < 1e-3)
    report = {
        "qualification": "reusable sensitivity adapter versus audited block; no optimizer acceptance",
        "candidate_sha256": candidate.sha256,
        "accepted_audit": accepted,
        "input_sha256": {
            key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest()
            for key in ("model", "candidate", "reference_report", "reference_jacobian")
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "total_elapsed_s": elapsed,
        "sensitivity_elapsed_s": result.sensitivity_elapsed_s,
        "max_relative_column_difference": float(np.max(relative)),
        "primal_marker_max_abs_difference_m": result.primal_marker_max_abs_difference_m,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not accepted:
        raise ValueError("Reusable native sensitivity adapter audit failed")


if __name__ == "__main__":
    main()
