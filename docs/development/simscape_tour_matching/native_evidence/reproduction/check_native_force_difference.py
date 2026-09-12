"""Measure central native marker derivatives over physical force step sizes."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.native_candidate import (
    NativeReplayCandidate,
    increment_native_bernstein,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "candidate", "output"):
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
    duration = candidate.document["duration_s"]
    clock = np.array([0.0, 0.6, 0.7, duration])

    def forward(amplitude: float) -> np.ndarray:
        controls = np.zeros((len(spec["coordinate_order"]), 7))
        controls[1, 4] = amplitude
        trial = increment_native_bernstein(
            candidate, controls, basis_duration_s=duration
        )
        return replay_candidate(
            raw, trial, clock, rtol=1e-11, atol=1e-13, max_step=0.00025
        ).markers_m

    initial = forward(0.0)
    report = {
        "qualification": "local central force-difference audit; not a fit",
        "candidate_sha256": candidate.sha256,
        "input_sha256": {
            key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest()
            for key in ("model", "candidate")
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "coordinate": spec["coordinate_order"][1],
        "bernstein_control": 4,
        "time_s": clock.tolist(),
        "samples": [],
    }
    previous = None
    for step in (0.125, 0.0125, 0.00125, 0.000125, 0.0000125):
        started = perf_counter()
        positive, negative = forward(step), forward(-step)
        derivative = (positive - negative) / (2 * step)
        scale = max(float(np.linalg.norm(derivative)), 1e-12)
        report["samples"].append(
            {
                "force_step_N": step,
                "terminal_positive_delta_rms_m": float(
                    np.sqrt(np.mean(np.sum((positive[-1] - initial[-1]) ** 2, axis=1)))
                ),
                "central_derivative_norm_m_per_N": scale,
                "relative_change_from_previous": float(
                    np.linalg.norm(derivative - previous) / scale
                )
                if previous is not None
                else None,
                "relative_even_residual": float(
                    np.linalg.norm(positive + negative - 2 * initial)
                    / (2 * step * scale)
                ),
                "elapsed_s": perf_counter() - started,
                "derivative_m_per_N": derivative.tolist(),
            }
        )
        previous = derivative
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
