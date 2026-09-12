"""Compare installed constrained-dynamics derivatives with central differences."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pinocchio as pin

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.estimation.residuals import finite_difference_jacobian
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate
from src.shared.python.motion_matching.native_effort_profile import NativeEffortProfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "candidate", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.model.read_bytes()
    spec = json.loads(raw)
    names = spec["coordinate_order"]
    n = len(names)
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_text()), names, hashlib.sha256(raw).hexdigest()
    )
    doc = candidate.document
    clock = np.array([0.0, 0.6, 0.8])
    trajectory = replay_candidate(
        raw, candidate, clock, rtol=1e-11, atol=1e-13, max_step=0.00025
    )
    engine = NativePinocchioModel(spec)
    root = next(joint for joint in spec["joints"] if joint["parent"] == "world")
    profile = NativeEffortProfile(
        names, doc["coefficients"], np.asarray(root["parent_to_base"])[:3, :3].T
    )

    def mapping(row: np.ndarray) -> dict:
        return dict(zip(names, map(float, row), strict=True))

    def forward(x: np.ndarray) -> np.ndarray:
        output = engine.accelerations(
            mapping(x[:n]), mapping(x[n : 2 * n]), mapping(x[2 * n :])
        )
        return np.array([output[name] for name in names])

    report = {
        "qualification": "local native derivative audit; no trajectory-sensitivity or optimizer qualification",
        "pinocchio_version": pin.__version__,
        "candidate_sha256": candidate.sha256,
        "input_sha256": {
            key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest()
            for key in ("model", "candidate")
        },
        "samples": [],
    }
    for time, state in zip(clock, trajectory.integration.state, strict=True):
        tau = profile.evaluate(float(time))
        x = np.r_[state, [tau[name] for name in names]]
        start = perf_counter()
        analytic = engine.acceleration_derivatives(
            mapping(x[:n]), mapping(x[n : 2 * n]), tau
        )
        analytic_s = perf_counter() - start
        blocks = [analytic.dq, analytic.dv, analytic.deffort]
        rows = []
        for step in (1e-4, 1e-5, 1e-6):
            start = perf_counter()
            numerical = finite_difference_jacobian(forward, x, step=step)
            elapsed = perf_counter() - start
            differences = []
            for i, block in enumerate(blocks):
                reference = numerical[:, i * n : (i + 1) * n]
                error = abs(block - reference)
                differences.append(
                    {
                        "max_abs": float(error.max()),
                        "max_scaled": float(
                            (error / np.maximum(1, abs(reference))).max()
                        ),
                    }
                )
            rows.append(
                {
                    "step": step,
                    "finite_difference_s": elapsed,
                    "q_v_effort_errors": differences,
                }
            )
        report["samples"].append(
            {"time_s": float(time), "analytic_s": analytic_s, "comparisons": rows}
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
