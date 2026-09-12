"""Qualify native marker derivatives against central tree-configuration changes."""

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
from src.shared.python.motion_matching.marker_projection import project_markers
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate


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
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_text()), names, hashlib.sha256(raw).hexdigest()
    )
    doc = candidate.document
    clock = np.array([0.0, 0.6, doc["duration_s"]])
    trajectory = replay_candidate(
        raw, candidate, clock, rtol=1e-11, atol=1e-13, max_step=0.00025
    )
    engine = NativePinocchioModel(spec)

    def mapping(values: np.ndarray) -> dict[str, float]:
        return dict(zip(names, map(float, values), strict=True))

    def forward(values: np.ndarray) -> np.ndarray:
        return project_markers(
            engine.frame_poses(mapping(values)),
            doc["marker_bodies"],
            doc["marker_offsets_m"],
        ).ravel()

    report = {
        "qualification": "native local marker derivative audit; no integrated sensitivity qualification",
        "pinocchio_version": pin.__version__,
        "candidate_sha256": candidate.sha256,
        "input_sha256": {
            key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest()
            for key in ("model", "candidate")
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "samples": [],
    }
    for time, state in zip(clock, trajectory.integration.state, strict=True):
        q = state[: len(names)]
        started = perf_counter()
        result = engine.marker_derivatives(
            mapping(q), doc["marker_bodies"], doc["marker_offsets_m"]
        )
        analytic_s = perf_counter() - started
        analytic = result.dposition_dq.reshape(-1, len(names))
        np.testing.assert_allclose(
            result.positions_m.ravel(), forward(q), rtol=0, atol=1e-14
        )
        row = {
            "time_s": float(time),
            "analytic_s": analytic_s,
            "shape": list(result.dposition_dq.shape),
            "steps": [],
        }
        for step in (1e-5, 1e-6, 1e-7):
            started = perf_counter()
            numeric = finite_difference_jacobian(forward, q, step=step)
            elapsed = perf_counter() - started
            np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)
            row["steps"].append(
                {
                    "step": step,
                    "numeric_s": elapsed,
                    "max_abs_error": float(np.max(np.abs(analytic - numeric))),
                }
            )
        report["samples"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
