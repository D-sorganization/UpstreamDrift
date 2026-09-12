"""Qualify an isolated namespace-only source bundle against a saved replay."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pinocchio
import scipy

from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "candidate", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "source_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source bundle changed: {name}")
    model = args.model.read_bytes()
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_text()),
        json.loads(model)["coordinate_order"],
        hashlib.sha256(model).hexdigest(),
    )
    reference = json.loads(args.reference.read_text())
    if candidate.document["coordinate_names"] != reference["coordinate_names"]:
        raise ValueError("Reference coordinate identity differs")
    clock = np.asarray(reference["time_s"])
    start = perf_counter()
    result = replay_candidate(
        model, candidate, clock, rtol=1e-11, atol=1e-13, max_step=0.00025
    )
    elapsed = perf_counter() - start
    n = len(candidate.document["coordinate_names"])
    q_error = float(np.max(np.abs(result.integration.state[:, :n] - reference["q"])))
    v_error = float(np.max(np.abs(result.integration.state[:, n:] - reference["qd"])))
    receipt = {
        "qualification": "packaged adapter versus qualified diagnostic; no C3D acceptance",
        "deployment": "namespace-only source bundle; full application imports not tested",
        "candidate_sha256": candidate.sha256,
        "source_manifest": manifest,
        "versions": {
            "python": platform.python_version(),
            "pinocchio": pinocchio.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "samples": len(clock),
        "duration_s": float(clock[-1]),
        "adapter_elapsed_s": elapsed,
        "integration_elapsed_s": result.integration.elapsed_s,
        "q_max_difference": q_error,
        "qd_max_difference": v_error,
        "closure_pose_max_abs": result.closure_pose_max_abs,
        "closure_velocity_max_abs": result.closure_velocity_max_abs,
        "input_sha256": {
            name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
            for name in ("model", "candidate", "reference")
        },
        "passed": q_error < 1e-7 and v_error < 1e-5,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["passed"]:
        raise AssertionError("Packaged replay differs from qualified diagnostic")


if __name__ == "__main__":
    main()
