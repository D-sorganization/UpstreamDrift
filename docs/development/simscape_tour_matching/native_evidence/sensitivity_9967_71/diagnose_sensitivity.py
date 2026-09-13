"""One fixed-candidate sensitivity check; no optimizer or numerical replacement."""

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_sensitivity import (
    replay_marker_sensitivities,
)
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "checkpoint", "target", "output", "runtime"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    raw = args.model.read_bytes()
    model_hash = hashlib.sha256(raw).hexdigest()
    checkpoint = json.loads(args.checkpoint.read_text())
    candidate = NativeReplayCandidate.from_document(
        checkpoint["candidate"], json.loads(raw)["coordinate_order"], model_hash
    )
    expected = "c8885ace0b1021dea6e2fbd824286fd37eeff8cb91498147d2d585cb64bed039"
    if candidate.sha256 != expected:
        raise ValueError("Checkpoint does not match authorized fixed candidate")
    payload = json.loads(args.target.read_text())
    if payload["source_sha256"] != candidate.document["capture_sha256"]:
        raise ValueError("Capture mismatch")
    clock = np.asarray(payload["time_s"])
    clock = clock[clock <= candidate.document["duration_s"]]
    kwargs = {
        "first_control": 4,
        "rtol": 3e-11,
        "atol": 3e-13,
        "max_step": 0.000125,
        "max_sensitivity_evaluations": 150000,
        "separate_error_control": True,
    }
    receipt = {
        "status": "running",
        "candidate_sha256": candidate.sha256,
        "arguments": kwargs,
        "clock_sha256": hashlib.sha256(clock.tobytes()).hexdigest(),
        "clock_samples": len(clock),
        "input_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.model, args.checkpoint, args.target, Path(__file__))
        },
        "source_hashes": {
            str(p.relative_to(args.runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.runtime.rglob("*.py")
        },
        "primal_policy": "Unchanged provider uses independent replay rtol=1e-11, atol=1e-13; supplied tolerance controls augmented solve. max_step applies to both.",
    }
    path = args.output / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    start = perf_counter()
    try:
        result = replay_marker_sensitivities(raw, candidate, clock, **kwargs)
        np.savez_compressed(
            args.output / "sensitivity.npz",
            time_s=clock,
            marker_jacobian=result.marker_jacobian,
            state_jacobian=result.state_jacobian,
            primal_markers_m=result.replay.markers_m,
        )
        receipt.update(
            status="passed",
            primal_marker_max_abs_difference_m=result.primal_marker_max_abs_difference_m,
            sensitivity_evaluations=result.sensitivity_evaluations,
            sensitivity_elapsed_s=result.sensitivity_elapsed_s,
        )
    except (ValueError, RuntimeError, FloatingPointError) as error:
        receipt.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        receipt["total_elapsed_s"] = perf_counter() - start
        path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
