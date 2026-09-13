"""Save one returned-candidate replay, actual native state and marker evidence."""

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
    for name in ("model", "candidate", "target", "baseline_samples", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    raw = args.model.read_bytes()
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_text()),
        json.loads(raw)["coordinate_order"],
        hashlib.sha256(raw).hexdigest(),
    )
    doc = candidate.document
    payload = json.loads(args.target.read_text())
    if payload["source_sha256"] != doc["capture_sha256"]:
        raise ValueError("Capture mismatch")
    indices = [payload["labels"].index(label) for label in doc["marker_labels"]]
    clock = np.asarray(payload["time_s"])
    mask = clock <= doc["duration_s"]
    clock = clock[mask]
    points = np.asarray(payload["points_world_m"])[mask][:, indices].copy()
    valid = np.asarray(payload["valid"], dtype=bool)[mask][:, indices]
    valid &= np.isfinite(points).all(axis=2)
    points[~valid] = np.nan
    baseline = np.load(args.baseline_samples)
    np.testing.assert_array_equal(baseline["time_s"], clock)
    np.testing.assert_array_equal(baseline["labels"], doc["marker_labels"])
    np.testing.assert_allclose(baseline["target_m"], points, rtol=0, atol=0)
    start = perf_counter()
    result = replay_candidate(
        raw, candidate, clock, rtol=1e-11, atol=1e-13, max_step=0.0000625
    )
    np.savez_compressed(
        args.output / "sampled-markers-state.npz",
        time_s=clock,
        target_m=points,
        valid=valid,
        baseline_m=baseline["baseline_m"],
        returned_m=result.markers_m,
        labels=np.asarray(doc["marker_labels"]),
        coordinate_names=np.asarray(doc["coordinate_names"]),
        native_state=result.integration.state,
    )
    error = np.sum((result.markers_m - points) ** 2, axis=2)
    receipt = {
        "status": "completed",
        "candidate_sha256": candidate.sha256,
        "qualification": "Exploratory/rejected 0–0.85 s prefix; no R2025b acceptance",
        "whole_rms_m": float(np.sqrt(np.mean(error[valid]))),
        "terminal_rms_m": float(np.sqrt(np.mean(error[-1, valid[-1]]))),
        "integration_elapsed_s": result.integration.elapsed_s,
        "integration_evaluations": result.integration.evaluations,
        "elapsed_s": perf_counter() - start,
        "state_shape": result.integration.state.shape,
        "state_layout": "Each time row: native q then native qd in coordinate_names order; no qdd saved",
        "replay_settings": {"rtol": 1e-11, "atol": 1e-13, "max_step": 0.0000625},
        "baseline_note": "Reuse measured run19 baseline from run62 visual, which used max_step .00025; no second baseline integration",
        "input_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                args.model,
                args.candidate,
                args.target,
                args.baseline_samples,
                Path(__file__),
            )
        },
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
