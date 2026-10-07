"""Export and replay same-input bundles (#11607, epic #11605).

    MUJOCO_GL=egl PYTHONPATH=.:src python3 -m scripts.same_input_bundle export \\
        --run-dir RUN --out bundle.npz [--duration 0.85]
    MUJOCO_GL=egl PYTHONPATH=.:src python3 -m scripts.same_input_bundle replay \\
        --bundle bundle.npz --engine drake --out receipt.json [--segment-ms 50]

``export`` reads a pipeline run directory (``full_body_spec_hipcal_scaled.json``
and the ``q_track``/``track_time_s`` arrays of ``dynamics_record.npz``) and
writes the MuJoCo reference bundle.  ``replay`` integrates the bundle efforts
open loop in one engine and writes the full-horizon and segmented scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from src.shared.python.motion_matching.same_input import (
    PARITY_ENGINES,
    InputBundle,
    VectorPlant,
    generate_reference_bundle,
    open_loop,
    score_replay,
    segmented_replay,
)

LOG = logging.getLogger("same_input_bundle")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(run_dir: Path, out: Path, duration_s: float | None) -> dict:
    """Build the MuJoCo reference bundle of a pipeline run directory."""
    spec_path = run_dir / "full_body_spec_hipcal_scaled.json"
    record_path = run_dir / "dynamics_record.npz"
    with np.load(record_path) as record:
        if "q_track" not in record:
            raise ValueError(f"{record_path} predates q_track; rerun the pipeline")
        times, q_track = record["track_time_s"], record["q_track"]
    receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    bundle = generate_reference_bundle(
        spec_path.read_bytes(),
        times,
        q_track,
        duration_s=duration_s,
        provenance={
            "run_dir": run_dir.name,
            "capture": receipt.get("capture"),
            "capture_sha256": receipt.get("capture_sha256"),
            "spec_file": spec_path.name,
            "dynamics_record_sha256": _sha256(record_path),
        },
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    bundle.save(out)
    return bundle.manifest()


def replay(bundle_path: Path, engine: str, segment_ms: float) -> dict:
    """Score an open-loop replay of ``bundle_path`` in ``engine``."""
    bundle = InputBundle.load(bundle_path)
    segment_steps = max(1, int(round(segment_ms * 1e-3 / bundle.dt_s)))
    plant = VectorPlant(engine, bundle.spec_bytes)
    started = time.perf_counter()
    rollout = open_loop(plant, bundle.q0, bundle.v0, bundle.efforts, dt_s=bundle.dt_s)
    elapsed = time.perf_counter() - started
    score = score_replay(plant, bundle, rollout)
    segments = segmented_replay(plant, bundle, segment_steps=segment_steps)
    checkpoints = {
        f"{t:.2f}": float(score.coordinate_error[int(round(t / bundle.dt_s))])
        for t in np.arange(0.05, bundle.steps * bundle.dt_s + 1e-9, 0.05)
    }
    return {
        "schema": "same-input-replay/v1",
        "bundle_sha256": _sha256(bundle_path),
        "bundle": bundle.manifest(),
        "engine": engine,
        "elapsed_s": elapsed,
        "full_horizon": score.summary(),
        "coordinate_error_at_s": checkpoints,
        "max_pose_drift_m": float(rollout.pose_drift.max()),
        "segment_steps": segment_steps,
        "segments": segments,
        "worst_segment": {
            "coordinate_error_rad": max(s["coordinate_error_rad"] for s in segments),
            "frame_error_m": max(s["frame_error_m"] for s in segments),
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--run-dir", type=Path, required=True)
    exp.add_argument("--out", type=Path, required=True)
    exp.add_argument("--duration", type=float, default=None)
    rep = sub.add_parser("replay")
    rep.add_argument("--bundle", type=Path, required=True)
    rep.add_argument("--engine", choices=PARITY_ENGINES, required=True)
    rep.add_argument("--out", type=Path, required=True)
    rep.add_argument("--segment-ms", type=float, default=50.0)
    args = parser.parse_args()
    if args.command == "export":
        manifest = export(args.run_dir, args.out, args.duration)
        LOG.info("wrote %s: %d steps", args.out, manifest["steps"])
        return
    receipt = replay(args.bundle, args.engine, args.segment_ms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    LOG.info("%s: %s", args.engine, json.dumps(receipt["full_horizon"]))
    LOG.info("worst %g ms segment: %s", args.segment_ms, receipt["worst_segment"])


if __name__ == "__main__":
    main()
