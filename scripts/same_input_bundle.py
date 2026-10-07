"""Export and replay same-input bundles (#11607, epic #11605).

    MUJOCO_GL=egl PYTHONPATH=.:src python3 -m scripts.same_input_bundle export \\
        --run-dir RUN --out bundle.npz [--duration 0.85]
    MUJOCO_GL=egl PYTHONPATH=.:src python3 -m scripts.same_input_bundle replay \\
        --bundle bundle.npz --engine drake --out receipt.json [--segment-ms 50]

    MUJOCO_GL=egl PYTHONPATH=.:src python3 -m scripts.same_input_bundle closed-loop \\
        --run-dir RUN --bundle bundle.npz --engine drake --out receipt.json

``export`` reads a pipeline run directory (``full_body_spec_hipcal_scaled.json``
and the ``q_track``/``track_time_s`` arrays of ``dynamics_record.npz``) and
writes the MuJoCo reference bundle.  ``replay`` integrates the bundle efforts
open loop in one engine and writes the full-horizon and segmented scores.
``closed-loop`` tracks the run's reference in one engine with the identical
controller function and scores it against the bundle's MuJoCo reference.
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
    ALL_ENGINES,
    InputBundle,
    VectorPlant,
    closed_loop,
    generate_reference_bundle,
    open_loop,
    score_replay,
    segmented_replay,
)

LOG = logging.getLogger("same_input_bundle")
_STATES: dict[str, np.ndarray] = {}  # last rollout, written beside the receipt


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _track(run_dir: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    record_path = run_dir / "dynamics_record.npz"
    with np.load(record_path) as record:
        if "q_track" not in record:
            raise ValueError(f"{record_path} predates q_track; rerun the pipeline")
        return record_path, record["track_time_s"], record["q_track"]


def export(run_dir: Path, out: Path, duration_s: float | None) -> dict:
    """Build the MuJoCo reference bundle of a pipeline run directory."""
    spec_path = run_dir / "full_body_spec_hipcal_scaled.json"
    record_path, times, q_track = _track(run_dir)
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
    rollout = open_loop(
        plant,
        bundle.q0,
        bundle.v0,
        bundle.efforts,
        dt_s=bundle.dt_s,
        stop_on_failure=True,
    )
    elapsed = time.perf_counter() - started
    score = score_replay(plant, bundle, rollout)
    segments = segmented_replay(plant, bundle, segment_steps=segment_steps)
    checkpoints = {
        f"{t:.2f}": float(score.coordinate_error[int(round(t / bundle.dt_s))])
        for t in np.arange(0.05, bundle.steps * bundle.dt_s + 1e-9, 0.05)
        if int(round(t / bundle.dt_s)) < score.coordinate_error.size
    }
    _STATES["q"], _STATES["time_s"] = rollout.q, rollout.time_s
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


def closed_loop_receipt(run_dir: Path, bundle_path: Path, engine: str) -> dict:
    """Score a closed-loop run in ``engine`` against the bundle reference."""
    bundle = InputBundle.load(bundle_path)
    _, times, q_track = _track(run_dir)
    started = time.perf_counter()
    rollout = closed_loop(
        engine,
        bundle.spec_bytes,
        times,
        q_track,
        duration_s=bundle.steps * bundle.dt_s,
    )
    elapsed = time.perf_counter() - started
    plant = VectorPlant(engine, bundle.spec_bytes)
    score = score_replay(plant, bundle, rollout)
    _STATES["q"], _STATES["time_s"] = rollout.q, rollout.time_s
    return {
        "schema": "same-input-closed-loop/v1",
        "bundle_sha256": _sha256(bundle_path),
        "bundle": bundle.manifest(),
        "engine": engine,
        "elapsed_s": elapsed,
        "full_horizon": score.summary(),
        "max_effort_difference_nm": float(
            np.abs(rollout.efforts - bundle.efforts).max()
        ),
        "max_pose_drift_m": float(rollout.pose_drift.max()),
    }


def build_parser() -> argparse.ArgumentParser:
    """Return the ``export`` / ``replay`` / ``closed-loop`` argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--run-dir", type=Path, required=True)
    exp.add_argument("--out", type=Path, required=True)
    exp.add_argument("--duration", type=float, default=None)
    rep = sub.add_parser("replay")
    rep.add_argument("--bundle", type=Path, required=True)
    rep.add_argument("--engine", choices=ALL_ENGINES, required=True)
    rep.add_argument("--out", type=Path, required=True)
    rep.add_argument("--segment-ms", type=float, default=50.0)
    clo = sub.add_parser("closed-loop")
    clo.add_argument("--run-dir", type=Path, required=True)
    clo.add_argument("--bundle", type=Path, required=True)
    clo.add_argument("--engine", choices=ALL_ENGINES, required=True)
    clo.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    if args.command == "export":
        manifest = export(args.run_dir, args.out, args.duration)
        LOG.info("wrote %s: %d steps", args.out, manifest["steps"])
        return
    if args.command == "closed-loop":
        receipt = closed_loop_receipt(args.run_dir, args.bundle, args.engine)
    else:
        receipt = replay(args.bundle, args.engine, args.segment_ms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.out.with_suffix(".npz"), **_STATES)
    LOG.info("%s: %s", args.engine, json.dumps(receipt["full_horizon"]))
    if "worst_segment" in receipt:
        LOG.info("worst %g ms segment: %s", args.segment_ms, receipt["worst_segment"])


if __name__ == "__main__":
    main()
