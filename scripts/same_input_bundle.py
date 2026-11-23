"""Export and replay same-input bundles (#11607, epic #11605).

    MUJOCO_GL=egl PYTHONPATH=.:src python3 -m scripts.same_input_bundle export \\
        --run-dir RUN --out bundle.npz [--duration 0.85] [--effort-hold-ms 0.125]
    MUJOCO_GL=egl PYTHONPATH=.:src python3 -m scripts.same_input_bundle replay \\
        --bundle bundle.npz --engine drake --out receipt.json [--segment-ms 50]

    MUJOCO_GL=egl PYTHONPATH=.:src python3 -m scripts.same_input_bundle closed-loop \\
        --run-dir RUN --bundle bundle.npz --engine drake --out receipt.json

``export --shaft-model chain`` replaces the rigid club by the R6 chain shaft
(#12686) and appends an undeflected shaft to the tracked reference; chain
bundles replay only in engines that pass the chain parity gate
(``CHAIN_ENGINES``, #12691), the others refuse them.

``export`` reads a pipeline run directory (``full_body_spec_hipcal_scaled.json``
and the ``q_track``/``track_time_s`` arrays of ``dynamics_record.npz``) and
writes the MuJoCo reference bundle (``--effort-hold-ms`` sets the opt-in
controller hold, #12279; default 1 ms).  ``replay`` integrates the bundle efforts
open loop in one engine and writes the full-horizon and segmented scores.
``closed-loop`` tracks the run's reference in one engine with the identical
controller function and scores it against the bundle's MuJoCo reference.
It also reports per-coordinate error against the actual controller position
target. The tracking record must match the bundle's recorded digest; this
binds the selected file but does not independently validate its provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from src.shared.python.motion_matching.club_shaft_chain import (
    DEFAULT_SHAFT_MODEL,
    SHAFT_MODELS,
    add_shaft_arguments,
    checked_shaft_options,
    passive_coordinates,
    shaft_options,
)
from src.shared.python.motion_matching.execution.spec_builder import with_shaft_model
from src.shared.python.motion_matching.impact_force import ImpactForce
from src.shared.python.motion_matching.pipeline.constants import DT_S
from src.shared.python.motion_matching.same_input import (
    ALL_ENGINES,
    InputBundle,
    VectorPlant,
    closed_loop,
    generate_reference_bundle,
    hold_policy,
    open_loop,
    score_replay,
    segmented_replay,
)
from src.shared.python.motion_matching.same_input.reference import drop_lead_in

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


def target_tracking_summary(
    track_time: np.ndarray,
    q_track: np.ndarray,
    rollout_time: np.ndarray,
    rollout_q: np.ndarray,
    *,
    max_endpoint_extension_s: float = 0.0,
) -> dict:
    """Compare sampled states with the actual linear controller position target.

    Coordinates retain their native units; no mixed-unit aggregate is formed.
    A short permitted final extension uses the controller's endpoint clamp.
    This measures sampled tracking error, not physiological feasibility.
    """
    return _target_summary(
        track_time,
        q_track,
        rollout_time,
        rollout_q,
        max_endpoint_extension_s=max_endpoint_extension_s,
    )


def target_velocity_tracking_summary(
    track_time: np.ndarray,
    q_track: np.ndarray,
    rollout_time: np.ndarray,
    rollout_v: np.ndarray,
    *,
    split_time_s: float | None = None,
    max_endpoint_extension_s: float = 0.0,
) -> dict:
    """Compare velocity with rates derived by the source controller.

    The rates are finite differences, optionally split at impact; they are not
    independent measurements or derivatives of linear position interpolation.
    """
    return _target_summary(
        track_time,
        q_track,
        rollout_time,
        rollout_v,
        max_endpoint_extension_s=max_endpoint_extension_s,
        velocity=True,
        split_time_s=split_time_s,
    )


def _target_summary(
    track_time: np.ndarray,
    q_track: np.ndarray,
    rollout_time: np.ndarray,
    rollout_q: np.ndarray,
    *,
    max_endpoint_extension_s: float,
    velocity: bool = False,
    split_time_s: float | None = None,
) -> dict:
    from src.shared.python.motion_matching.impact_force import (
        reference_rates,
        sample_rate_table,
    )

    source_time, target = np.asarray(track_time, float), np.asarray(q_track, float)
    time_s, q = np.asarray(rollout_time, float), np.asarray(rollout_q, float)
    if (
        source_time.ndim != 1
        or source_time.size < 2
        or target.ndim != 2
        or target.shape[0] != source_time.size
        or time_s.ndim != 1
        or not time_s.size
        or q.shape != (time_s.size, target.shape[1])
        or not target.shape[1]
        or not all(np.isfinite(a).all() for a in (source_time, target, time_s, q))
        or np.any(np.diff(source_time) <= 0)
        or np.any(np.diff(time_s) <= 0)
    ):
        raise ValueError(
            "target and rollout must have finite, aligned, increasing samples"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        source_time = source_time - source_time[0]
    if not np.isfinite(source_time).all():
        raise ValueError("target time normalization overflowed")
    if not np.isfinite(max_endpoint_extension_s) or max_endpoint_extension_s < 0:
        raise ValueError("endpoint extension must be finite and nonnegative")
    if time_s[0] < 0 or time_s[-1] - source_time[-1] > max_endpoint_extension_s:
        raise ValueError("rollout times must lie within the target span")
    split = None
    if velocity:
        target, _, last = reference_rates(source_time, target, split_time_s)
        if last is not None:
            if split_time_s is None:
                raise ValueError("split rate table requires split time")
            split = (float(split_time_s), last)
    sampled = np.array(
        [sample_rate_table(source_time, target, t, split) for t in time_s]
    )
    with np.errstate(over="ignore", invalid="ignore"):
        error = q - sampled
    if not np.isfinite(error).all():
        raise ValueError("target tracking difference overflowed")
    magnitude = np.max(np.abs(error), axis=0)
    scale = np.where(magnitude == 0.0, 1.0, magnitude)
    return {
        "reference": (
            "controller_derived_velocity_target"
            if velocity
            else "controller_position_target"
        ),
        "endpoint_clamped_duration_s": max(0.0, float(time_s[-1] - source_time[-1])),
        "rms_by_coordinate": (
            scale * np.sqrt(np.mean((error / scale) ** 2, axis=0))
        ).tolist(),
        "max_abs_by_coordinate": magnitude.tolist(),
    }


def padded_track(spec_bytes: bytes, q_track: np.ndarray) -> np.ndarray:
    """``q_track`` in ``spec_bytes``' coordinates: a chain shaft starts straight.

    The pipeline tracks the rigid-club coordinates; a chain spec appends its
    passive shaft coordinates (#12686), whose reference is zero deflection.
    A track that already has every coordinate is returned unchanged.

    Raises:
        ValueError: ``q_track`` has neither the full nor the rigid width.
    """
    spec = json.loads(spec_bytes)
    track = np.asarray(q_track, dtype=float)
    nv = len(spec["coordinate_order"])
    shaft = len(passive_coordinates(spec))
    if track.ndim == 2 and track.shape[1] == nv:
        return track
    if track.ndim != 2 or track.shape[1] != nv - shaft:
        raise ValueError(
            f"q_track has {track.shape[-1]} columns; the spec needs {nv} "
            f"({nv - shaft} tracked plus {shaft} chain-shaft coordinates)"
        )
    return np.hstack([track, np.zeros((track.shape[0], shaft))])


def shaft_inputs(
    spec_bytes: bytes,
    q_track: np.ndarray,
    shaft_model: str,
    options: dict | None = None,
) -> tuple[bytes, np.ndarray]:
    """Spec and track of a ``shaft_model`` bundle (#12691).

    ``"rigid"`` returns both inputs unchanged; ``"chain"`` applies the R6
    chain of the spec's club, with the flex, stiffness scale and link
    ``options`` (#15150; empty is the default chain), and pads the track with
    a straight shaft. Options with a rigid club raise ``ValueError``.
    """
    checked_shaft_options(shaft_model, options)
    if shaft_model == DEFAULT_SHAFT_MODEL:
        return spec_bytes, q_track
    document = json.loads(spec_bytes)
    chain = with_shaft_model(document, document["club"]["name"], shaft_model, options)
    chain_bytes = json.dumps(chain).encode()
    return chain_bytes, padded_track(chain_bytes, q_track)


def _impact_plan(
    receipt: dict, times: np.ndarray
) -> tuple[ImpactForce | None, float | None]:
    """The run's ball impact plan and reference impact time (GCV-20), on the
    bundle clock; the plan is re-armed so the bundle run finds its own contact."""
    record = receipt.get("dynamics", {}).get("ball_impact")
    split = receipt.get("ik", {}).get("impact_split", {}).get("impact_time_s")
    offset = -float(times[0])
    plan = None
    if record is not None:
        plan = ImpactForce.from_record(record).rearmed().shifted(offset)
    return plan, None if split is None else float(split) + offset


def export(
    run_dir: Path,
    out: Path,
    duration_s: float | None,
    effort_hold_ms: float = 1.0,
    shaft_model: str = DEFAULT_SHAFT_MODEL,
    options: dict | None = None,
) -> dict:
    """Build the MuJoCo reference bundle of a pipeline run directory.

    ``effort_hold_ms`` is the controller hold recorded in the manifest
    (#12279); it must split the 1 ms step evenly into at most eight holds.
    ``shaft_model`` selects the rigid club (default) or the chain shaft, and
    the chain ``options`` (#15150) are recorded in the provenance when given.
    """
    spec_path = run_dir / "full_body_spec_hipcal_scaled.json"
    record_path, times, q_track = _track(run_dir)
    spec_bytes, q_track = shaft_inputs(
        spec_path.read_bytes(), q_track, shaft_model, options
    )
    receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    impact, split = _impact_plan(receipt, times)
    bundle = generate_reference_bundle(
        spec_bytes,
        times,
        q_track,
        duration_s=duration_s,
        impact=impact,
        split_time_s=split,
        policy=hold_policy(effort_hold_ms * 1e-3),
        provenance={
            "run_dir": run_dir.name,
            "capture": receipt.get("capture"),
            "capture_sha256": receipt.get("capture_sha256"),
            "spec_file": spec_path.name,
            "shaft_model": shaft_model,
            **({"shaft_options": dict(options)} if options else {}),
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
        bundle.replay_efforts(),
        dt_s=bundle.dt_s,
        policy=bundle.step_policy(stop_on_failure=True),
        impact=bundle.ball_impact(),
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
    record_path, times, q_track = _track(run_dir)
    if _sha256(record_path) != bundle.provenance.get("dynamics_record_sha256"):
        raise ValueError("tracking record does not match bundle provenance")
    run_spec = (run_dir / "full_body_spec_hipcal_scaled.json").read_bytes()
    expected_spec, target = shaft_inputs(
        run_spec,
        q_track,
        bundle.provenance.get("shaft_model", DEFAULT_SHAFT_MODEL),
        bundle.provenance.get("shaft_options"),
    )

    def canonical(data: bytes) -> str:
        return json.dumps(json.loads(data), sort_keys=True, allow_nan=False)

    if canonical(expected_spec) != canonical(bundle.spec_bytes):
        raise ValueError("run model does not match bundle model")
    target = padded_track(bundle.spec_bytes, target)
    impact = bundle.ball_impact()
    lead = bundle.provenance.get("lead_in")
    lead_steps = 0
    if lead is not None:
        if not isinstance(lead, dict):
            raise ValueError("lead-in metadata must be a mapping")
        candidate = lead.get("steps")
        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, int)
            or candidate <= 0
            or bundle.dt_s != DT_S
            or lead.get("duration_s") != candidate * DT_S
            or impact is not None
            or bundle.provenance.get("impact_split_time_s") is not None
        ):
            raise ValueError("unsupported or inconsistent lead-in metadata")
        lead_steps = candidate
    started = time.perf_counter()
    rollout = closed_loop(
        engine,
        bundle.spec_bytes,
        times,
        target,
        duration_s=(bundle.steps + lead_steps) * bundle.dt_s,
        impact=None if impact is None else impact.rearmed(),
        split_time_s=bundle.provenance.get("impact_split_time_s"),
        policy=bundle.step_policy(),
    )
    if lead_steps:
        rollout, _ = drop_lead_in(rollout, lead_steps, bundle.effort_holds)
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
        "target_clock_offset_s": lead_steps * DT_S,
        "target_tracking": {
            "coordinate_order": bundle.coordinate_order,
            **target_tracking_summary(
                times,
                target,
                rollout.time_s + lead_steps * DT_S,
                rollout.q,
                max_endpoint_extension_s=DT_S / 2,
            ),
        },
        "velocity_target_tracking": {
            "coordinate_order": bundle.coordinate_order,
            **target_velocity_tracking_summary(
                times,
                target,
                rollout.time_s + lead_steps * DT_S,
                rollout.v,
                split_time_s=bundle.provenance.get("impact_split_time_s"),
                max_endpoint_extension_s=DT_S / 2,
            ),
        },
        "max_effort_difference_nm": float(
            np.abs(rollout.efforts - bundle.replay_efforts()).max()
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
    exp.add_argument(
        "--effort-hold-ms",
        type=float,
        default=1.0,
        help="controller hold recorded in the bundle (1, 0.5, 0.25 or 0.125 ms)",
    )
    exp.add_argument(
        "--shaft-model",
        choices=SHAFT_MODELS,
        default=DEFAULT_SHAFT_MODEL,
        help="rigid club (default) or the R6 chain shaft (#12686, #12691)",
    )
    add_shaft_arguments(exp)
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
        manifest = export(
            args.run_dir,
            args.out,
            args.duration,
            args.effort_hold_ms,
            args.shaft_model,
            shaft_options(args),
        )
        LOG.info("wrote %s: %d steps", args.out, manifest["steps"])
        return
    if args.command == "closed-loop":
        receipt = closed_loop_receipt(args.run_dir, args.bundle, args.engine)
    else:
        receipt = replay(args.bundle, args.engine, args.segment_ms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.out.with_suffix(".npz"), **_STATES)  # type: ignore[arg-type]  # NumPy's named-array stub
    LOG.info("%s: %s", args.engine, json.dumps(receipt["full_horizon"]))
    if "worst_segment" in receipt:
        LOG.info("worst %g ms segment: %s", args.segment_ms, receipt["worst_segment"])


if __name__ == "__main__":
    main()
