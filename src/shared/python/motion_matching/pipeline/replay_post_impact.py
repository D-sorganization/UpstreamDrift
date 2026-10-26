"""Post-impact replay growth audit for capture-A (#13780, ML-WS-3).

Quantifies per-phase and per-segment forward-dynamics marker RMS on the
canonical driver replay, split at the detected impact and the backswing top,
and runs a small perturbation sweep over the initial state and integrator step.
Reuses :mod:`fd_phase`, :mod:`replay_sensitivity` and
:func:`dynamics.segment_rms`; reports only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from src.shared.python.motion_matching.pipeline import dynamics as dyn
from src.shared.python.motion_matching.pipeline.fd_phase import (
    fd_phase_report,
    model_face_path,
    phase_marker_rms,
)
from src.shared.python.motion_matching.pipeline.replay_sensitivity import (
    PERTURBATION,
    perturbation_direction,
    replay_sensitivity,
)

#: Documented address-to-impact RMS on capture-A (CANONICAL_RUN.md): 31–34 mm.
ADDRESS_TO_IMPACT_DOCUMENTED_MAX_M = 0.034
#: Acceptance: after-impact whole-phase RMS must not exceed this multiple of
#: address-to-impact RMS (G1 whole-swing gate is broken by post-impact growth).
POST_IMPACT_TO_ADDRESS_RATIO_MAX = 2.0
#: Integrator step scale factors for the perturbation sweep.
DT_SCALE_FACTORS: tuple[float, ...] = (0.5, 1.0, 2.0)
#: Initial-coordinate perturbation magnitudes (m or rad); 0 is nominal.
Q0_EPSILON_FACTORS: tuple[float, ...] = (0.0, PERTURBATION)
CAPTURE_A_ID = "capture-A"
CANONICAL_SEEDS_DIR = (
    Path(__file__).resolve().parents[5]
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
)


@dataclass(frozen=True)
class PhaseBoundaries:
    """Monotone swing phase split times (seconds)."""

    top_time_s: float
    impact_time_s: float
    top_frame: int
    impact_frame: int


def resolve_phase_boundaries(
    times: Sequence[float] | np.ndarray,
    face_centres: np.ndarray,
) -> PhaseBoundaries:
    """Impact and backswing-top times from the reference face-centre path.

    Preconditions:
        ``times`` is 1-D, ``face_centres`` is ``(len(times), 3)``, and impact
        lies inside the span with a preceding backswing top.

    Postcondition:
        ``top_time_s < impact_time_s`` and both lie in ``[times[0], times[-1]]``.
    """
    from src.shared.python.model_appearance import club_face

    t = np.asarray(times, dtype=float)
    head = np.asarray(face_centres, dtype=float)
    if t.ndim != 1 or head.shape != (t.size, 3):
        raise ValueError("times and face_centres must share length (frames, 3)")
    impact_time, impact_k, _ = club_face.ball_passage(t, head)
    top_k = int(club_face.top_of_backswing_index(t, head))
    top_time = float(t[top_k])
    if not (t[0] <= top_time < impact_time <= t[-1]):
        raise ValueError("phase boundaries must be monotone inside the capture span")
    return PhaseBoundaries(
        top_time_s=top_time,
        impact_time_s=float(impact_time),
        top_frame=top_k,
        impact_frame=int(impact_k),
    )


def three_phase_marker_rms(
    times: Sequence[float] | np.ndarray,
    errors: np.ndarray,
    valid: np.ndarray,
    boundaries: PhaseBoundaries,
) -> dict[str, Any]:
    """Valid-marker RMS for address→top, top→impact and after impact."""
    t = np.asarray(times, dtype=float)
    err = np.asarray(errors, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if err.shape != mask.shape or err.shape[0] != t.shape[0]:
        raise ValueError("errors, valid and times must share shape (frames, markers)")
    top, impact = boundaries.top_time_s, boundaries.impact_time_s
    phases = {
        "address_to_top": t <= top,
        "top_to_impact": (t > top) & (t <= impact),
        "after_impact": t > impact,
    }

    def rms(rows: np.ndarray) -> float | None:
        sample = err[rows][mask[rows]]
        return float(np.sqrt(np.mean(sample**2))) if sample.size else None

    out: dict[str, Any] = {
        "top_time_s": top,
        "impact_time_s": impact,
        "top_frame": boundaries.top_frame,
        "impact_frame": boundaries.impact_frame,
    }
    for name, rows in phases.items():
        out[f"fd_rms_{name}_m"] = rms(rows)
        out[f"{name}_frames"] = int(rows.sum())
    return out


def phase_segment_rms(
    labels: Sequence[str],
    errors: np.ndarray,
    valid: np.ndarray,
    times: Sequence[float] | np.ndarray,
    boundaries: PhaseBoundaries,
) -> dict[str, dict[str, float]]:
    """Per-segment RMS in each swing phase."""
    t = np.asarray(times, dtype=float)
    top, impact = boundaries.top_time_s, boundaries.impact_time_s
    masks = {
        "address_to_top": t <= top,
        "top_to_impact": (t > top) & (t <= impact),
        "after_impact": t > impact,
    }
    return {
        phase: dyn.segment_rms(labels, errors[rows], valid[rows])
        for phase, rows in masks.items()
    }


def assert_post_impact_acceptance(report: Mapping[str, Any]) -> None:
    """DbC gate: post-impact replay must not dominate address-to-impact RMS."""
    phases = report.get("phases", {})
    pre = phases.get("fd_rms_address_to_impact_m")
    post = phases.get("fd_rms_after_impact_m")
    if pre is None or post is None:
        raise ValueError("phase RMS unavailable; cannot evaluate acceptance")
    if pre <= 0:
        raise ValueError("address-to-impact RMS must be positive")
    ratio = float(post) / float(pre)
    if ratio > POST_IMPACT_TO_ADDRESS_RATIO_MAX:
        raise AssertionError(
            f"post-impact RMS {post:.4f} m is {ratio:.1f}x address-to-impact "
            f"{pre:.4f} m (acceptance <= {POST_IMPACT_TO_ADDRESS_RATIO_MAX}x)"
        )


def _attachments_from_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, tuple[str, tuple[float, ...]]]:
    raw = receipt["ik"]["attachments_m"]
    return {k: (v["body"], tuple(v["offset_m"])) for k, v in raw.items()}


def _interpolate_track(
    lane_times: np.ndarray, track_times: np.ndarray, q_track: np.ndarray, nv: int
) -> np.ndarray:
    return np.array(
        [
            [np.interp(t, track_times, q_track[:, k]) for k in range(nv)]
            for t in lane_times
        ]
    )


def _after_impact_rms(
    sim: Any,
    lane: Any,
    kin: Any,
    q_track: np.ndarray,
    spec: Mapping[str, Any],
    *,
    dt_scale: float = 1.0,
    q0_delta: np.ndarray | None = None,
) -> dict[str, float | None]:
    q = np.asarray(q_track, dtype=float).copy()
    if q0_delta is not None:
        q[0] = q[0] + q0_delta
    base_dt = dyn.replay_dt_s(sim.adapter) * dt_scale
    with patch(
        "src.shared.python.motion_matching.pipeline.dynamics.replay_dt_s",
        return_value=base_dt,
    ):
        _, sim_q = dyn.replay(sim, lane, q)
    errors = dyn.marker_errors(kin, sim_q, lane.points)
    face = model_face_path(kin, q, lane.times, spec)
    block = fd_phase_report(lane.times, errors, lane.valid, face.centres)
    return {
        "fd_rms_address_to_impact_m": block.get("fd_rms_address_to_impact_m"),
        "fd_rms_after_impact_m": block.get("fd_rms_after_impact_m"),
    }


def perturbation_sweep(
    sim: Any,
    lane: Any,
    kin: Any,
    q_track: np.ndarray,
    spec: Mapping[str, Any],
    impact_time_s: float,
) -> dict[str, Any]:
    """Small initial-state and timestep perturbations; where growth starts."""
    direction = perturbation_direction(q_track.shape)[0]

    def run(reference: np.ndarray) -> np.ndarray:
        _, sim_q = dyn.replay(sim, lane, reference)
        return sim_q

    nominal = run(q_track)
    sensitivity = replay_sensitivity(
        run, kin.marker_positions, lane.times, q_track, nominal, impact_time_s
    )
    q0_rows = []
    for eps in Q0_EPSILON_FACTORS:
        delta = direction * eps if eps else None
        rms = _after_impact_rms(sim, lane, kin, q_track, spec, q0_delta=delta)
        q0_rows.append({"epsilon": eps, **rms})
    dt_rows = []
    for scale in DT_SCALE_FACTORS:
        rms = _after_impact_rms(sim, lane, kin, q_track, spec, dt_scale=scale)
        dt_rows.append({"dt_scale": scale, **rms})
    return {
        "replay_sensitivity": sensitivity,
        "initial_state": q0_rows,
        "timestep": dt_rows,
    }


def capture_a_replay_report(
    *,
    seeds_dir: Path | None = None,
    capture_path: Path | None = None,
) -> dict[str, Any]:
    """Full receipt block for capture-A on the canonical driver seeds run."""
    from src.engines.physics_engines.mujoco.python.full_body_ik import (
        FullBodyMarkerKinematics,
    )
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )
    from src.motion_capture.capture_registry import resolve_capture
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs
    from src.shared.python.motion_matching.pipeline.lane import Lane

    import json

    seeds = seeds_dir or CANONICAL_SEEDS_DIR
    receipt = json.loads((seeds / "receipt.json").read_text(encoding="utf-8"))
    spec_bytes = (seeds / "full_body_spec_hipcal_scaled.json").read_bytes()
    spec = json.loads(spec_bytes)
    record = np.load(seeds / "dynamics_record.npz")
    labels = tuple(receipt["labels"])
    attachments = _attachments_from_receipt(receipt)
    c3d = capture_path or resolve_capture(CAPTURE_A_ID)
    lane = Lane(c3d=c3d, labels=labels)
    lane.set_impact_split(attachments, spec)
    sim = fs.FullBodySimulator(NativeMujocoFullBodyModel(spec_bytes))
    kin = FullBodyMarkerKinematics(sim.adapter, attachments)
    q_track = _interpolate_track(
        lane.times, record["track_time_s"], record["q_track"], sim.nv
    )
    _, sim_q = dyn.replay(sim, lane, q_track)
    errors = dyn.marker_errors(kin, sim_q, lane.points)
    face = model_face_path(kin, q_track, lane.times, spec)
    if face.centres is None:
        raise ValueError(f"reference face centre unavailable: {face.reason}")
    boundaries = resolve_phase_boundaries(lane.times, face.centres)
    fd_block = fd_phase_report(lane.times, errors, lane.valid, face.centres)
    if fd_block.get("status") != "ok":
        raise ValueError(f"fd_phase unavailable: {fd_block.get('reason')}")
    two_phase = phase_marker_rms(
        lane.times, errors, lane.valid, boundaries.impact_time_s
    )
    three_phase = three_phase_marker_rms(lane.times, errors, lane.valid, boundaries)
    phases = {
        "fd_rms_address_to_impact_m": two_phase["fd_rms_address_to_impact_m"],
        "fd_rms_after_impact_m": two_phase["fd_rms_after_impact_m"],
        "address_to_impact_frames": two_phase["address_to_impact_frames"],
        "after_impact_frames": two_phase["after_impact_frames"],
        "three_phase": three_phase,
    }
    return {
        "capture": CAPTURE_A_ID,
        "canonical_run": seeds.name,
        "impact_detector": fd_block["impact_detector"],
        "impact_time_s": boundaries.impact_time_s,
        "phases": phases,
        "segment_rms_by_phase_m": phase_segment_rms(
            labels, errors, lane.valid, lane.times, boundaries
        ),
        "perturbation_sweep": perturbation_sweep(
            sim, lane, kin, q_track, spec, boundaries.impact_time_s
        ),
        "acceptance": {
            "post_impact_to_address_ratio_max": POST_IMPACT_TO_ADDRESS_RATIO_MAX,
            "address_to_impact_documented_max_m": ADDRESS_TO_IMPACT_DOCUMENTED_MAX_M,
        },
    }
