"""Generate a same-input bundle from a MuJoCo tracking run (#11607).

The canonical pipeline evaluates its computed-torque controller at every RK4
stage, so its logged torques cannot be replayed.  Here the same controller
(same gains and tracked reference) is evaluated once per step and held, with
the exact-KKT MuJoCo plant and per-step closure projection, so the logged
efforts reproduce the reference motion when replayed open loop.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.impact_force import ImpactForce
from src.shared.python.motion_matching.pipeline.constants import DT_S
from src.shared.python.motion_matching.same_input.bundle import InputBundle
from src.shared.python.motion_matching.same_input.closure import project_to_closure
from src.shared.python.motion_matching.same_input.integrator import (
    Rollout,
    integrate,
)
from src.shared.python.motion_matching.same_input.plant import VectorPlant

Array = NDArray[np.float64]


def _tracking_setup(
    spec_bytes: bytes,
    track_time_s: Array,
    q_track: Array,
    duration_s: float | None,
    split_time_s: float | None = None,
) -> tuple[Any, Array, Array, int]:
    """Controller, start state and step count shared by every closed-loop run.

    The controller is the pipeline computed-torque law built once on the
    MuJoCo model, so closed-loop runs in different engines use the identical
    control function of ``(t, q, v)``.  ``split_time_s`` (bundle clock) splits
    its reference rates at the ball impact.
    """
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs
    from src.shared.python.motion_matching.pipeline.dynamics import (
        build_tracking_controller,
    )

    times = np.asarray(track_time_s, dtype=float) - float(track_time_s[0])
    track = np.asarray(q_track, dtype=float)
    if times.ndim != 1 or track.shape[0] != times.size or np.any(np.diff(times) <= 0):
        raise ValueError("q_track rows must match strictly increasing times")
    span = float(times[-1]) if duration_s is None else float(duration_s)
    if not 0.0 < span <= float(times[-1]) + DT_S / 2:
        raise ValueError("duration_s must lie within the reference span")
    sim = fs.FullBodySimulator(NativeMujocoFullBodyModel(spec_bytes))
    q0 = fs.preload_feet(sim, track[0])
    v0 = sim.consistent_velocity(q0, np.gradient(track, times, axis=0)[0])
    controller = build_tracking_controller(sim, times, track, split_time_s=split_time_s)
    return controller, q0, v0, int(round(span / DT_S))


def closed_loop(
    engine: str,
    spec_bytes: bytes,
    track_time_s: Array,
    q_track: Array,
    *,
    duration_s: float | None = None,
    impact: ImpactForce | None = None,
    split_time_s: float | None = None,
) -> Rollout:
    """Track ``q_track`` in ``engine`` with the shared controller held per step.

    ``impact`` is the ball force plan on the bundle clock (time zero at
    ``track_time_s[0]``); an unscheduled passage plan is re-addressed to this
    run's start pose and latched from this engine's own state.
    ``split_time_s`` (same clock) splits the controller's reference rates at
    the reference's impact.
    """
    controller, q0, v0, steps = _tracking_setup(
        spec_bytes, track_time_s, q_track, duration_s, split_time_s
    )
    plant = VectorPlant(engine, spec_bytes)
    start = project_to_closure(plant, q0, v0)
    if impact is not None and impact.trigger == "passage" and not impact.scheduled:
        impact = impact.at_address(start.q, plant.kinematic_frames)
    return integrate(
        plant,
        start.q,
        start.v,
        lambda _k, t, q, v: controller(t, q, v),
        steps=steps,
        dt_s=DT_S,
        impact=impact,
    )


def generate_reference_bundle(
    spec_bytes: bytes,
    track_time_s: Array,
    q_track: Array,
    *,
    duration_s: float | None = None,
    provenance: dict[str, Any] | None = None,
    impact: ImpactForce | None = None,
    split_time_s: float | None = None,
) -> InputBundle:
    """Track ``q_track`` in MuJoCo with the pipeline controller held per step.

    Args:
        spec_bytes: Scaled full-body spec of the run.
        track_time_s: Reference sample times (seconds, increasing).
        q_track: Tracked reference (len(track_time_s), nv) in spec order.
        duration_s: Horizon; defaults to the reference span.
        provenance: Extra manifest fields (source run, capture, ...).
        impact: Ball force plan on the bundle clock; the force latched in
            MuJoCo is recorded as ``provenance["ball_impact"]`` so every
            engine's replay applies it identically (GCV-20).
        split_time_s: Reference impact time on the bundle clock, splitting
            the controller's reference rates (recorded in the provenance).
    """
    rollout = closed_loop(
        "mujoco",
        spec_bytes,
        track_time_s,
        q_track,
        duration_s=duration_s,
        impact=impact,
        split_time_s=split_time_s,
    )
    return InputBundle(
        spec_bytes=bytes(spec_bytes),
        coordinate_order=tuple(json.loads(spec_bytes)["coordinate_order"]),
        dt_s=DT_S,
        q0=rollout.q[0],
        v0=rollout.v[0],
        efforts=rollout.efforts,
        reference_q=rollout.q,
        reference_v=rollout.v,
        reference_engine="mujoco",
        provenance={
            "controller": "pipeline computed-torque (kkt), held per step",
            "max_pose_drift_m": float(rollout.pose_drift.max()),
            "ball_impact": rollout.ball_impact,
            "impact_split_time_s": split_time_s,
            **(provenance or {}),
        },
    )
