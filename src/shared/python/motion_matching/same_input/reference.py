"""Generate a same-input bundle from a MuJoCo tracking run (#11607).

The canonical pipeline evaluates its computed-torque controller at every RK4
stage, so its logged torques cannot be replayed.  Here the same controller
(same gains and tracked reference) is evaluated once per step and held, with
the exact-KKT MuJoCo plant and per-step closure projection, so the logged
efforts reproduce the reference motion when replayed open loop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.same_input.bundle import InputBundle
from src.shared.python.motion_matching.same_input.closure import project_to_closure
from src.shared.python.motion_matching.same_input.integrator import integrate
from src.shared.python.motion_matching.same_input.plant import VectorPlant

Array = NDArray[np.float64]


def generate_reference_bundle(
    spec_bytes: bytes,
    track_time_s: Array,
    q_track: Array,
    *,
    dt_s: float = 1e-3,
    duration_s: float | None = None,
    provenance: dict[str, Any] | None = None,
) -> InputBundle:
    """Track ``q_track`` in MuJoCo with the pipeline controller held per step.

    Args:
        spec_bytes: Scaled full-body spec of the run.
        track_time_s: Reference sample times (seconds, increasing).
        q_track: Tracked reference (len(track_time_s), nv) in spec order.
        dt_s: Integration step.
        duration_s: Horizon; defaults to the reference span.
        provenance: Extra manifest fields (source run, capture, ...).
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
    if not 0.0 < span <= float(times[-1]) + 1e-12:
        raise ValueError("duration_s must lie within the reference span")
    plant = VectorPlant("mujoco", spec_bytes)
    sim = fs.FullBodySimulator(NativeMujocoFullBodyModel(spec_bytes))
    if sim.names != plant.coordinate_order:
        raise ValueError("simulator and plant coordinate orders differ")
    q0 = fs.preload_feet(sim, track[0])
    v0 = sim.consistent_velocity(q0, np.gradient(track, times, axis=0)[0])
    start = project_to_closure(plant, q0, v0)
    controller = build_tracking_controller(sim, times, track)
    rollout = integrate(
        plant,
        start.q,
        start.v,
        lambda _k, t, q, v: controller(t, q, v),
        steps=int(round(span / dt_s)),
        dt_s=dt_s,
    )
    return InputBundle(
        spec_bytes=bytes(spec_bytes),
        coordinate_order=plant.coordinate_order,
        dt_s=dt_s,
        q0=rollout.q[0],
        v0=rollout.v[0],
        efforts=rollout.efforts,
        reference_q=rollout.q,
        reference_v=rollout.v,
        reference_engine="mujoco",
        provenance={
            "controller": "pipeline computed-torque (kkt), held per step",
            "max_pose_drift_m": float(rollout.pose_drift.max()),
            **(provenance or {}),
        },
    )
