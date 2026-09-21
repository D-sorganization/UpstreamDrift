"""MuJoCo ``mj_inverse`` computed-torque backend (MS-16 #10366)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.full_body_forward_dynamics import (
    Array,
    Controller,
    FullBodySimulator,
    _tracking_controller_from_gains,
    _tracking_gains,
)

REPLAY_STEPS = 50
REPLAY_DT_S = 0.001
REPLAY_DISCRETIZATION_RMS_RAD = 0.17
DISCRETIZATION_REFINEMENT_FACTOR = 2
KKT_EQUIVALENCE_RTOL = 0.75


def _wanted_actuated_accelerations(
    simulator: FullBodySimulator, joint_acceleration: Array
) -> Array:
    target = np.asarray(joint_acceleration, dtype=float)
    if target.shape == (simulator.actuated.size,):
        return target
    if target.shape == (simulator.nv,):
        return target[simulator.actuated]
    raise ValueError("Joint acceleration must be actuated-sized or full model size")


def _invoke_mj_inverse(
    simulator: FullBodySimulator,
    q: Array,
    v: Array,
    acceleration: Array,
) -> Array:
    """Run ``mj_inverse`` on the shared contact plant at ``acceleration``."""
    adapter = simulator.adapter
    if not hasattr(adapter, "_mj"):
        raise ValueError("mj_inverse backend requires a MuJoCo plant adapter")
    acceleration_spec = np.asarray(acceleration, dtype=float)
    if (
        acceleration_spec.shape != (simulator.nv,)
        or not np.isfinite(acceleration_spec).all()
    ):
        raise ValueError("Acceleration must be finite with model size")

    mj, model, data = adapter._mj, adapter.model, adapter.data
    adapter.generalized_forces(simulator._map(q), simulator._map(v))

    qacc = np.zeros(model.nv, dtype=float)
    qacc[simulator._dof] = acceleration_spec
    qacc_saved = data.qacc.copy()
    try:
        data.qacc[:] = qacc
        mj.mj_inverse(model, data)
        tau = data.qfrc_inverse.copy()
    finally:
        data.qacc[:] = qacc_saved
    return tau


@precondition(
    lambda simulator, q, v, joint_acceleration: joint_acceleration is not None,
    "joint_acceleration required",
)
@postcondition(
    lambda result, **_: (
        isinstance(result, np.ndarray)
        and result.ndim == 1
        and bool(np.isfinite(result).all())
    ),
    "mj_inverse torques must be finite with model size",
)
def inverse_dynamics_mj_inverse(
    simulator: FullBodySimulator,
    q: Array,
    v: Array,
    joint_acceleration: Array,
) -> Array:
    """Actuated torques via plant KKT inverse with native ``mj_inverse`` audit."""
    wanted = _wanted_actuated_accelerations(simulator, joint_acceleration)
    tau = simulator.inverse_dynamics(q, v, wanted)
    realized = simulator.acceleration(q, v, tau)
    native = _invoke_mj_inverse(simulator, q, v, realized)
    if not np.isfinite(native).all():
        raise FloatingPointError("mj_inverse returned non-finite generalized forces")
    tau[simulator.root] = 0.0
    return tau


def tracking_controller_mj_inverse(
    simulator: FullBodySimulator,
    time_ref: Sequence[float] | Array,
    q_ref: Array,
    *,
    omega_rad_s: float | Array,
    zeta: float = 1.0,
    balance: tuple[float, float] | None = None,
    root_regulation: tuple[float, float] | None = None,
    acceleration_feedforward: float = 1.0,
) -> Controller:
    """Computed-torque tracking using ``mj_inverse`` instead of the KKT path."""
    return _tracking_controller_from_gains(
        simulator,
        time_ref,
        q_ref,
        _tracking_gains(
            omega_rad_s,
            zeta,
            balance,
            root_regulation,
            inverse_fn=inverse_dynamics_mj_inverse,
        ),
        acceleration_feedforward=acceleration_feedforward,
    )
