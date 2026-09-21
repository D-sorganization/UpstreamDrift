"""MuJoCo ``mj_inverse`` computed-torque backend (MS-16 #10366)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.full_body_forward_dynamics import (
    Array,
    ComputedTorqueGains,
    Controller,
    FullBodySimulator,
    _balance_acceleration,
    _check_gains,
    _root_regulation_acceleration,
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

    eq_saved = np.asarray(data.eq_active, dtype=int).copy()
    data.eq_active[:] = 0

    qacc = np.zeros(model.nv, dtype=float)
    qacc[simulator._dof] = acceleration_spec
    qacc_saved = data.qacc.copy()
    try:
        data.qacc[:] = qacc
        mj.mj_inverse(model, data)
        tau = data.qfrc_inverse.copy()
    finally:
        data.qacc[:] = qacc_saved
        data.eq_active[:] = eq_saved
    return tau


@precondition(
    lambda simulator, q, v, joint_acceleration: joint_acceleration is not None,
    "joint_acceleration required",
)
@postcondition(
    lambda result, **_: (
        isinstance(result, np.ndarray)
        and result.ndim == 1
        and np.isfinite(result).all()
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


def _computed_torque_mj_inverse(
    simulator: FullBodySimulator,
    q: Array,
    v: Array,
    q_ref: Array,
    v_ref: Array,
    a_ref: Array,
    gains: ComputedTorqueGains,
    com_ref: Array | None = None,
) -> Array:
    act = simulator.actuated
    omega = np.broadcast_to(
        np.asarray(gains.omega_rad_s, dtype=float), (simulator.nv,)
    )[act]
    wanted = (
        a_ref[act]
        + 2.0 * gains.zeta * omega * (v_ref[act] - v[act])
        + omega**2 * (q_ref[act] - q[act])
    )
    if gains.balance is not None and com_ref is not None:
        wanted = (
            wanted + _balance_acceleration(simulator, q, v, com_ref, gains.balance)[act]
        )
    if gains.root_regulation is not None:
        wanted = (
            wanted
            + _root_regulation_acceleration(
                simulator, q, v, q_ref, v_ref, gains.root_regulation
            )[act]
        )
    return inverse_dynamics_mj_inverse(simulator, q, v, wanted)


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
    if not 0.0 <= acceleration_feedforward <= 1.0:
        raise ValueError("acceleration_feedforward must lie in [0, 1]")
    times = np.asarray(time_ref, dtype=float)
    reference = np.asarray(q_ref, dtype=float)
    if (
        times.ndim != 1
        or np.any(np.diff(times) <= 0)
        or reference.shape != (times.size, simulator.nv)
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Reference times must increase with one finite q row each")
    _check_gains(omega_rad_s, zeta, balance)
    _check_gains(1.0, 1.0, root_regulation)
    if times.size > 1:
        velocity = np.gradient(reference, times, axis=0)
        acceleration = acceleration_feedforward * np.gradient(velocity, times, axis=0)
    else:
        velocity = np.zeros_like(reference)
        acceleration = np.zeros_like(reference)
    gains = ComputedTorqueGains(
        omega_rad_s=omega_rad_s,
        zeta=zeta,
        balance=balance,
        root_regulation=root_regulation,
    )

    def sample(table: Array, t: float) -> Array:
        return np.array([np.interp(t, times, table[:, k]) for k in range(simulator.nv)])

    def controller(t: float, q: Array, v: Array) -> Array:
        q_t, v_t, a_t = (
            sample(reference, t),
            sample(velocity, t),
            sample(acceleration, t),
        )
        com_ref = simulator.centre_of_mass(q_t)[0] if balance is not None else None
        return _computed_torque_mj_inverse(
            simulator, q, v, q_t, v_t, a_t, gains, com_ref
        )

    return controller
