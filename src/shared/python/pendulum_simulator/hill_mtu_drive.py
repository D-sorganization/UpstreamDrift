"""HILL_MTU drive mode for the canonical 3DOF swing (Refs #13888, epic #13789)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from src.shared.python.biomechanics.activation_dynamics import ActivationDynamics
from src.shared.python.biomechanics.hill_muscle import HillMuscleModel, MuscleParameters
from src.shared.python.biomechanics.mtu_joint_actuator import (
    MTUJointActuator,
    MTUJointGeometry,
    MTUJointState,
    TendonMode,
)
from src.shared.python.biomechanics.muscle_equilibrium import EquilibriumSolver
from src.shared.python.loop_nullspace.muscle_stiffness import StiffnessNotQualified

MTU_STATE_DIM = 12
JOINT_STATE_DIM = 6
HILL_STATE_DIM = JOINT_STATE_DIM + MTU_STATE_DIM
RESIDUAL_TORQUE_RMS_LIMIT_NM = 1e-6

DEFAULT_COCONTRACTION_LEVELS: tuple[float, ...] = (0.05, 0.2, 0.5)
SMOKE_COCONTRACTION_LEVELS: tuple[float, ...] = (0.05, 0.2)
CANONICAL_JOINT_Q_REF: tuple[float, float, float] = (-1.5, 0.5, 0.2)


@dataclass(frozen=True)
class HillMtuDriveSpec:
    """Open-loop Hill MTU drive for one 3DOF nominal swing."""

    cocontraction: float
    tendon_mode: TendonMode
    actuators: tuple[MTUJointActuator, MTUJointActuator, MTUJointActuator]
    activation: ActivationDynamics
    excitation: np.ndarray
    activation_target: np.ndarray
    torque_command_nm: np.ndarray
    residual_torque_nm: np.ndarray
    residual_torque_rms_nm: float
    effective_kq: np.ndarray
    effective_dq: np.ndarray
    tau_act_s: float
    tau_deact_s: float
    qualified: bool

    def __post_init__(self) -> None:
        c = self.cocontraction
        if not math.isfinite(c) or c < 0.0 or c > 2.0:
            raise ValueError("cocontraction must lie in [0, 2]")
        if self.excitation.ndim != 3 or self.excitation.shape[2] != 2:
            raise ValueError("excitation must be (n_steps, 3 joints, 2 muscles)")
        if self.activation_target.shape != self.excitation.shape:
            raise ValueError("activation_target must match excitation shape")
        if self.torque_command_nm.shape != self.residual_torque_nm.shape:
            raise ValueError("torque_command_nm must match residual_torque_nm")
        if self.residual_torque_nm.shape != self.excitation.shape[:2]:
            raise ValueError("residual_torque_nm must match (n_steps, 3 joints)")
        if self.effective_kq.shape != self.effective_dq.shape:
            raise ValueError("effective K_q and D_q series must match shape")
        if self.residual_torque_rms_nm < 0.0 or not math.isfinite(
            self.residual_torque_rms_nm
        ):
            raise ValueError("residual_torque_rms_nm must be finite and non-negative")


def default_triple_mtu_actuators(
    tendon_mode: TendonMode = TendonMode.RIGID,
    q_ref: Sequence[float] = CANONICAL_JOINT_Q_REF,
) -> tuple[MTUJointActuator, MTUJointActuator, MTUJointActuator]:
    """Three joint MTUs scaled for canonical torque magnitudes."""
    activation = ActivationDynamics(tau_act=0.010, tau_deact=0.040)
    if len(q_ref) != 3:
        raise ValueError("q_ref must contain three reference joint angles")

    def _one(scale: float, q0: float) -> MTUJointActuator:
        r_ag, r_ant = 0.045, -0.040
        ag_p = MuscleParameters(
            F_max=2800.0 * scale, l_opt=0.10, l_slack=0.25, v_max=12.0
        )
        ant_p = MuscleParameters(
            F_max=3200.0 * scale, l_opt=0.10, l_slack=0.25, v_max=12.0
        )
        geom = MTUJointGeometry(
            moment_arm_agonist=r_ag,
            moment_arm_antagonist=r_ant,
            l_mt_nominal_agonist=ag_p.l_opt + ag_p.l_slack + r_ag * q0,
            l_mt_nominal_antagonist=ant_p.l_opt + ant_p.l_slack + r_ant * q0,
        )
        return MTUJointActuator(
            HillMuscleModel(ag_p),
            HillMuscleModel(ant_p),
            geom,
            tendon_mode=tendon_mode,
            activation=activation,
        )

    joint_scales = (18.0, 7.0, 2.5)
    return tuple(_one(joint_scales[j], float(q_ref[j])) for j in range(3))


def invert_excitation(
    dynamics: ActivationDynamics, a: float, dadt_target: float
) -> float:
    """Invert Thelen activation ODE for open-loop u given desired dadt."""
    a_cl = float(np.clip(a, dynamics.min_activation, 1.0))
    tau = dynamics._effective_tau(a_cl + dadt_target * 0.001, a_cl)
    return float(np.clip(a_cl + dadt_target * tau, dynamics.min_activation, 1.0))


class EquilibriumNotConverged(RuntimeError):
    """MTU fiber-length equilibrium did not converge at the sampled pose."""


def _equilibrium_state(
    actuator: MTUJointActuator, q: float, a_ag: float, a_ant: float
) -> MTUJointState:
    floor = actuator._activation.min_activation
    a_ag = float(np.clip(a_ag, floor, 1.0))
    a_ant = float(np.clip(a_ant, floor, 1.0))
    sag, sant = (
        EquilibriumSolver(actuator.agonist),
        EquilibriumSolver(actuator.antagonist),
    )
    l_ag = actuator.geometry.l_mt(True, q)
    l_ant = actuator.geometry.l_mt(False, q)
    try:
        l_m_ag = sag.solve_fiber_length(l_ag, a_ag)
        l_m_ant = sant.solve_fiber_length(l_ant, a_ant)
    except RuntimeError as exc:
        raise EquilibriumNotConverged(str(exc)) from exc
    return MTUJointState(a_ag, a_ant, l_m_ag, l_m_ant)


def solve_matched_activations(
    actuator: MTUJointActuator,
    q: float,
    qdot: float,
    tau_target: float,
    cocontraction: float,
) -> tuple[float, float, bool]:
    """Match net torque at fixed co-contraction sum a_ag + a_ant = c."""
    c = float(cocontraction)
    if c <= 0.0:
        return (0.0, 0.0, False)

    def torque_at(s: float) -> float:
        a_ag = float(np.clip(0.5 * c + s, 0.0, 1.0))
        a_ant = float(np.clip(0.5 * c - s, 0.0, 1.0))
        try:
            state = _equilibrium_state(actuator, q, a_ag, a_ant)
        except EquilibriumNotConverged:
            return float("nan")
        return actuator.torque(q, qdot, state)

    s_min, s_max = -0.5 * c, 0.5 * c
    t_lo, t_hi = torque_at(s_min), torque_at(s_max)
    if not (math.isfinite(t_lo) and math.isfinite(t_hi)):
        return (0.0, 0.0, False)
    if min(t_lo, t_hi) - 1e-9 > tau_target or max(t_lo, t_hi) + 1e-9 < tau_target:
        return (0.0, 0.0, False)
    for _ in range(80):
        mid = 0.5 * (s_min + s_max)
        if (torque_at(mid) - tau_target) * (t_hi - tau_target) > 0.0:
            s_max = mid
        else:
            s_min = mid
    s = 0.5 * (s_min + s_max)
    a_ag = float(np.clip(0.5 * c + s, 0.0, 1.0))
    a_ant = float(np.clip(0.5 * c - s, 0.0, 1.0))
    return (a_ag, a_ant, True)


def pack_hill_state(
    qv: Sequence[float], mtu_states: Sequence[MTUJointState]
) -> np.ndarray:
    if len(qv) != JOINT_STATE_DIM or len(mtu_states) != 3:
        raise ValueError("expected 6 joint states and 3 MTU states")
    parts = list(map(float, qv))
    for st in mtu_states:
        parts.extend((st.a_agonist, st.a_antagonist, st.l_m_agonist, st.l_m_antagonist))
    return np.asarray(parts, dtype=np.float64)


def unpack_hill_state(x: np.ndarray) -> tuple[np.ndarray, list[MTUJointState]]:
    if x.shape != (HILL_STATE_DIM,):
        raise ValueError(f"hill state must be length {HILL_STATE_DIM}")
    qv = x[:JOINT_STATE_DIM]
    states: list[MTUJointState] = []
    for j in range(3):
        b = JOINT_STATE_DIM + 4 * j
        states.append(
            MTUJointState(
                float(x[b]), float(x[b + 1]), float(x[b + 2]), float(x[b + 3])
            )
        )
    return qv, states


def build_matched_excitation(
    nominal: Any,
    cocontraction: float,
    actuators: Sequence[MTUJointActuator],
    tendon_mode: TendonMode = TendonMode.RIGID,
) -> HillMtuDriveSpec:
    """Solve open-loop u(t) along the nominal for one co-contraction level."""
    from src.shared.python.pendulum_simulator import drive_modes as dm

    if not isinstance(nominal, dm.NominalTrajectory):
        raise TypeError("nominal must be NominalTrajectory")
    n = len(nominal.torques)
    dt = float(nominal.times[1] - nominal.times[0])
    acts = tuple(
        MTUJointActuator(
            a.agonist,
            a.antagonist,
            a.geometry,
            tendon_mode=tendon_mode,
            activation=a._activation,
        )
        for a in actuators
    )
    dynamics = acts[0]._activation
    a_target = np.zeros((n, 3, 2), dtype=np.float64)
    tau_cmd = np.zeros((n, 3), dtype=np.float64)
    feasible = True
    for k in range(n):
        tau_cmd[k] = nominal.torques[k]
        q, v, _tau_c, _tau_u = nominal.at(float(nominal.times[k]))
        for j in range(3):
            a_ag, a_ant, ok = solve_matched_activations(
                acts[j], float(q[j]), float(v[j]), float(tau_cmd[k, j]), cocontraction
            )
            if not ok:
                feasible = False
            a_target[k, j, 0] = a_ag
            a_target[k, j, 1] = a_ant
    excitation = np.zeros((n, 3, 2), dtype=np.float64)
    residual = np.zeros((n, 3), dtype=np.float64)
    kq = np.full((n, 3), np.nan, dtype=np.float64)
    dq = np.full((n, 3), np.nan, dtype=np.float64)
    for k in range(n):
        q, v, _tau_c, _tau_u = nominal.at(float(nominal.times[k]))
        for j in range(3):
            try:
                st = _equilibrium_state(
                    acts[j], float(q[j]), a_target[k, j, 0], a_target[k, j, 1]
                )
            except EquilibriumNotConverged:
                feasible = False
                residual[k, j] = float("nan")
                kq[k, j], dq[k, j] = float("nan"), float("nan")
                continue
            tau_hat = acts[j].torque(float(q[j]), float(v[j]), st)
            residual[k, j] = tau_hat - float(tau_cmd[k, j])
            excitation[k, j, 0] = float(a_target[k, j, 0])
            excitation[k, j, 1] = float(a_target[k, j, 1])
            try:
                lin = acts[j].linearise(float(q[j]), float(v[j]), st)
                kq[k, j], dq[k, j] = lin.K_q, lin.D_q
            except StiffnessNotQualified:
                kq[k, j], dq[k, j] = float("nan"), float("nan")
    if feasible and np.all(np.isfinite(residual)):
        rms = float(np.sqrt(np.mean(residual**2)))
    else:
        feasible = False
        rms = float(RESIDUAL_TORQUE_RMS_LIMIT_NM)
    spec = HillMtuDriveSpec(
        cocontraction=cocontraction,
        tendon_mode=tendon_mode,
        actuators=acts,
        activation=dynamics,
        excitation=excitation,
        activation_target=a_target,
        torque_command_nm=tau_cmd,
        residual_torque_nm=residual,
        residual_torque_rms_nm=rms,
        effective_kq=kq,
        effective_dq=dq,
        tau_act_s=dynamics.tau_act,
        tau_deact_s=dynamics.tau_deact,
        qualified=False,
    )
    return _finalize_hill_spec(spec, nominal, feasible=feasible)


def _finalize_hill_spec(
    spec: HillMtuDriveSpec, nominal: Any, *, feasible: bool
) -> HillMtuDriveSpec:
    from src.shared.python.pendulum_simulator import sensitivity as sens

    residual = np.zeros_like(spec.residual_torque_nm)
    n = len(nominal.torques)
    dt = float(nominal.times[1] - nominal.times[0])
    if not feasible:
        return HillMtuDriveSpec(
            spec.cocontraction,
            spec.tendon_mode,
            spec.actuators,
            spec.activation,
            spec.excitation,
            spec.activation_target,
            spec.torque_command_nm,
            spec.residual_torque_nm,
            spec.residual_torque_rms_nm,
            spec.effective_kq,
            spec.effective_dq,
            spec.tau_act_s,
            spec.tau_deact_s,
            qualified=False,
        )
    x = sens._initial_hill_state(spec, nominal)
    for k in range(n):
        qv, mtu_states = unpack_hill_state(x)
        q, v = qv[:3], qv[3:]
        for j in range(3):
            tau_hat = spec.actuators[j].torque(float(q[j]), float(v[j]), mtu_states[j])
            residual[k, j] = tau_hat - float(spec.torque_command_nm[k, j])
        try:
            x = sens.step_hill_mtu_rk4(spec, x, k, dt)
        except EquilibriumNotConverged:
            feasible = False
            break
    if not np.all(np.isfinite(residual)):
        feasible = False
        rms = float(RESIDUAL_TORQUE_RMS_LIMIT_NM)
    else:
        rms = float(np.sqrt(np.mean(residual**2)))
    qualified = feasible and rms < RESIDUAL_TORQUE_RMS_LIMIT_NM
    return HillMtuDriveSpec(
        spec.cocontraction,
        spec.tendon_mode,
        spec.actuators,
        spec.activation,
        spec.excitation,
        spec.activation_target,
        spec.torque_command_nm,
        residual,
        rms,
        spec.effective_kq,
        spec.effective_dq,
        spec.tau_act_s,
        spec.tau_deact_s,
        qualified,
    )
