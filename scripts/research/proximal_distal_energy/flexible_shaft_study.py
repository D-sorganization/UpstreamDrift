"""Reduced-order planar shaft-flex model with explicit energy accounting.

The model reuses the repository's tested relative-coordinate triple-pendulum
mass, velocity-bias, gravity, kinematics, and joint-force functions.  The third
joint is interpreted as a lumped shaft-bending mode with a linear spring and
viscous damper.  The corresponding rigid model is the exact coordinate
reduction ``phi2 = dphi2 = 0`` of the same mass distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from src.shared.python.pendulum_simulator.physics_triple import (
    TriplePendulumParams,
    coriolis_vector,
    forward_kinematics,
    gravity_vector,
    kinetic_energy,
    mass_matrix,
    net_joint_forces,
    potential_energy,
)


@dataclass(frozen=True)
class FlexibleShaftParams:
    """Declared parameters for the three-link shaft-flex surrogate."""

    arm_mass_kg: float
    proximal_shaft_mass_kg: float
    distal_head_mass_kg: float
    arm_length_m: float
    proximal_shaft_length_m: float
    distal_shaft_length_m: float
    gravity_m_s2: float
    gravity_enabled: bool
    joint_damping_enabled: bool
    shoulder_damping_nms_rad: float
    wrist_damping_nms_rad: float
    shaft_stiffness_nm_rad: float
    shaft_damping_nms_rad: float
    shoulder_torque_nm: float
    wrist_drive_nm: float
    wrist_restrain_nm: float
    wrist_onset_s: float
    torque_cut_time_s: float | None = None

    @classmethod
    def reference(cls) -> FlexibleShaftParams:
        """Return the declared reference case used by the publication."""
        return cls(
            arm_mass_kg=7.5,
            proximal_shaft_mass_kg=0.15,
            distal_head_mass_kg=0.20,
            arm_length_m=0.75,
            proximal_shaft_length_m=0.45,
            distal_shaft_length_m=0.55,
            gravity_m_s2=8.033,
            gravity_enabled=True,
            joint_damping_enabled=True,
            shoulder_damping_nms_rad=0.4,
            wrist_damping_nms_rad=0.25,
            shaft_stiffness_nm_rad=80.0,
            shaft_damping_nms_rad=0.6,
            shoulder_torque_nm=60.0,
            wrist_drive_nm=15.0,
            wrist_restrain_nm=10.0,
            wrist_onset_s=0.10,
        )

    def with_updates(self, **updates: Any) -> FlexibleShaftParams:
        """Return an immutable copy with declared fields replaced."""
        return replace(self, **updates)

    def __post_init__(self) -> None:
        positive = (
            "arm_mass_kg",
            "proximal_shaft_mass_kg",
            "distal_head_mass_kg",
            "arm_length_m",
            "proximal_shaft_length_m",
            "distal_shaft_length_m",
            "shaft_stiffness_nm_rad",
        )
        nonnegative = (
            "gravity_m_s2",
            "shoulder_damping_nms_rad",
            "wrist_damping_nms_rad",
            "shaft_damping_nms_rad",
        )
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in nonnegative:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def triple(self) -> TriplePendulumParams:
        """Build shared three-link parameters with losses decomposed externally."""
        return TriplePendulumParams(
            m1=self.arm_mass_kg,
            m2=self.proximal_shaft_mass_kg,
            m3=self.distal_head_mass_kg,
            L1=self.arm_length_m,
            L2=self.proximal_shaft_length_m,
            L3=self.distal_shaft_length_m,
            g=self.gravity_m_s2 if self.gravity_enabled else 0.0,
        )


@dataclass(frozen=True)
class FlexibleTrace:
    """Deterministic flexible or rigid rollout and term-level accelerations."""

    t: np.ndarray
    state: np.ndarray
    qddot: np.ndarray
    controls: np.ndarray
    contributions: dict[str, np.ndarray]
    rigid: bool


def default_initial_state() -> np.ndarray:
    """Return arm, wrist, flex, and velocity initial conditions."""
    # theta1 + phi1 = -pi/2 makes the undeformed club horizontal at release.
    return np.array([-2.2, -np.pi / 2.0 + 2.2, 0.0, 0.0, 0.0, 0.0])


def control_torque(time_s: float, params: FlexibleShaftParams) -> np.ndarray:
    """Return shoulder/wrist generalized torque with an optional killswitch."""
    if params.torque_cut_time_s is not None and time_s >= params.torque_cut_time_s:
        return np.zeros(3)
    wrist = (
        -abs(params.wrist_restrain_nm)
        if time_s < params.wrist_onset_s
        else params.wrist_drive_nm
    )
    return np.array([params.shoulder_torque_nm, wrist, 0.0])


def _generalized_terms(
    state: np.ndarray, time_s: float, params: FlexibleShaftParams
) -> dict[str, np.ndarray]:
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("state must be a finite six-vector")
    theta1, phi1, phi2, dtheta1, dphi1, dphi2 = state
    triple = params.triple()
    control = control_torque(time_s, params)
    momentum = -coriolis_vector(phi1, phi2, dtheta1, dphi1, dphi2, triple)
    gravity = -gravity_vector(theta1, phi1, phi2, triple)
    joint_damping = np.zeros(3)
    if params.joint_damping_enabled:
        joint_damping[:2] = (
            -params.shoulder_damping_nms_rad * dtheta1,
            -params.wrist_damping_nms_rad * dphi1,
        )
    shaft_elastic = np.array([0.0, 0.0, -params.shaft_stiffness_nm_rad * phi2])
    shaft_damping = np.array([0.0, 0.0, -params.shaft_damping_nms_rad * dphi2])
    return {
        "control": control,
        "momentum": momentum,
        "gravity": gravity,
        "joint_damping": joint_damping,
        "shaft_elastic": shaft_elastic,
        "shaft_damping": shaft_damping,
    }


def acceleration_decomposition(
    state: np.ndarray, time_s: float, params: FlexibleShaftParams
) -> dict[str, np.ndarray]:
    """Return additive generalized-acceleration contributions and their sum."""
    terms = _generalized_terms(state, time_s, params)
    matrix = mass_matrix(state[1], state[2], params.triple())
    contributions = {
        name: np.linalg.solve(matrix, torque) for name, torque in terms.items()
    }
    contributions["total"] = sum(contributions.values())
    return contributions


def _flexible_rhs(
    state: np.ndarray, time_s: float, params: FlexibleShaftParams
) -> np.ndarray:
    acceleration = acceleration_decomposition(state, time_s, params)["total"]
    return np.concatenate((state[3:], acceleration))


def _rk4_step(
    rhs: Any, state: np.ndarray, time_s: float, dt_s: float, params: FlexibleShaftParams
) -> np.ndarray:
    k1 = rhs(state, time_s, params)
    k2 = rhs(state + 0.5 * dt_s * k1, time_s + 0.5 * dt_s, params)
    k3 = rhs(state + 0.5 * dt_s * k2, time_s + 0.5 * dt_s, params)
    k4 = rhs(state + dt_s * k3, time_s + dt_s, params)
    return state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout_flexible(
    params: FlexibleShaftParams,
    *,
    initial_state: np.ndarray | None = None,
    horizon_s: float = 0.36,
    dt_s: float = 0.0005,
) -> FlexibleTrace:
    """Integrate the three-coordinate flexible model with fixed-step RK4."""
    if horizon_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("horizon_s and dt_s must be positive")
    state0 = (
        default_initial_state()
        if initial_state is None
        else np.asarray(initial_state, dtype=float)
    )
    if state0.shape != (6,) or not np.all(np.isfinite(state0)):
        raise ValueError("initial_state must be a finite six-vector")
    count = int(round(horizon_s / dt_s)) + 1
    time = np.arange(count, dtype=float) * dt_s
    state = np.empty((count, 6))
    state[0] = state0
    for index in range(count - 1):
        state[index + 1] = _rk4_step(
            _flexible_rhs, state[index], time[index], dt_s, params
        )
    names = (
        "control",
        "momentum",
        "gravity",
        "joint_damping",
        "shaft_elastic",
        "shaft_damping",
        "total",
    )
    contributions = {name: np.empty((count, 3)) for name in names}
    controls = np.empty((count, 3))
    for index, (time_s, sample) in enumerate(zip(time, state, strict=True)):
        decomposition = acceleration_decomposition(sample, time_s, params)
        for name in names:
            contributions[name][index] = decomposition[name]
        controls[index] = control_torque(time_s, params)
    return FlexibleTrace(
        t=time,
        state=state,
        qddot=contributions["total"],
        controls=controls,
        contributions=contributions,
        rigid=False,
    )


def _rigid_acceleration_decomposition(
    state: np.ndarray, time_s: float, params: FlexibleShaftParams
) -> dict[str, np.ndarray]:
    if state.shape != (4,) or not np.all(np.isfinite(state)):
        raise ValueError("rigid state must be a finite four-vector")
    embedded = np.array([state[0], state[1], 0.0, state[2], state[3], 0.0])
    terms = _generalized_terms(embedded, time_s, params)
    matrix = mass_matrix(state[1], 0.0, params.triple())[:2, :2]
    contributions = {
        name: np.linalg.solve(matrix, torque[:2])
        for name, torque in terms.items()
        if name not in ("shaft_elastic", "shaft_damping")
    }
    contributions["shaft_elastic"] = np.zeros(2)
    contributions["shaft_damping"] = np.zeros(2)
    contributions["total"] = sum(contributions.values())
    return contributions


def _rigid_rhs(
    state: np.ndarray, time_s: float, params: FlexibleShaftParams
) -> np.ndarray:
    acceleration = _rigid_acceleration_decomposition(state, time_s, params)["total"]
    return np.concatenate((state[2:], acceleration))


def rollout_rigid(
    params: FlexibleShaftParams,
    *,
    initial_state: np.ndarray | None = None,
    horizon_s: float = 0.36,
    dt_s: float = 0.0005,
) -> FlexibleTrace:
    """Integrate the exact locked-flex coordinate reduction."""
    flexible_initial = (
        default_initial_state()
        if initial_state is None
        else np.asarray(initial_state, dtype=float)
    )
    if flexible_initial.shape != (6,) or not np.all(np.isfinite(flexible_initial)):
        raise ValueError("initial_state must be a finite six-vector")
    state0 = flexible_initial[[0, 1, 3, 4]]
    count = int(round(horizon_s / dt_s)) + 1
    time = np.arange(count, dtype=float) * dt_s
    reduced = np.empty((count, 4))
    reduced[0] = state0
    for index in range(count - 1):
        reduced[index + 1] = _rk4_step(
            _rigid_rhs, reduced[index], time[index], dt_s, params
        )
    state = np.column_stack(
        (reduced[:, :2], np.zeros(count), reduced[:, 2:], np.zeros(count))
    )
    names = (
        "control",
        "momentum",
        "gravity",
        "joint_damping",
        "shaft_elastic",
        "shaft_damping",
        "total",
    )
    contributions = {name: np.zeros((count, 3)) for name in names}
    controls = np.empty((count, 3))
    for index, (time_s, sample) in enumerate(zip(time, reduced, strict=True)):
        decomposition = _rigid_acceleration_decomposition(sample, time_s, params)
        for name in names:
            contributions[name][index, :2] = decomposition[name]
        controls[index] = control_torque(time_s, params)
    return FlexibleTrace(
        t=time,
        state=state,
        qddot=contributions["total"],
        controls=controls,
        contributions=contributions,
        rigid=True,
    )


def shaft_energy(flex_rad: float | np.ndarray, stiffness_nm_rad: float) -> Any:
    """Return linear torsional strain energy, ``0.5 k phi^2``."""
    return 0.5 * stiffness_nm_rad * np.asarray(flex_rad) ** 2


def shaft_power_terms(
    flex_rad: float | np.ndarray,
    flex_rate_rad_s: float | np.ndarray,
    stiffness_nm_rad: float,
    damping_nms_rad: float,
) -> tuple[Any, Any]:
    """Return strain-energy rate and nonpositive damping power."""
    flex = np.asarray(flex_rad)
    rate = np.asarray(flex_rate_rad_s)
    return stiffness_nm_rad * flex * rate, -damping_nms_rad * rate**2


def mechanical_energy(state: np.ndarray, params: FlexibleShaftParams) -> float:
    """Return kinetic, gravitational, and shaft-strain energy."""
    checked = np.asarray(state, dtype=float)
    if checked.shape != (6,):
        raise ValueError("state must have shape (6,)")
    triple = params.triple()
    return float(
        kinetic_energy(checked, triple)
        + potential_energy(checked, triple)
        + shaft_energy(checked[2], params.shaft_stiffness_nm_rad)
    )


def trace_kinematics(
    trace: FlexibleTrace, params: FlexibleShaftParams
) -> dict[str, np.ndarray]:
    """Return joint positions, clubhead velocity, and clubhead speed."""
    triple = params.triple()
    positions = {
        name: np.empty((trace.t.size, 2)) for name in ("wrist1", "wrist2", "tip")
    }
    for index, sample in enumerate(trace.state):
        pose = forward_kinematics(sample[0], sample[1], sample[2], triple)
        for name in positions:
            positions[name][index] = pose[name]
    velocity = np.gradient(positions["tip"], trace.t, axis=0, edge_order=2)
    return {
        **positions,
        "tip_velocity": velocity,
        "tip_speed": np.linalg.norm(velocity, axis=1),
    }


def shaft_interface_force(
    trace: FlexibleTrace, params: FlexibleShaftParams
) -> np.ndarray:
    """Return the proximal-on-distal reaction force at the shaft flex joint."""
    triple = params.triple()
    result = np.empty((trace.t.size, 2))
    for index, sample in enumerate(trace.state):
        forces = net_joint_forces(sample, trace.qddot[index], triple)
        result[index] = forces["wrist2"]
    return result


def energy_accounting(
    trace: FlexibleTrace, params: FlexibleShaftParams
) -> dict[str, np.ndarray]:
    """Return energy, power, work, and first-law residual arrays."""
    kinetic = np.array(
        [kinetic_energy(sample, params.triple()) for sample in trace.state]
    )
    potential = np.array(
        [potential_energy(sample, params.triple()) for sample in trace.state]
    )
    strain = shaft_energy(trace.state[:, 2], params.shaft_stiffness_nm_rad)
    velocity = trace.state[:, 3:]
    control_power = np.sum(trace.controls * velocity, axis=1)
    joint_damping_power = np.zeros(trace.t.size)
    if params.joint_damping_enabled:
        joint_damping_power = -(
            params.shoulder_damping_nms_rad * velocity[:, 0] ** 2
            + params.wrist_damping_nms_rad * velocity[:, 1] ** 2
        )
    _, shaft_damping_power = shaft_power_terms(
        trace.state[:, 2],
        velocity[:, 2],
        params.shaft_stiffness_nm_rad,
        0.0 if trace.rigid else params.shaft_damping_nms_rad,
    )
    total_energy = kinetic + potential + strain
    energy_rate = np.gradient(total_energy, trace.t, edge_order=2)
    supplied_power = control_power + joint_damping_power + shaft_damping_power
    residual = energy_rate - supplied_power
    cumulative_work = np.zeros(trace.t.size)
    cumulative_work[1:] = np.cumsum(
        0.5 * (supplied_power[1:] + supplied_power[:-1]) * np.diff(trace.t)
    )
    return {
        "kinetic_energy": kinetic,
        "potential_energy": potential,
        "shaft_strain_energy": strain,
        "total_mechanical_energy": total_energy,
        "control_power": control_power,
        "joint_damping_power": joint_damping_power,
        "shaft_damping_power": shaft_damping_power,
        "supplied_power": supplied_power,
        "energy_rate": energy_rate,
        "energy_rate_residual": residual,
        "cumulative_nonconservative_work": cumulative_work,
    }
