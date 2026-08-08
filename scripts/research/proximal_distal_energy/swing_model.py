"""Kinematics and energetics post-processing for double-pendulum swings.

Conventions (matching the ODE backend / ``DoublePendulumDynamics``):

- ``theta1`` is the arm (upper-segment) angle measured from the downward
  vertical in the inclined swing plane; ``theta2`` is the wrist angle of the
  club relative to the arm. The club's absolute angle is
  ``theta_club = theta1 + theta2``.
- Plane coordinates: ``x`` to the target side, ``y`` up; the projected
  gravity vector is ``(0, -g_proj)`` with
  ``g_proj = g * cos(plane_inclination)`` as computed by
  ``DoublePendulumParameters.projected_gravity``.
- A point at angle ``theta`` on a unit segment sits at
  ``u(theta) = (sin(theta), -cos(theta))``; its velocity direction is
  ``u'(theta) = (cos(theta), sin(theta))`` times the angular velocity.

All functions are pure NumPy over trace arrays ``(T,)`` / ``(T, 2)`` so
they can post-process any backend's ``Trace`` without touching engine
internals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.simulation_backends import GolfModelParams


@dataclass(frozen=True)
class PlanarInertials:
    """Composite planar inertial properties used by the energy accounting."""

    l1: float
    m1: float
    lc1: float
    i1_com: float
    i1_pivot: float
    l2: float
    m2: float
    lc2: float
    i2_com: float
    g_proj: float
    damping_shoulder: float
    damping_wrist: float

    @classmethod
    def from_params(cls, params: GolfModelParams) -> PlanarInertials:
        """Extract inertial constants from the shared model parameters."""
        dp = params.to_double_pendulum_parameters()
        upper = dp.upper_segment
        lower = dp.lower_segment
        return cls(
            l1=upper.length_m,
            m1=upper.mass_kg,
            lc1=upper.center_of_mass_distance,
            i1_com=upper.inertia_about_com,
            i1_pivot=upper.inertia_about_proximal_joint,
            l2=lower.length_m,
            m2=lower.total_mass,
            lc2=lower.center_of_mass_distance,
            i2_com=lower.inertia_about_com,
            g_proj=dp.projected_gravity,
            damping_shoulder=dp.damping_shoulder,
            damping_wrist=dp.damping_wrist,
        )


def _unit(theta: np.ndarray) -> np.ndarray:
    """Position direction of a point at angle ``theta`` (from down-vertical)."""
    return np.stack([np.sin(theta), -np.cos(theta)], axis=-1)


def _unit_perp(theta: np.ndarray) -> np.ndarray:
    """Velocity direction (d/dtheta of :func:`_unit`)."""
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


def clubhead_speed(
    inertials: PlanarInertials, q: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Clubhead (distal tip) speed magnitude [m/s] at every sample."""
    theta1 = q[:, 0]
    theta_club = q[:, 0] + q[:, 1]
    omega1 = v[:, 0]
    omega_club = v[:, 0] + v[:, 1]
    vel = inertials.l1 * omega1[:, None] * _unit_perp(
        theta1
    ) + inertials.l2 * omega_club[:, None] * _unit_perp(theta_club)
    return np.linalg.norm(vel, axis=1)


def hand_velocity(
    inertials: PlanarInertials, q: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Velocity of the wrist joint (arm tip), shape ``(T, 2)``."""
    return inertials.l1 * v[:, 0:1] * _unit_perp(q[:, 0])


def club_com_position(inertials: PlanarInertials, q: np.ndarray) -> np.ndarray:
    """Club-segment centre-of-mass position in plane coordinates ``(T, 2)``."""
    theta1 = q[:, 0]
    theta_club = q[:, 0] + q[:, 1]
    return inertials.l1 * _unit(theta1) + inertials.lc2 * _unit(theta_club)


def club_com_velocity(
    inertials: PlanarInertials, q: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Club-segment centre-of-mass velocity ``(T, 2)``."""
    theta1 = q[:, 0]
    theta_club = q[:, 0] + q[:, 1]
    omega1 = v[:, 0]
    omega_club = v[:, 0] + v[:, 1]
    return inertials.l1 * omega1[:, None] * _unit_perp(
        theta1
    ) + inertials.lc2 * omega_club[:, None] * _unit_perp(theta_club)


def segment_kinetic_energies(
    inertials: PlanarInertials, q: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Kinetic energy of the arm and club segments [J], shapes ``(T,)``.

    The arm rotates about the fixed hub, so its kinetic energy is
    ``0.5 * I1_pivot * omega1**2``. The club's energy combines its
    centre-of-mass translation and rotation about the centre of mass with
    the *absolute* angular velocity ``omega1 + omega2``.
    """
    omega1 = v[:, 0]
    omega_club = v[:, 0] + v[:, 1]
    e_arm = 0.5 * inertials.i1_pivot * omega1**2
    v_com = club_com_velocity(inertials, q, v)
    e_club = 0.5 * inertials.m2 * np.sum(v_com**2, axis=1) + (
        0.5 * inertials.i2_com * omega_club**2
    )
    return e_arm, e_club


def segment_potential_energies(
    inertials: PlanarInertials, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """In-plane gravitational potential energy of arm and club [J].

    Zero reference is the hub height; the projected gravity magnitude is
    used because motion is constrained to the inclined plane.
    """
    theta1 = q[:, 0]
    y_arm = -inertials.lc1 * np.cos(theta1)
    e_arm = inertials.m1 * inertials.g_proj * y_arm
    y_club = club_com_position(inertials, q)[:, 1]
    e_club = inertials.m2 * inertials.g_proj * y_club
    return e_arm, e_club


def wrist_interface_powers(
    inertials: PlanarInertials,
    t: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    u: np.ndarray,
) -> dict[str, np.ndarray]:
    """Robertson-Winter style power accounting at the wrist interface.

    Returns a dict of ``(T,)`` arrays:

    - ``joint_force_power``: joint reaction force on the club, dotted with
      the wrist-joint velocity (energy carried through the joint by the
      force). The club centre-of-mass acceleration is obtained by central
      finite differences of the analytic centre-of-mass velocity.
    - ``moment_power_on_club``: net wrist moment acting on the club
      (applied wrist torque minus wrist damping) times the club's absolute
      angular velocity.
    - ``muscle_moment_power``: applied wrist torque times the *relative*
      wrist angular velocity (mechanical power generated by the wrist
      actuator).
    - ``gravity_power_on_club``: projected-gravity power on the club mass.
    - ``club_energy_rate``: finite-difference rate of the club's total
      mechanical (kinetic + potential) energy, for residual validation:
      ``club_energy_rate ~= joint_force_power + moment_power_on_club +
      gravity_power_on_club`` (gravity excluded when using mechanical
      energy; included when using kinetic energy only — here the club
      energy is kinetic + potential, so gravity power is *not* added).
    """
    omega_rel = v[:, 1]
    omega_club = v[:, 0] + v[:, 1]
    tau_wrist = u[:, 1]
    net_wrist_moment = tau_wrist - inertials.damping_wrist * omega_rel

    v_com = club_com_velocity(inertials, q, v)
    a_com = np.gradient(v_com, t, axis=0)
    g_vec = np.array([0.0, -inertials.g_proj])
    joint_force = inertials.m2 * a_com - inertials.m2 * g_vec[None, :]
    v_joint = hand_velocity(inertials, q, v)
    joint_force_power = np.sum(joint_force * v_joint, axis=1)

    moment_power_on_club = net_wrist_moment * omega_club
    muscle_moment_power = tau_wrist * omega_rel
    gravity_power_on_club = inertials.m2 * (v_com @ g_vec)

    e_kin_arm, e_kin_club = segment_kinetic_energies(inertials, q, v)
    _, e_pot_club = segment_potential_energies(inertials, q)
    del e_kin_arm
    club_energy_rate = np.gradient(e_kin_club + e_pot_club, t)

    return {
        "joint_force_power": joint_force_power,
        "moment_power_on_club": moment_power_on_club,
        "muscle_moment_power": muscle_moment_power,
        "gravity_power_on_club": gravity_power_on_club,
        "club_energy_rate": club_energy_rate,
    }


#: Arm angle beyond which a club-vertical crossing no longer represents a
#: first-pass impact: the arm has rotated past the delivery zone and the
#: constant shoulder torque is pumping energy into extra rotations, so the
#: trial is scored invalid rather than rewarded for the longer runway.
MAX_ARM_ANGLE_AT_IMPACT_RAD = 2.0


def find_impact(
    t: np.ndarray, q: np.ndarray, v: np.ndarray, inertials: PlanarInertials
) -> tuple[float, float, float] | None:
    """Return ``(t_impact, clubhead_speed, theta1_at_impact)`` or ``None``.

    Impact is the first upward crossing of the club's absolute angle
    ``theta1 + theta2`` through zero (club pointing straight down in the
    swing plane), linearly interpolated between samples. Swings that never
    reach club-vertical within the horizon, or whose arm has already
    rotated past :data:`MAX_ARM_ANGLE_AT_IMPACT_RAD` when the club gets
    there (not a first-pass delivery), return ``None``.
    """
    theta_club = q[:, 0] + q[:, 1]
    speeds = clubhead_speed(inertials, q, v)
    below = theta_club < 0.0
    crossings = np.nonzero(below[:-1] & ~below[1:])[0]
    if crossings.size == 0:
        return None
    k = int(crossings[0])
    th0, th1 = theta_club[k], theta_club[k + 1]
    frac = 0.0 if th1 == th0 else float(-th0 / (th1 - th0))
    t_imp = float(t[k] + frac * (t[k + 1] - t[k]))
    speed = float(speeds[k] + frac * (speeds[k + 1] - speeds[k]))
    theta1_imp = float(q[k, 0] + frac * (q[k + 1, 0] - q[k, 0]))
    if theta1_imp > MAX_ARM_ANGLE_AT_IMPACT_RAD:
        return None
    return t_imp, speed, theta1_imp
