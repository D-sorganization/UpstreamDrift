"""The OpenSim ``BushingForce`` law, engine-agnostic (issue #11739, OSV-7).

Every engine that cannot evaluate an OpenSim ``BushingForce`` natively
(MuJoCo through ``mjcb_passive``, Pinocchio through ``aba``) evaluates this
law instead, so that the same-input parity runs compare integrators and
kinematics, not force laws.

Definitions (Simbody ``LinearBushing`` / OpenSim ``TwoFrameLinker``).  Frame
``F1`` sits on the hand (frame1, prescribed), frame ``F2`` on the club
(frame2).  World poses ``(R_i, p_i)`` with ``R_i`` world <- frame; ``v_i`` is
the world velocity of the frame origin and ``w_i`` the world angular velocity.

* Rotational deflection ``theta``: the body-fixed X-Y-Z angles of
  ``R_12 = R_1^T R_2``, i.e. ``R_12 = Rx(t0) Ry(t1) Rz(t2)``.
* Translational deflection ``delta = R_1^T (p_2 - p_1)`` (frame-1 axes).
* Rates: ``delta_dot = R_1^T (v_2 - v_1 - w_1 x (p_2 - p_1))`` (the time
  derivative of the frame-1 components) and ``theta_dot = N(theta) w_rel``
  with ``w_rel = R_2^T (w_2 - w_1)``, the angular velocity of F2 in F1
  expressed in F2, and ``N`` the body-fixed X-Y-Z kinematic matrix.
* Generalised forces ``f_theta = -(K_r theta + C_r theta_dot)`` and
  ``f_delta = -(K_t delta + C_t delta_dot)`` per axis.
* Wrench ON FRAME 2 (the club; hand-on-club sign), applied at the frame-2
  origin: force ``R_1 f_delta`` and moment ``R_2 N(theta)^T f_theta``.  The
  transpose of ``N`` maps the angle-space generalised force to a physical
  moment by virtual power, ``f_theta . theta_dot = (N^T f_theta) . w_rel``.

The law is singular at ``|t1| = pi/2`` (gimbal lock of the X-Y-Z sequence);
the grip deflections are below one degree, so a deflection that comes within
``GIMBAL_MARGIN_RAD`` of the singularity is a precondition violation, not a
silent large force.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.shared.python.grip_contact.parameters import BushingParameters

#: Refuse middle angles within this margin of the X-Y-Z gimbal lock (rad).
GIMBAL_MARGIN_RAD = 1e-3
_EYE = np.eye(3)

__all__ = [
    "GIMBAL_MARGIN_RAD",
    "BushingState",
    "BushingWrench",
    "body_xyz_angles",
    "body_xyz_n_matrix",
    "bushing_wrench",
    "cross3",
]


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a x b`` for two 3-vectors (``np.cross`` is ~50x slower per call).

    The law runs inside every integrator stage of every engine, so its
    per-call cost matters.
    """
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


@dataclass(frozen=True)
class BushingState:
    """World pose and velocity of one bushing frame.

    ``rotation`` is world <- frame, ``position_m`` the frame origin,
    ``velocity_m_s`` the velocity of that origin and ``omega_rad_s`` the
    angular velocity, all in world axes.
    """

    rotation: np.ndarray
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    omega_rad_s: np.ndarray

    def __post_init__(self) -> None:
        rot = np.asarray(self.rotation, dtype=float)
        if rot.shape != (3, 3) or not np.isfinite(rot).all():
            raise ValueError("rotation must be a finite 3x3 matrix")
        if not np.abs(rot.T @ rot - _EYE).max() <= 1e-6:
            raise ValueError("rotation must be orthonormal")
        object.__setattr__(self, "rotation", rot)
        for name in ("position_m", "velocity_m_s", "omega_rad_s"):
            vec = np.asarray(getattr(self, name), dtype=float)
            if vec.shape != (3,) or not np.isfinite(vec).all():
                raise ValueError(f"{name} must be a finite 3-vector")
            object.__setattr__(self, name, vec)


@dataclass(frozen=True)
class BushingWrench:
    """Wrench of one bushing on the club frame (F2), world axes.

    ``moment_nm`` is the moment about the frame-2 origin (the grip point on
    the club), so it is the free torque of the hand at that point.
    """

    force_n: np.ndarray
    moment_nm: np.ndarray
    angles_rad: np.ndarray  # body-fixed X-Y-Z angles of R_1^T R_2
    translation_m: np.ndarray  # frame-1 axes
    angle_rates_rad_s: np.ndarray
    translation_rate_m_s: np.ndarray


def body_xyz_angles(rotation: np.ndarray) -> np.ndarray:
    """Body-fixed X-Y-Z angles ``(t0, t1, t2)`` with ``R = Rx Ry Rz``.

    Raises:
        ValueError: when the middle angle is within ``GIMBAL_MARGIN_RAD`` of
            ``+-pi/2``, where the sequence is singular.
    """
    r = np.asarray(rotation, dtype=float)
    t1 = math.atan2(r[0, 2], math.hypot(r[0, 0], r[0, 1]))
    if abs(t1) > 0.5 * math.pi - GIMBAL_MARGIN_RAD:
        raise ValueError(f"X-Y-Z angles are singular at t1 = {t1:.6f} rad")
    t0 = math.atan2(-r[1, 2], r[2, 2])
    t2 = math.atan2(-r[0, 1], r[0, 0])
    return np.array([t0, t1, t2])


def body_xyz_n_matrix(angles_rad: np.ndarray) -> np.ndarray:
    """``N`` with ``theta_dot = N(theta) w_B`` (``w_B`` in the moving frame).

    Raises:
        ValueError: at the gimbal lock (see :func:`body_xyz_angles`).
    """
    _, t1, t2 = (float(a) for a in angles_rad)
    c1 = math.cos(t1)
    if abs(c1) < math.sin(GIMBAL_MARGIN_RAD):
        raise ValueError("N is singular at the X-Y-Z gimbal lock")
    s1, s2, c2 = math.sin(t1), math.sin(t2), math.cos(t2)
    return np.array(
        [
            [c2 / c1, -s2 / c1, 0.0],
            [s2, c2, 0.0],
            [-s1 * c2 / c1, s1 * s2 / c1, 1.0],
        ]
    )


def bushing_wrench(
    params: BushingParameters, hand: BushingState, club: BushingState
) -> BushingWrench:
    """Wrench of the bushing ``params`` on the club frame (see module docstring).

    Preconditions: both states are valid :class:`BushingState` objects and
    the relative rotation is away from the X-Y-Z gimbal lock.
    Postcondition: with zero rates the force equals ``-R_1 K_t delta`` exactly.

    Raises:
        TypeError: if ``params`` is not a :class:`BushingParameters`.
        ValueError: at the gimbal lock.
    """
    if not isinstance(params, BushingParameters):
        raise TypeError("params must be BushingParameters")
    r1, r2 = hand.rotation, club.rotation
    arm = club.position_m - hand.position_m
    translation = r1.T @ arm
    translation_rate = r1.T @ (
        club.velocity_m_s - hand.velocity_m_s - cross3(hand.omega_rad_s, arm)
    )
    angles = body_xyz_angles(r1.T @ r2)
    n_matrix = body_xyz_n_matrix(angles)
    angle_rates = n_matrix @ (r2.T @ (club.omega_rad_s - hand.omega_rad_s))
    f_delta = -(
        np.asarray(params.translational_stiffness_n_m) * translation
        + np.asarray(params.translational_damping_ns_m) * translation_rate
    )
    f_theta = -(
        np.asarray(params.rotational_stiffness_nm_rad) * angles
        + np.asarray(params.rotational_damping_nms_rad) * angle_rates
    )
    return BushingWrench(
        force_n=r1 @ f_delta,
        moment_nm=r2 @ (n_matrix.T @ f_theta),
        angles_rad=angles,
        translation_m=translation,
        angle_rates_rad_s=angle_rates,
        translation_rate_m_s=translation_rate,
    )
