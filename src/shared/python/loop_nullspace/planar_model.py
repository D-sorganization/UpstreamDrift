"""Independent numpy model of the NSX-00 planar two-arm golfer (NSX-12).

This is a second derivation of the model whose sympy form lives in
``docs/research/loop_nullspace/derivations/planar_golfer.py``: the Jacobians are
written out in closed form (no symbolic engine, no finite differences) so the
goldens can be reproduced by an independent code path.  It is a test fixture,
not a golfer model.

Coordinates ``q = (lead shoulder, lead elbow, club angle, trail shoulder, trail
elbow)``; every segment of absolute angle ``a`` points along ``u(a) =
(sin a, -cos a)`` and ``d u / d a = t(a) = (cos a, sin a)``.  The trail hand
closes onto the club grip point: ``Phi(q) = H_R - G``.

A body point built from segments ``(l_k, A_k)`` (length and the set of
coordinates summing to the segment's absolute angle) has Jacobian column
``sum_{k : j in A_k} l_k t(a_k)`` and drift ``-sum_k l_k w_k^2 u(a_k)`` with
``w_k = sum_{j in A_k} v_j``.  For ``M = sum m J^T J + I Jw^T Jw`` the velocity
product is ``h = sum m J^T Jdot v + grad V``.

Units: m, kg, s, rad; gravity acts along ``-y``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from src.shared.python.loop_nullspace.fixtures import GoldenValues

Array = NDArray[np.float64]
N_Q = 5
# (length attribute, coordinate set) segments of each chain.
_LEAD_UPPER = (0,)
_LEAD_FORE = (0, 1)
_CLUB = (0, 1, 2)
_TRAIL_UPPER = (3,)
_TRAIL_FORE = (3, 4)


@dataclass(frozen=True)
class PlanarGolferParams:
    """Parameters of the planar golfer (floats parsed from NSX-00 rationals)."""

    shoulder_separation: float
    upper_arm_length: float
    forearm_length: float
    upper_arm_mass: float
    forearm_mass: float
    club_mass: float
    club_length: float
    club_com_from_lead_hand: float
    trail_grip_from_lead_hand: float
    gravity: float

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not np.isfinite(value):
                raise ValueError(f"{field.name} must be finite, got {value}")
            if field.name != "gravity" and value <= 0.0:
                raise ValueError(f"{field.name} must be positive, got {value}")

    @classmethod
    def from_golden(cls, golden: GoldenValues) -> PlanarGolferParams:
        """Parse the exact-rational parameter strings of an NSX-00 golden."""
        raw = golden.payload["model"]["parameters"]
        table = {
            "shoulder_separation": "shoulder_separation_m",
            "upper_arm_length": "upper_arm_length_m",
            "forearm_length": "forearm_length_m",
            "upper_arm_mass": "upper_arm_mass_kg",
            "forearm_mass": "forearm_mass_kg",
            "club_mass": "club_mass_kg",
            "club_length": "club_length_m",
            "club_com_from_lead_hand": "club_com_from_lead_hand_m",
            "trail_grip_from_lead_hand": "trail_grip_from_lead_hand_m",
            "gravity": "gravity_m_s2",
        }
        return cls(**{k: float(Fraction(raw[v])) for k, v in table.items()})


def _u(angle: float) -> Array:
    return np.array([np.sin(angle), -np.cos(angle)])


def _t(angle: float) -> Array:
    return np.array([np.cos(angle), np.sin(angle)])


class _Kinematics:
    """Absolute angles and angular rates of the five chains for one state."""

    def __init__(self, q: Array, v: Array) -> None:
        self.q, self.v = q, v

    def angle(self, deps: tuple[int, ...]) -> float:
        return float(sum(self.q[j] for j in deps))

    def rate(self, deps: tuple[int, ...]) -> float:
        return float(sum(self.v[j] for j in deps))

    def point(
        self, base: Array, segments: list[tuple[float, tuple[int, ...]]]
    ) -> tuple[Array, Array, Array]:
        """Return ``(position, Jacobian (2, 5), Jdot v (2,))`` of a chain end."""
        pos, jac, drift = base.astype(float).copy(), np.zeros((2, N_Q)), np.zeros(2)
        for length, deps in segments:
            ang = self.angle(deps)
            pos += length * _u(ang)
            for j in deps:
                jac[:, j] += length * _t(ang)
            drift -= length * self.rate(deps) ** 2 * _u(ang)
        return pos, jac, drift


def planar_dynamics(q: Array, v: Array, params: PlanarGolferParams) -> dict[str, Array]:
    """Evaluate ``M, h, phi, Jc, Jdot_v`` of the planar golfer.

    Preconditions:
        ``q`` and ``v`` are finite vectors of shape ``(5,)``.

    Postconditions:
        ``M`` is exactly symmetric positive definite; ``Jc = d phi / d q``;
        ``Jdot_v = d/dt (Jc) v`` for the given ``v``; ``h`` includes gravity.
    """
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    for name, vec in (("q", q), ("v", v)):
        if vec.shape != (N_Q,) or not np.all(np.isfinite(vec)):
            raise ValueError(f"{name} must have shape ({N_Q},) and be finite")
    p = params
    kin = _Kinematics(q, v)
    s_l = np.array([-p.shoulder_separation / 2.0, 0.0])
    s_r = np.array([p.shoulder_separation / 2.0, 0.0])
    lu, lf = p.upper_arm_length, p.forearm_length
    upper_l: list[tuple[float, tuple[int, ...]]] = [(lu, _LEAD_UPPER)]
    fore_l = [*upper_l, (lf, _LEAD_FORE)]
    upper_r: list[tuple[float, tuple[int, ...]]] = [(lu, _TRAIL_UPPER)]
    fore_r = [*upper_r, (lf, _TRAIL_FORE)]
    # (mass, inertia about COM, base, segments to the COM, angle coordinates)
    bodies: list[tuple[float, float, Array, list[tuple[float, tuple[int, ...]]]]] = [
        (p.upper_arm_mass, p.upper_arm_mass * lu**2 / 12, s_l, [(lu / 2, _LEAD_UPPER)]),
        (
            p.forearm_mass,
            p.forearm_mass * lf**2 / 12,
            s_l,
            [*upper_l, (lf / 2, _LEAD_FORE)],
        ),
        (
            p.club_mass,
            p.club_mass * p.club_length**2 / 12,
            s_l,
            [*fore_l, (p.club_com_from_lead_hand, _CLUB)],
        ),
        (
            p.upper_arm_mass,
            p.upper_arm_mass * lu**2 / 12,
            s_r,
            [(lu / 2, _TRAIL_UPPER)],
        ),
        (
            p.forearm_mass,
            p.forearm_mass * lf**2 / 12,
            s_r,
            [*upper_r, (lf / 2, _TRAIL_FORE)],
        ),
    ]
    angle_sets = [_LEAD_UPPER, _LEAD_FORE, _CLUB, _TRAIL_UPPER, _TRAIL_FORE]
    mass = np.zeros((N_Q, N_Q))
    h = np.zeros(N_Q)
    for (m_i, i_i, base, segs), deps in zip(bodies, angle_sets, strict=True):
        _, jv, drift = kin.point(base, segs)
        jw = np.zeros(N_Q)
        jw[list(deps)] = 1.0
        mass += m_i * jv.T @ jv + i_i * np.outer(jw, jw)
        h += m_i * jv.T @ drift + m_i * p.gravity * jv[1]
    pos_hr, j_hr, d_hr = kin.point(s_r, fore_r)
    grip_segments = [*fore_l, (p.trail_grip_from_lead_hand, _CLUB)]
    pos_g, j_g, d_g = kin.point(s_l, grip_segments)
    out = {
        "M": 0.5 * (mass + mass.T),
        "h": h,
        "phi": pos_hr - pos_g,
        "Jc": j_hr - j_g,
        "Jdot_v": d_hr - d_g,
        "grip": pos_g,
    }
    return out
