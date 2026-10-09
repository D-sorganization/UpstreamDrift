"""Same-input prescribed motion for the grip-kinetics parity (issue #11739).

The OpenSim reference prescribes every body coordinate with a ``SimmSpline``
through the 2 ms fixture samples; its coordinate speeds are the spline
derivative.  For the MuJoCo, Drake and Pinocchio runs to see the *same*
input, they use :class:`CoordinateSpline`, a vectorised implementation of the
same Forsythe-Malcolm-Moler cubic spline (end conditions: the third
derivative at each end matches that of the cubic through the first or last
four samples), which reproduces ``SimmSpline`` to round-off
(``test_prescribed_motion.py`` checks it against OpenSim).

Each engine then computes the club pose and spatial velocity of its own
rigid-weld model from these coordinates (its own forward kinematics); the hand
bushing frames are the grip frames carried by that weld pose
(:func:`hand_frame_states`), exactly where the OpenSim bushing model puts them
(``full_body_grip_topology.build_bushing_spec`` and the right frame tied to
the left hand body).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.grip_contact.bushing_law import BushingState, cross3
from src.shared.python.grip_contact.interface import GripInterface

__all__ = [
    "CoordinateSpline",
    "RigidBodyState",
    "fmm_spline_coefficients",
    "hand_frame_states",
]


def fmm_spline_coefficients(
    time_s: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forsythe-Malcolm-Moler cubic spline coefficients, column-wise.

    ``s(t) = y_i + b_i dt + c_i dt^2 + d_i dt^3`` on ``[t_i, t_i+1)``.

    Preconditions: ``time_s`` strictly increasing with at least four samples,
    ``values`` finite with shape ``(n,)`` or ``(n, m)``.

    Raises:
        ValueError: on a violated precondition.
    """
    x = np.asarray(time_s, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 4 or not np.all(np.diff(x) > 0.0):
        raise ValueError("time_s must be strictly increasing with >= 4 samples")
    if y.shape[0] != x.size or not np.all(np.isfinite(y)):
        raise ValueError("values must be finite with one row per time sample")
    y2 = y.reshape(x.size, -1)
    n, last = x.size, x.size - 1
    h = np.diff(x)[:, None]
    b = np.zeros_like(y2)
    c = np.zeros_like(y2)
    d = np.zeros_like(y2)
    d[:last] = h
    c[1] = (y2[1] - y2[0]) / h[0]
    for i in range(1, last):
        b[i] = 2.0 * (d[i - 1] + d[i])
        c[i + 1] = (y2[i + 1] - y2[i]) / h[i]
        c[i] = c[i + 1] - c[i]
    b[0], b[last] = -d[0], -d[n - 2]
    c0 = c[2] / (x[3] - x[1]) - c[1] / (x[2] - x[0])
    cl = c[n - 2] / (x[last] - x[n - 3]) - c[n - 3] / (x[n - 2] - x[n - 4])
    c[0] = c0 * d[0] * d[0] / (x[3] - x[0])
    c[last] = -cl * d[n - 2] * d[n - 2] / (x[last] - x[n - 4])
    for i in range(1, n):
        t = d[i - 1] / b[i - 1]
        b[i] = b[i] - t * d[i - 1]
        c[i] = c[i] - t * c[i - 1]
    c[last] = c[last] / b[last]
    for i in range(n - 2, -1, -1):
        c[i] = (c[i] - d[i] * c[i + 1]) / b[i]
    b[last] = (y2[last] - y2[n - 2]) / d[n - 2] + d[n - 2] * (c[n - 2] + 2 * c[last])
    hb = d[:last].copy()
    b[:last] = (y2[1:] - y2[:last]) / hb - hb * (c[1:] + 2.0 * c[:last])
    d[:last] = (c[1:] - c[:last]) / hb
    c *= 3.0
    d[last] = d[n - 2]
    shape = y.shape
    return b.reshape(shape), c.reshape(shape), d.reshape(shape)


class CoordinateSpline:
    """``SimmSpline``-equivalent interpolation of every coordinate column.

    Args:
        time_s: strictly increasing sample times ``(n,)``.
        q: coordinates ``(n, m)``.
    """

    def __init__(self, time_s: np.ndarray, q: np.ndarray) -> None:
        self.time_s = np.asarray(time_s, dtype=float)
        self.q = np.asarray(q, dtype=float)
        if self.q.ndim != 2:
            raise ValueError("q must have shape (n, m)")
        self._b, self._c, self._d = fmm_spline_coefficients(self.time_s, self.q)

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates and their time derivatives at ``t`` (clamped interval).

        Raises:
            ValueError: if ``t`` is not finite.
        """
        if not np.isfinite(t):
            raise ValueError("t must be finite")
        x = self.time_s
        i = int(np.clip(np.searchsorted(x, t, side="right") - 1, 0, x.size - 2))
        dt = float(t) - x[i]
        b, c, d = self._b[i], self._c[i], self._d[i]
        value = self.q[i] + dt * (b + dt * (c + dt * d))
        rate = b + dt * (2.0 * c + 3.0 * dt * d)
        return value, rate


@dataclass(frozen=True)
class RigidBodyState:
    """World pose and spatial velocity of a body frame (origin velocity)."""

    rotation: np.ndarray  # world <- body
    position_m: np.ndarray
    velocity_m_s: np.ndarray  # of the body-frame origin
    omega_rad_s: np.ndarray

    def frame(self, offset: np.ndarray) -> BushingState:
        """State of the frame fixed on this body at the 4x4 ``offset``."""
        off = np.asarray(offset, dtype=float)
        if off.shape != (4, 4):
            raise ValueError("offset must be a 4x4 homogeneous transform")
        arm = self.rotation @ off[:3, 3]
        return BushingState(
            rotation=self.rotation @ off[:3, :3],
            position_m=self.position_m + arm,
            velocity_m_s=self.velocity_m_s + cross3(self.omega_rad_s, arm),
            omega_rad_s=self.omega_rad_s,
        )


def hand_frame_states(
    weld_club: RigidBodyState, interface: GripInterface
) -> dict[str, BushingState]:
    """Hand bushing frames (``'L'``, ``'R'``) for the weld-model club state.

    In the bushing topology both hand frames are fixed on the left hand body
    where the weld model's grip frames sit, so in world they are the grip
    frames carried by the weld club pose.
    """
    return {side: weld_club.frame(interface.frame(side).matrix()) for side in "LR"}
