"""Shared small models for the NSX-84 tangent-linearization tests."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np

GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "loop_nullspace"
    / "goldens"
    / "variability_sensitivity.json"
)


def golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def frac(text: str) -> float:
    return float(Fraction(text))


class Pendulum:
    """Point mass on a circle with two world-frame actuator forces (NSX-80)."""

    m, length, g = 2.0, 1.0, 10.0
    omega = 1.0
    u_particular = np.array([7.0, 5.0])

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        return self.m * np.eye(2)

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.array([0.0, self.m * self.g])

    def closure_jacobian(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(q, dtype=float).reshape(1, 2)

    def closure_drift(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.array([float(np.dot(v, v))])

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        q = self.length * np.array([0.6, -0.8])
        v = self.length * self.omega * np.array([0.8, 0.6])
        return q, v

    def load(self, z: float) -> np.ndarray:
        """Feed-forward ``u(z) = u_p - q z`` of the allocation family."""
        q, _ = self.state()
        return self.u_particular - q * z


def tangent_field(q: np.ndarray) -> np.ndarray:
    """Smooth unit tangent ``t(theta)`` of the circle as an ``(2, 1)`` basis."""
    q = np.asarray(q, dtype=float)
    return (np.array([-q[1], q[0]]) / np.linalg.norm(q)).reshape(2, 1)


class RankDropModel:
    """``J(q) = [q0, 0]``: rank 0 at ``q0 = 0`` and rank 1 anywhere near it."""

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        return np.eye(2)

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.zeros(2)

    def closure_jacobian(self, q: np.ndarray) -> np.ndarray:
        return np.array([[q[0], 0.0]])

    def closure_drift(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.array([0.0])


class FreeFlyer:
    """Rigid body, body-frame velocity ``(v_lin, omega)``, quaternion ``wxyz``.

    ``h = (m (w x vl) + m R^T g_world, w x I w)``; no closure rows.
    """

    m = 3.0
    inertia = np.diag([0.5, 0.7, 1.1])
    gravity = np.array([0.0, 0.0, -9.81])

    @staticmethod
    def rotation(quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat / np.linalg.norm(quat)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        out = np.zeros((6, 6))
        out[:3, :3] = self.m * np.eye(3)
        out[3:, 3:] = self.inertia
        return out

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        rot = self.rotation(q[3:7])
        vl, w = v[:3], v[3:]
        return np.concatenate(
            [
                self.m * np.cross(w, vl) + self.m * rot.T @ self.gravity,
                np.cross(w, self.inertia @ w),
            ]
        )

    def closure_jacobian(self, q: np.ndarray) -> np.ndarray:
        return np.zeros((0, 6))

    def closure_drift(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.zeros(0)

    def analytic_jacobians(
        self, q: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact ``(d a / d delta_q, d a / d v)`` for ``a = M^-1 (tau - h)``."""

        def skew(x: np.ndarray) -> np.ndarray:
            return np.array(
                [[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]], dtype=float
            )

        rot = self.rotation(q[3:7])
        vl, w = v[:3], v[3:]
        inv_i = np.linalg.inv(self.inertia)
        d_q = np.zeros((6, 6))
        # R -> R exp(dth):  d(R^T g) = -[dth]x R^T g, so d a_lin = -[R^T g]x dth.
        d_q[:3, 3:] = -skew(rot.T @ self.gravity)
        d_v = np.zeros((6, 6))
        d_v[:3, :3] = -skew(w)
        d_v[:3, 3:] = skew(vl)
        d_v[3:, 3:] = -inv_i @ (skew(w) @ self.inertia - skew(self.inertia @ w))
        return d_q, d_v


FIVE_BAR_GOLDEN = GOLDEN.with_name("planar_five_bar_tangent_state_space.json")


def five_bar_golden() -> dict:
    return json.loads(FIVE_BAR_GOLDEN.read_text(encoding="utf-8"))


class FiveBar:
    """Planar five-bar, ``q = (a1, b1, a2, b2)``, tips coincide (NSX-84b, k = 2).

    Point masses ``m1`` at the elbows and ``m2`` at the tips; the closure is
    ``Phi = tip_1 - tip_2`` (two rows).  Parameters are read from the golden so
    the model and its symbolic oracle cannot drift apart.
    """

    def __init__(self) -> None:
        params = five_bar_golden()["model"]["parameters"]
        for key, text in params.items():
            setattr(self, key, frac(text))

    # absolute angle, radius, sign of every term of Phi
    def _terms(self, q: np.ndarray) -> list[tuple[float, float, float, int, int]]:
        return [
            (q[0], self.L1, 1.0, 0, -1),
            (q[0] + q[1], self.L2, 1.0, 0, 1),
            (q[2], self.L1, -1.0, 2, -1),
            (q[2] + q[3], self.L2, -1.0, 2, 1),
        ]

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        out = np.zeros((4, 4))
        m1, m2, l1, l2 = self.m1, self.m2, self.L1, self.L2
        for i in (0, 2):
            cb = np.cos(q[i + 1])
            out[i, i] = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * cb
            out[i, i + 1] = out[i + 1, i] = m2 * l2**2 + m2 * l1 * l2 * cb
            out[i + 1, i + 1] = m2 * l2**2
        return out

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        m1, m2, l1, l2, g = self.m1, self.m2, self.L1, self.L2, self.g
        out = np.zeros(4)
        for i in (0, 2):
            a, b, ad, bd = q[i], q[i + 1], v[i], v[i + 1]
            out[i] = -m2 * l1 * l2 * np.sin(b) * (2 * ad * bd + bd * bd) + g * (
                (m1 + m2) * l1 * np.cos(a) + m2 * l2 * np.cos(a + b)
            )
            out[i + 1] = m2 * l1 * l2 * np.sin(b) * ad * ad + g * m2 * l2 * np.cos(
                a + b
            )
        return out

    def closure_jacobian(self, q: np.ndarray) -> np.ndarray:
        jac = np.zeros((2, 4))
        for ang, rad, sign, first, back in self._terms(q):
            du = rad * sign * np.array([-np.sin(ang), np.cos(ang)])
            jac[:, first] += du
            if back == 1:
                jac[:, first + 1] += du
        return jac

    def closure_drift(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        out = np.zeros(2)
        for ang, rad, sign, first, back in self._terms(q):
            rate = v[first] + (v[first + 1] if back == 1 else 0.0)
            out -= rad * sign * np.array([np.cos(ang), np.sin(ang)]) * rate**2
        return out

    def frame(self, q: np.ndarray) -> np.ndarray:
        """Ambient tangent frame: Gram-Schmidt of ``(Pi e_1, Pi e_3)`` (NSX-84b)."""
        jac = self.closure_jacobian(q)
        proj = np.eye(4) - jac.T @ np.linalg.solve(jac @ jac.T, jac)
        p1 = proj[:, 0] / np.linalg.norm(proj[:, 0])
        res = proj[:, 2] - (p1 @ proj[:, 2]) * p1
        return np.column_stack([p1, res / np.linalg.norm(res)])

    def state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        st = five_bar_golden()["state"]
        return (
            np.array(st["q"]),
            np.array(st["v"]),
            np.array([frac(t) for t in st["tau"]]),
        )
