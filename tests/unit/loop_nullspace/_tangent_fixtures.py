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
