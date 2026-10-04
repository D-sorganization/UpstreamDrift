"""DIME Manifold Contracts and Implementations (#11421, #11423).

Defines manifold operations for continuous state estimation, retract and local-coordinate
maps, declared tangent dimensions, Jacobians, and quaternion sign equivalence in SO(3) and SE(3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.shared.python.contracts import require


def _make_readonly_array(arr: np.ndarray) -> np.ndarray:
    """Return a read-only copy of a float64 numpy array."""
    out = np.array(arr, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


class ManifoldContract(ABC):
    """Abstract contract for configuration manifolds supporting nq != nv and sign equivalence."""

    @property
    @abstractmethod
    def n_q(self) -> int:
        """Configuration space dimension."""
        ...

    @property
    @abstractmethod
    def n_v(self) -> int:
        """Tangent velocity space dimension."""
        ...

    @property
    @abstractmethod
    def frame_convention(self) -> str:
        """Declared frame and coordinate convention."""
        ...

    @abstractmethod
    def retract(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Retract configuration q along tangent perturbation v."""
        ...

    @abstractmethod
    def local_coordinates(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        """Compute tangent vector v in T_{q0} M such that retract(q0, v) ~ q1."""
        ...

    @abstractmethod
    def retract_jacobian(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Jacobian d(retract(q, v))/dv of shape (n_q, n_v)."""
        ...

    @abstractmethod
    def local_coordinates_jacobian(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        """Jacobian of local_coordinates with respect to tangent perturbation (n_v, n_v)."""
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize manifold specification to dictionary."""
        ...


class VectorSpaceManifold(ManifoldContract):
    """Euclidean vector space R^n where n_q == n_v."""

    def __init__(self, dim: int) -> None:
        require(dim > 0, "Dimension must be strictly positive", dim)
        self._dim = dim

    @property
    def n_q(self) -> int:
        return self._dim

    @property
    def n_v(self) -> int:
        return self._dim

    @property
    def frame_convention(self) -> str:
        return "cartesian_linear"

    def retract(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return _make_readonly_array(q + v)

    def local_coordinates(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        return _make_readonly_array(q1 - q0)

    def retract_jacobian(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.eye(self._dim, dtype=np.float64)

    def local_coordinates_jacobian(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        return np.eye(self._dim, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "VectorSpaceManifold", "dim": self._dim}


class QuaternionManifold(ManifoldContract):
    """Unit quaternion SO(3) manifold (nq = 4, nv = 3) with exact sign equivalence."""

    @property
    def n_q(self) -> int:
        return 4

    @property
    def n_v(self) -> int:
        return 3

    @property
    def frame_convention(self) -> str:
        return "quaternion_wxyz"

    def retract(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        theta = float(np.linalg.norm(v))
        if theta < 1e-12:
            dq = np.array([1.0, 0.5 * v[0], 0.5 * v[1], 0.5 * v[2]], dtype=np.float64)
            dq /= np.linalg.norm(dq)
        else:
            half = 0.5 * theta
            dq = np.array(
                [
                    np.cos(half),
                    np.sin(half) * v[0] / theta,
                    np.sin(half) * v[1] / theta,
                    np.sin(half) * v[2] / theta,
                ],
                dtype=np.float64,
            )

        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = dq
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        out = np.array([w, x, y, z], dtype=np.float64)
        out /= np.linalg.norm(out)
        return _make_readonly_array(out)

    def local_coordinates(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        dot = float(np.dot(q0, q1))
        q1_eff = -q1 if dot < 0.0 else q1

        w0, x0, y0, z0 = q0
        w1, x1, y1, z1 = q1_eff
        dw = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1
        dx = w0 * x1 - x0 * w1 - y0 * z1 + z0 * y1
        dy = w0 * y1 + x0 * z1 - y0 * w1 - z0 * x1
        dz = w0 * z1 - x0 * y1 + y0 * x1 - z0 * w1

        if dw < 0.0:
            dw, dx, dy, dz = -dw, -dx, -dy, -dz
        dw = float(np.clip(dw, -1.0, 1.0))

        v_vec = np.array([dx, dy, dz], dtype=np.float64)
        s = float(np.linalg.norm(v_vec))
        angle = 2.0 * np.arccos(dw)

        if s < 1e-12:
            return _make_readonly_array(2.0 * v_vec)
        return _make_readonly_array((angle / s) * v_vec)

    def retract_jacobian(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        e_matrix = np.array(
            [
                [-x, -y, -z],
                [w, -z, y],
                [z, w, -x],
                [-y, x, w],
            ],
            dtype=np.float64,
        )
        return 0.5 * e_matrix

    def local_coordinates_jacobian(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        return np.eye(3, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "QuaternionManifold"}


class SE3Manifold(ManifoldContract):
    """SE(3) rigid body manifold combining R^3 translation with SO(3) quaternion (nq = 7, nv = 6)."""

    def __init__(self) -> None:
        self._pos = VectorSpaceManifold(3)
        self._rot = QuaternionManifold()

    @property
    def n_q(self) -> int:
        return 7

    @property
    def n_v(self) -> int:
        return 6

    @property
    def frame_convention(self) -> str:
        return "se3_xyz_wxyz"

    def retract(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q_pos = self._pos.retract(q[:3], v[:3])
        q_rot = self._rot.retract(q[3:7], v[3:6])
        return _make_readonly_array(np.concatenate([q_pos, q_rot]))

    def local_coordinates(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        v_pos = self._pos.local_coordinates(q0[:3], q1[:3])
        v_rot = self._rot.local_coordinates(q0[3:7], q1[3:7])
        return _make_readonly_array(np.concatenate([v_pos, v_rot]))

    def retract_jacobian(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        jac = np.zeros((7, 6), dtype=np.float64)
        jac[:3, :3] = self._pos.retract_jacobian(q[:3], v[:3])
        jac[3:7, 3:6] = self._rot.retract_jacobian(q[3:7], v[3:6])
        return jac

    def local_coordinates_jacobian(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        jac = np.zeros((6, 6), dtype=np.float64)
        jac[:3, :3] = self._pos.local_coordinates_jacobian(q0[:3], q1[:3])
        jac[3:6, 3:6] = self._rot.local_coordinates_jacobian(q0[3:7], q1[3:7])
        return jac

    def to_dict(self) -> dict[str, Any]:
        return {"type": "SE3Manifold"}


def manifold_from_dict(data: dict[str, Any]) -> ManifoldContract:
    """Deserialize manifold from dictionary representation."""
    m_type = data.get("type")
    if m_type == "VectorSpaceManifold":
        return VectorSpaceManifold(int(data["dim"]))
    if m_type == "QuaternionManifold":
        return QuaternionManifold()
    if m_type == "SE3Manifold":
        return SE3Manifold()
    raise ValueError(f"Unknown manifold type: {m_type}")
