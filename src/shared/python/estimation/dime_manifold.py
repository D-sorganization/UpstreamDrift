"""DIME Manifold Contracts and Implementations (#11421, #11423).

Defines manifold operations for continuous state estimation, retract and local-coordinate
maps, declared tangent dimensions, Jacobians, and quaternion sign equivalence in SO(3) and SE(3).

Chart convention (#11550): rotations use Hamilton unit quaternions ``[w, x, y, z]`` and the
right (body-frame) retraction ``retract(q, v) = q (x) Exp(v)``, with ``Exp`` the rotation by
``|v|`` about ``v / |v|``. ``local_coordinates(q0, q1) = Log(q0^-1 (x) q1)`` with ``|v| <= pi``.
The Jacobians are the exact derivatives of these maps at any configuration and velocity:

* ``retract_jacobian(q, v) = L(q) dExp(v)/dv`` (``L`` the left quaternion-product matrix);
* ``local_coordinates_jacobian(q0, q1) = J_r^{-1}(phi)`` with ``phi = local_coordinates(q0, q1)``,
  the derivative of ``local_coordinates(q0, retract(q1, d))`` with respect to ``d`` at ``d = 0``.

``SE3Manifold`` is the product chart ``R^3 x SO(3)`` (translation first, then rotation), so its
Jacobians are exactly block diagonal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.shared.python.contracts import require, require_finite, require_unit_vector
from src.shared.python.spatial_algebra.spatial_vectors import skew

# Below this rotation angle the closed-form coefficients lose precision to cancellation and
# the Taylor expansions (truncation error O(theta^4) < 1e-16) are used instead.
_SMALL_ANGLE = 1e-4


def _make_readonly_array(arr: np.ndarray) -> np.ndarray:
    """Return a read-only copy of a float64 numpy array."""
    out = np.array(arr, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


def _require_vector(arr: np.ndarray, size: int, name: str) -> None:
    """Precondition: ``arr`` is a finite 1-D array of length ``size``."""
    require(
        np.shape(arr) == (size,), f"{name} must have shape ({size},)", np.shape(arr)
    )
    require_finite(arr, name)


def _require_unit_quaternion(q: np.ndarray, name: str) -> None:
    """Precondition: ``q`` is a finite unit quaternion ``[w, x, y, z]``."""
    _require_vector(q, 4, name)
    require_unit_vector(q, name)


def _quaternion_left_matrix(q: np.ndarray) -> np.ndarray:
    """Return ``L(q)`` with ``q (x) p = L(q) @ p`` for Hamilton quaternions ``[w, x, y, z]``."""
    w, x, y, z = q
    return np.array(
        [
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w],
        ],
        dtype=np.float64,
    )


def _quaternion_exp_jacobian(v: np.ndarray) -> np.ndarray:
    """Return ``dExp(v)/dv`` (4 x 3) for ``Exp(v) = [cos(t/2), sin(t/2) v / t]``, ``t = |v|``.

    With ``s(t) = sin(t/2) / t`` the derivative is
    ``[[-s(t)/2 * v^T], [s(t) I + (s'(t)/t) v v^T]]``; the small-angle branch uses
    ``s ~ 1/2 - t^2/48`` and ``s'/t ~ -1/24 + t^2/960``.
    """
    theta = float(np.linalg.norm(v))
    if theta < _SMALL_ANGLE:
        t2 = theta * theta
        s = 0.5 - t2 / 48.0
        c = -1.0 / 24.0 + t2 / 960.0
    else:
        half = 0.5 * theta
        s = np.sin(half) / theta
        c = (0.5 * np.cos(half) - s) / (theta * theta)
    jac = np.empty((4, 3), dtype=np.float64)
    jac[0, :] = -0.5 * s * v
    jac[1:, :] = s * np.eye(3) + c * np.outer(v, v)
    return jac


def so3_right_jacobian_inverse(phi: np.ndarray) -> np.ndarray:
    """Inverse right Jacobian of SO(3): ``Log(Exp(phi) Exp(d)) = phi + J_r^{-1}(phi) d + O(d^2)``.

    ``J_r^{-1}(phi) = I + 1/2 [phi]x + (1/t^2 - cot(t/2) / (2 t)) [phi]x^2`` with ``t = |phi|``;
    the coefficient tends to ``1/12 + t^2/720`` as ``t -> 0`` and stays finite at ``t = pi``.

    Preconditions: ``phi`` is a finite 3-vector with ``|phi| <= pi`` (principal branch).
    Postcondition: a finite (3, 3) matrix.
    """
    _require_vector(phi, 3, "phi")
    theta = float(np.linalg.norm(phi))
    require(
        theta <= np.pi + 1e-9, "phi must lie on the principal branch |phi| <= pi", theta
    )
    if theta < _SMALL_ANGLE:
        coeff = 1.0 / 12.0 + theta * theta / 720.0
    else:
        half = 0.5 * theta
        coeff = 1.0 / (theta * theta) - np.cos(half) / (2.0 * theta * np.sin(half))
    phi_x = skew(phi)
    return np.eye(3) + 0.5 * phi_x + coeff * (phi_x @ phi_x)


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
        # atan2 keeps full precision near the identity, where arccos(dw ~ 1) loses ~8 digits.
        angle = 2.0 * np.arctan2(s, dw)

        if s < 1e-12:
            return _make_readonly_array(2.0 * v_vec)
        return _make_readonly_array((angle / s) * v_vec)

    def retract_jacobian(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Exact ``d retract(q, v) / dv = L(q) dExp(v)/dv`` (4 x 3) at any ``v``.

        Preconditions: ``q`` is a finite unit quaternion; ``v`` is a finite 3-vector.
        At ``v = 0`` this reduces to ``0.5 * L(q)[:, 1:]``.
        """
        _require_unit_quaternion(q, "q")
        _require_vector(v, 3, "v")
        return _quaternion_left_matrix(q) @ _quaternion_exp_jacobian(v)

    def local_coordinates_jacobian(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        """Exact ``d local_coordinates(q0, retract(q1, d)) / dd`` at ``d = 0`` (3 x 3).

        Equals ``J_r^{-1}(phi)`` with ``phi = local_coordinates(q0, q1)``; it is the identity
        only when ``q1`` and ``q0`` represent the same rotation.

        Preconditions: ``q0`` and ``q1`` are finite unit quaternions.
        """
        _require_unit_quaternion(q0, "q0")
        _require_unit_quaternion(q1, "q1")
        return so3_right_jacobian_inverse(np.asarray(self.local_coordinates(q0, q1)))

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
        """Exact block-diagonal ``d retract(q, v) / dv`` (7 x 6), translation block first.

        Preconditions: ``q`` is a finite 7-vector with a unit quaternion tail; ``v`` is a
        finite 6-vector ``[translation(3), rotation(3)]``.
        """
        _require_vector(q, 7, "q")
        _require_vector(v, 6, "v")
        jac = np.zeros((7, 6), dtype=np.float64)
        jac[:3, :3] = self._pos.retract_jacobian(q[:3], v[:3])
        jac[3:7, 3:6] = self._rot.retract_jacobian(q[3:7], v[3:6])
        return jac

    def local_coordinates_jacobian(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        """Exact ``d local_coordinates(q0, retract(q1, d)) / dd`` at ``d = 0`` (6 x 6).

        Block diagonal ``diag(I_3, J_r^{-1}(phi))`` for the ``R^3 x SO(3)`` product chart.
        Preconditions: ``q0`` and ``q1`` are finite 7-vectors with unit quaternion tails.
        """
        _require_vector(q0, 7, "q0")
        _require_vector(q1, 7, "q1")
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
