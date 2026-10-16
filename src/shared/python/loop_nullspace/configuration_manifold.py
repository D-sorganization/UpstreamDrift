"""Configuration manifolds with a retraction, for on-manifold differencing (NSX-84).

Finite differences of engine dynamics must perturb ``q`` along a *retraction*,
not by adding to raw coordinates: a quaternion free-flyer has ``nq = 7`` but
``nv = 6``, and ``q + dq`` is not even a valid configuration.  A manifold maps a
tangent step ``dq`` (length ``nv``, in the engine's velocity convention) at ``q``
to a new configuration (length ``nq``).  Pure numpy; no engine imports.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.evidence import Array

__all__ = ["ConfigurationManifold", "EuclideanManifold", "FreeFlyerManifold"]


@runtime_checkable
class ConfigurationManifold(Protocol):
    """Retraction contract: ``retract(q, 0) = q`` and first-order ``q + dq``."""

    @property
    def nq(self) -> int:
        """Length of a configuration vector."""
        ...

    @property
    def nv(self) -> int:
        """Length of a tangent (velocity) vector."""
        ...

    def retract(self, q: ArrayLike, dq: ArrayLike) -> Array:
        """Configuration reached from ``q`` by the tangent step ``dq``."""
        ...


def _vec(value: ArrayLike, n: int, name: str) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


class EuclideanManifold:
    """``R^n`` with ``retract(q, dq) = q + dq`` (``nq = nv = n``)."""

    def __init__(self, n: int) -> None:
        if not isinstance(n, int | np.integer) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}")
        self._n = int(n)

    @property
    def nq(self) -> int:
        """Configuration length."""
        return self._n

    @property
    def nv(self) -> int:
        """Tangent length."""
        return self._n

    def retract(self, q: ArrayLike, dq: ArrayLike) -> Array:
        """``q + dq``."""
        return _vec(q, self._n, "q") + _vec(dq, self._n, "dq")


def _quat_to_rotation(quat: Array) -> Array:
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_multiply(a: Array, b: Array) -> Array:
    aw, av = a[0], a[1:]
    bw, bv = b[0], b[1:]
    return np.concatenate([[aw * bw - av @ bv], aw * bv + bw * av + np.cross(av, bv)])


def _quat_exp(rotvec: Array) -> Array:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.concatenate([[1.0], 0.5 * rotvec])
    half = 0.5 * angle
    return np.concatenate([[np.cos(half)], np.sin(half) * rotvec / angle])


class FreeFlyerManifold:
    """``R^3 x S^3``: ``q = (p, quat_wxyz)``, ``dq = (dp_body, dtheta_body)``.

    Matches a free joint with body-frame velocity ``(v_lin, omega)`` (nq = 7,
    nv = 6).  ``retract`` is ``p + R(quat) dp`` and ``quat * exp(dtheta)``, so
    the result is always a unit quaternion.  Adapters with another quaternion or
    frame convention convert at their own boundary.
    """

    nq = 7
    nv = 6

    def retract(self, q: ArrayLike, dq: ArrayLike) -> Array:
        """Right-multiplicative retraction of a free-flyer configuration."""
        q_arr = _vec(q, 7, "q")
        step = _vec(dq, 6, "dq")
        norm = float(np.linalg.norm(q_arr[3:]))
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError("quaternion part of q must be non-zero")
        quat = q_arr[3:] / norm
        pos = q_arr[:3] + _quat_to_rotation(quat) @ step[:3]
        new = _quat_multiply(quat, _quat_exp(step[3:]))
        return np.concatenate([pos, new / np.linalg.norm(new)])
