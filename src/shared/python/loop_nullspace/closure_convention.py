"""Canonical grip-closure Jacobian convention and native conversion (NSX-02, R8).

Canonical convention (``loop_nullspace.tex``, "Canonical Closure Convention"):

* rows ``[fx fy fz tx ty tz]`` = ``[linear; angular]`` of the relative twist
  ``V_a - V_b`` (closing hand ``a`` relative to the club grip frame ``b``);
* components expressed in the grip frame (rotation ``R_g``: world <- grip);
* the linear rows are the velocity of the reference point ``P_g``;
* ``+J_c^T lambda`` is the generalised force the closure applies to the tree.

A native engine Jacobian is the same relative twist with a different row order,
frame, reference point and sign.  With the native local frame ``R_f``, native
reference point ``p_n`` and canonical point ``p_g``::

    C = blkdiag(R_g^T, R_g^T) @ T(p_g - p_n) @ E,
    E = blkdiag(R_f, R_f)[LOCAL] @ Pi[ang_lin] @ sigma,

so ``J_c = C J_native`` and multipliers transform contragrediently,
``lambda_native = C^T lambda_c``, which preserves the power ``lambda . (J v)``.
The drift ``Jdot v`` is converted with the same ``C``; that is exact only on
``J_c v = 0`` (admission precedes use).  Pure numpy; no engine imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict, Unpack

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.core.contracts import ensure, require

Array = NDArray[np.float64]

_ROT_TOL = 1e-9


@dataclass(frozen=True)
class NativeClosureConvention:
    """How one engine lays out its closure Jacobian.

    Attributes:
        row_order: ``"lin_ang"`` or ``"ang_lin"``.
        frame: ``"LOCAL"`` (a body/constraint frame) or ``"WORLD"``.
        point: Name of the reference point of the linear rows (documentation).
        relative_sign: ``+1`` for ``V_a - V_b``, ``-1`` for ``V_b - V_a``.
    """

    row_order: Literal["lin_ang", "ang_lin"]
    frame: Literal["LOCAL", "WORLD"]
    point: str
    relative_sign: Literal[1, -1]

    def __post_init__(self) -> None:
        if self.row_order not in ("lin_ang", "ang_lin"):
            raise ValueError(
                f"row_order must be lin_ang or ang_lin: {self.row_order!r}"
            )
        if self.frame not in ("LOCAL", "WORLD"):
            raise ValueError(f"frame must be LOCAL or WORLD: {self.frame!r}")
        if self.relative_sign not in (1, -1):
            raise ValueError(f"relative_sign must be +1 or -1: {self.relative_sign!r}")
        if not isinstance(self.point, str) or not self.point:
            raise ValueError("point must be a non-empty string")


def _skew(v: Array) -> Array:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _as_rotation(value: ArrayLike, name: str) -> Array:
    rot = np.asarray(value, dtype=float)
    if rot.shape != (3, 3) or not np.isfinite(rot).all():
        raise ValueError(f"{name} must be a finite 3 x 3 rotation matrix")
    if (
        np.abs(rot.T @ rot - np.eye(3)).max() > _ROT_TOL
        or abs(np.linalg.det(rot) - 1.0) > _ROT_TOL
    ):
        raise ValueError(f"{name} must be a rotation (orthonormal, det +1)")
    return rot


def _as_point(value: ArrayLike, name: str) -> Array:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must be a finite vector of length 3")
    return point


def closure_conversion_matrix(
    *,
    native: NativeClosureConvention,
    native_frame_rotation: ArrayLike,
    grip_frame_rotation: ArrayLike,
    native_reference_point: ArrayLike,
    grip_reference_point: ArrayLike,
) -> Array:
    """Return the 6 x 6 matrix ``C`` with ``J_canonical = C @ J_native``.

    Preconditions: both rotations are orthonormal with determinant +1 (the
    native one is the identity for a WORLD convention); both points are finite
    length-3 world positions.  Postcondition: ``C`` is finite and invertible.
    """
    r_f = _as_rotation(native_frame_rotation, "native_frame_rotation")
    r_g = _as_rotation(grip_frame_rotation, "grip_frame_rotation")
    p_n = _as_point(native_reference_point, "native_reference_point")
    p_g = _as_point(grip_reference_point, "grip_reference_point")
    if native.frame == "WORLD" and np.abs(r_f - np.eye(3)).max() > _ROT_TOL:
        raise ValueError("native_frame_rotation must be identity for a WORLD frame")

    eye, zero = np.eye(3), np.zeros((3, 3))
    undo = float(native.relative_sign) * np.eye(6)
    if native.row_order == "ang_lin":
        undo = np.block([[zero, eye], [eye, zero]]) @ undo
    if native.frame == "LOCAL":
        undo = np.block([[r_f, zero], [zero, r_f]]) @ undo
    shift = np.block([[eye, -_skew(p_g - p_n)], [zero, eye]])
    to_grip = np.block([[r_g.T, zero], [zero, r_g.T]])
    matrix = to_grip @ shift @ undo
    ensure(bool(np.isfinite(matrix).all()), "conversion matrix must be finite")
    return matrix


def _check_closure_arrays(jacobian: ArrayLike, drift: ArrayLike) -> tuple[Array, Array]:
    jac = np.asarray(jacobian, dtype=float)
    vec = np.asarray(drift, dtype=float)
    require(jac.ndim == 2 and jac.shape[0] == 6, "jacobian must be 6 x n")
    require(vec.shape == (6,), "drift must have length 6")
    require(
        bool(np.isfinite(jac).all() and np.isfinite(vec).all()), "values must be finite"
    )
    return jac, vec


def _frozen(array: Array) -> Array:
    out = np.array(array, dtype=float, copy=True)
    out.setflags(write=False)
    return out


class ClosureFrames(TypedDict):
    """Keyword arguments of :func:`closure_conversion_matrix`, shared by the
    native <-> canonical conversions so their frame inputs cannot diverge."""

    native: NativeClosureConvention
    native_frame_rotation: ArrayLike
    grip_frame_rotation: ArrayLike
    native_reference_point: ArrayLike
    grip_reference_point: ArrayLike


def to_canonical_closure(
    jacobian: ArrayLike,
    drift: ArrayLike,
    **frames: Unpack[ClosureFrames],
) -> tuple[Array, Array]:
    """Convert a native closure ``(J, Jdot v)`` to the canonical convention.

    Preconditions: ``jacobian`` is ``6 x n``, ``drift`` has length 6, all values
    finite; rotation and point preconditions of :func:`closure_conversion_matrix`.
    Postconditions: read-only ``J_c`` (``6 x n``) and ``Jdot_v`` (length 6) with
    ``J_c = C J_native``; power ``lambda_n . (J_native v)`` equals
    ``lambda_c . (J_c v)`` for ``lambda_n = C^T lambda_c``.  The drift is exact
    only on ``J_c v = 0``.
    """
    jac, vec = _check_closure_arrays(jacobian, drift)
    matrix = closure_conversion_matrix(**frames)
    return _frozen(matrix @ jac), _frozen(matrix @ vec)


def from_canonical_closure(
    jacobian: ArrayLike,
    drift: ArrayLike,
    **frames: Unpack[ClosureFrames],
) -> tuple[Array, Array]:
    """Inverse of :func:`to_canonical_closure`.

    Postcondition: ``from_canonical_closure(*to_canonical_closure(J, d, **k),
    **k)`` reproduces ``(J, d)`` to round-off.
    """
    jac, vec = _check_closure_arrays(jacobian, drift)
    matrix = closure_conversion_matrix(**frames)
    inverse_jac = np.asarray(np.linalg.solve(matrix, jac), dtype=np.float64)
    inverse_vec = np.asarray(np.linalg.solve(matrix, vec), dtype=np.float64)
    return _frozen(inverse_jac), _frozen(inverse_vec)
