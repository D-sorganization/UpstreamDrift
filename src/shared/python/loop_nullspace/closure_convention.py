"""Canonical grip-closure Jacobian frame, row-order, and sign conventions (NSX-02).

Provides a single deterministic conversion boundary between per-engine native
6-DOF weld closure linearizations and the NSX-00 / R8 canonical closure
convention:

- **Frame**: canonical grip frame ``G`` with world rotation ``R_WG``
  (``grip_frame_rotation``).
- **Point**: canonical grip origin ``p_G`` in world coordinates
  (``grip_origin``).
- **Row order**: ``[fx, fy, fz, tx, ty, tz]`` (``"lin_ang"``).
- **Sign**: ``+J_c.T @ lambda`` is the generalized load acting on the human
  kinematic tree (relative velocity convention ``a - b``).

When a native Jacobian evaluates spatial motion at a reference point ``p_P``
different from ``p_G``, the linear rows shift according to rigid-body spatial
kinematics:

    v_G = v_P + omega x (p_G - p_P) = v_P - (p_G - p_P) x omega

and by virtual power duality ``lambda_G . v_G = lambda_P . v_P``:

    f_P   = f_G
    tau_P = tau_G + (p_G - p_P) x f_G
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.contracts import require

Array: TypeAlias = NDArray[np.float64]
RowOrder: TypeAlias = Literal["lin_ang", "ang_lin"]
ClosureFrame: TypeAlias = Literal["local", "world"]
ClosurePoint: TypeAlias = Literal["grip_midpoint", "placement_a", "site_xpos"]
RelativeSign: TypeAlias = Literal[1, -1]

_CLOSURE_ROWS: Final[int] = 6
_ORTHO_TOL: Final[float] = 1e-6

__all__ = [
    "CANONICAL_CLOSURE_CONVENTION",
    "ClosureFrame",
    "ClosurePoint",
    "DRAKE_CLOSURE_CONVENTION",
    "MUJOCO_CLOSURE_CONVENTION",
    "NativeClosureConvention",
    "OPENSIM_CLOSURE_CONVENTION",
    "PINOCCHIO_CLOSURE_CONVENTION",
    "RelativeSign",
    "RowOrder",
    "dual_wrench_to_canonical",
    "dual_wrench_to_native",
    "from_canonical_closure",
    "to_canonical_closure",
]


@dataclass(frozen=True, slots=True)
class NativeClosureConvention:
    """Declarative specification of an engine's native 6-DOF weld closure Jacobian."""

    engine: str
    row_order: RowOrder
    frame: ClosureFrame
    point: ClosurePoint
    relative_sign: RelativeSign

    def __post_init__(self) -> None:
        require(bool(self.engine.strip()), "engine name must be non-empty")
        require(
            self.row_order in ("lin_ang", "ang_lin"),
            f"row_order must be 'lin_ang' or 'ang_lin', got {self.row_order!r}",
        )
        require(
            self.frame in ("local", "world"),
            f"frame must be 'local' or 'world', got {self.frame!r}",
        )
        require(
            self.point in ("grip_midpoint", "placement_a", "site_xpos"),
            f"unsupported closure point {self.point!r}",
        )
        require(
            self.relative_sign in (1, -1),
            f"relative_sign must be +1 or -1, got {self.relative_sign}",
        )


CANONICAL_CLOSURE_CONVENTION: Final[NativeClosureConvention] = NativeClosureConvention(
    engine="canonical",
    row_order="lin_ang",
    frame="local",
    point="grip_midpoint",
    relative_sign=1,
)

PINOCCHIO_CLOSURE_CONVENTION: Final[NativeClosureConvention] = NativeClosureConvention(
    engine="pinocchio",
    row_order="lin_ang",
    frame="local",
    point="placement_a",
    relative_sign=1,
)

DRAKE_CLOSURE_CONVENTION: Final[NativeClosureConvention] = NativeClosureConvention(
    engine="drake",
    row_order="ang_lin",
    frame="local",
    point="placement_a",
    relative_sign=1,
)

MUJOCO_CLOSURE_CONVENTION: Final[NativeClosureConvention] = NativeClosureConvention(
    engine="mujoco",
    row_order="lin_ang",
    frame="world",
    point="site_xpos",
    relative_sign=1,
)

OPENSIM_CLOSURE_CONVENTION: Final[NativeClosureConvention] = NativeClosureConvention(
    engine="opensim",
    row_order="lin_ang",
    frame="world",
    point="placement_a",
    relative_sign=-1,
)


def _skew(vec: Array) -> Array:
    """Return 3x3 skew-symmetric cross-product matrix ``[v]_x``."""
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def _validate_rotation(rotation: ArrayLike, name: str) -> Array:
    if not isinstance(rotation, (np.ndarray, list, tuple)):
        raise TypeError(f"{name} must be an array-like 3x3 matrix")
    rot = np.asarray(rotation, dtype=np.float64)
    require(rot.shape == (3, 3), f"{name} must have shape (3, 3), got {rot.shape}")
    require(bool(np.isfinite(rot).all()), f"{name} must contain finite entries")
    gram_err = float(np.max(np.abs(rot.T @ rot - np.eye(3))))
    det_err = abs(float(np.linalg.det(rot)) - 1.0)
    require(
        gram_err <= _ORTHO_TOL and det_err <= _ORTHO_TOL,
        f"{name} must be a proper orthonormal rotation in SO(3) "
        f"(gram_err={gram_err:.3e}, det_err={det_err:.3e})",
    )
    return rot


def _validate_point(point: ArrayLike, name: str) -> Array:
    if not isinstance(point, (np.ndarray, list, tuple)):
        raise TypeError(f"{name} must be a 3-element vector")
    arr = np.asarray(point, dtype=np.float64)
    require(arr.shape == (3,), f"{name} must have shape (3,), got {arr.shape}")
    require(bool(np.isfinite(arr).all()), f"{name} must be finite")
    return arr


def _validate_closure_pair(
    jacobian: ArrayLike, drift: ArrayLike
) -> tuple[Array, Array]:
    if not isinstance(jacobian, (np.ndarray, list, tuple)):
        raise TypeError("jacobian must be a 2-D array-like")
    if not isinstance(drift, (np.ndarray, list, tuple)):
        raise TypeError("drift must be a 1-D array-like")
    jac = np.asarray(jacobian, dtype=np.float64)
    dft = np.asarray(drift, dtype=np.float64)
    require(
        jac.ndim == 2 and jac.shape[0] == _CLOSURE_ROWS,
        f"jacobian must have shape (6, nv), got {jac.shape}",
    )
    require(
        dft.shape == (_CLOSURE_ROWS,),
        f"drift must have shape (6,), got {dft.shape}",
    )
    require(bool(np.isfinite(jac).all()), "jacobian must be finite")
    require(bool(np.isfinite(dft).all()), "drift must be finite")
    return jac, dft


def _motion_transform_to_canonical(
    native: NativeClosureConvention,
    grip_frame_rotation: ArrayLike,
    grip_origin: ArrayLike,
    reference_point: ArrayLike | None,
) -> Array:
    """Build the 6x6 invertible motion transform ``M`` mapping native rows to canonical."""
    if not isinstance(native, NativeClosureConvention):
        raise TypeError("native must be a NativeClosureConvention instance")
    rot_wg = _validate_rotation(grip_frame_rotation, "grip_frame_rotation")
    p_grip = _validate_point(grip_origin, "grip_origin")
    p_ref = (
        p_grip
        if reference_point is None
        else _validate_point(reference_point, "reference_point")
    )

    eye3 = np.eye(3, dtype=np.float64)
    zero3 = np.zeros((3, 3), dtype=np.float64)

    # 1. Row-order and sign normalization to [lin; ang] with +1 tree sign
    if native.row_order == "ang_lin":
        perm = np.block([[zero3, eye3], [eye3, zero3]])
    else:
        perm = np.eye(6, dtype=np.float64)
    signed_perm = float(native.relative_sign) * perm

    # 2. Rotate from local reference frame (if local) to world frame
    if native.frame == "local":
        to_world = np.block([[rot_wg, zero3], [zero3, rot_wg]])
    else:
        to_world = np.eye(6, dtype=np.float64)

    # 3. Shift evaluation point from reference_point p_P to grip_origin p_G in world:
    #    v_G = v_P + omega x (p_G - p_P) = v_P - [p_G - p_P]_x omega
    lever_w = p_grip - p_ref
    shift_world = np.block([[eye3, -_skew(lever_w)], [zero3, eye3]])

    # 4. Express in canonical grip frame G via R_WG^T
    rot_gw = rot_wg.T
    to_grip = np.block([[rot_gw, zero3], [zero3, rot_gw]])

    return to_grip @ shift_world @ to_world @ signed_perm


def to_canonical_closure(
    jacobian: ArrayLike,
    drift: ArrayLike,
    *,
    native: NativeClosureConvention,
    grip_frame_rotation: ArrayLike,
    grip_origin: ArrayLike,
    reference_point: ArrayLike | None = None,
) -> tuple[Array, Array]:
    """Convert a native 6-row closure Jacobian and drift into the R8 canonical convention.

    Postconditions:
        Returns ``(J_c, Jdot_v)`` with read-only ``float64`` arrays of shape
        ``(6, nv)`` and ``(6,)`` expressed in the canonical grip frame at
        ``grip_origin`` with ``[fx, fy, fz, tx, ty, tz]`` row ordering and
        ``+J_c.T @ lambda`` acting on the tree.
    """
    jac, dft = _validate_closure_pair(jacobian, drift)
    transform = _motion_transform_to_canonical(
        native, grip_frame_rotation, grip_origin, reference_point
    )
    j_canon = np.ascontiguousarray(transform @ jac, dtype=np.float64)
    dft_canon = np.ascontiguousarray(transform @ dft, dtype=np.float64)
    j_canon.setflags(write=False)
    dft_canon.setflags(write=False)
    return j_canon, dft_canon


def from_canonical_closure(
    j_canonical: ArrayLike,
    drift_canonical: ArrayLike,
    *,
    native: NativeClosureConvention,
    grip_frame_rotation: ArrayLike,
    grip_origin: ArrayLike,
    reference_point: ArrayLike | None = None,
) -> tuple[Array, Array]:
    """Inverse of :func:`to_canonical_closure`, mapping canonical rows back to native."""
    jac, dft = _validate_closure_pair(j_canonical, drift_canonical)
    transform = _motion_transform_to_canonical(
        native, grip_frame_rotation, grip_origin, reference_point
    )
    inv_transform = np.linalg.inv(transform)
    j_native = np.ascontiguousarray(inv_transform @ jac, dtype=np.float64)
    dft_native = np.ascontiguousarray(inv_transform @ dft, dtype=np.float64)
    j_native.setflags(write=False)
    dft_native.setflags(write=False)
    return j_native, dft_native


def dual_wrench_to_canonical(
    lambda_native: ArrayLike,
    *,
    native: NativeClosureConvention,
    grip_frame_rotation: ArrayLike,
    grip_origin: ArrayLike,
    reference_point: ArrayLike | None = None,
) -> Array:
    """Convert a native constraint multiplier wrench ``lambda_native`` to canonical.

    Preserves virtual power ``lambda_canonical . (J_c v) == lambda_native . (J_nat v)``
    and generalized tree load ``J_c.T @ lambda_canonical == J_nat.T @ lambda_native``.
    """
    if not isinstance(lambda_native, (np.ndarray, list, tuple)):
        raise TypeError("lambda_native must be a 6-element vector")
    lam = np.asarray(lambda_native, dtype=np.float64)
    require(
        lam.shape == (_CLOSURE_ROWS,),
        f"lambda_native must have shape (6,), got {lam.shape}",
    )
    require(bool(np.isfinite(lam).all()), "lambda_native must be finite")
    transform = _motion_transform_to_canonical(
        native, grip_frame_rotation, grip_origin, reference_point
    )
    # Since J_c = M @ J_nat, J_c.T @ lam_c = J_nat.T @ lam_nat => M.T @ lam_c = lam_nat
    lam_canon = np.ascontiguousarray(
        np.linalg.solve(transform.T, lam), dtype=np.float64
    )
    lam_canon.setflags(write=False)
    return lam_canon


def dual_wrench_to_native(
    lambda_canonical: ArrayLike,
    *,
    native: NativeClosureConvention,
    grip_frame_rotation: ArrayLike,
    grip_origin: ArrayLike,
    reference_point: ArrayLike | None = None,
) -> Array:
    """Convert a canonical grip wrench multiplier ``lambda_canonical`` to native."""
    if not isinstance(lambda_canonical, (np.ndarray, list, tuple)):
        raise TypeError("lambda_canonical must be a 6-element vector")
    lam = np.asarray(lambda_canonical, dtype=np.float64)
    require(
        lam.shape == (_CLOSURE_ROWS,),
        f"lambda_canonical must have shape (6,), got {lam.shape}",
    )
    require(bool(np.isfinite(lam).all()), "lambda_canonical must be finite")
    transform = _motion_transform_to_canonical(
        native, grip_frame_rotation, grip_origin, reference_point
    )
    lam_native = np.ascontiguousarray(transform.T @ lam, dtype=np.float64)
    lam_native.setflags(write=False)
    return lam_native
