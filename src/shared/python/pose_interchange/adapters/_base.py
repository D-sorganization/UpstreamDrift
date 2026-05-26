"""Shared utilities for per-engine pose-convention adapters.

This module collects pure-function helpers that are reused across the
per-engine adapter implementations (Drake, MuJoCo, Pinocchio, OpenSim,
Simscape). Keeping them here avoids per-adapter duplication of
quaternion-order and Euler-conversion code that has historically been
a sign-bug factory in this codebase.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
)
from src.shared.python.pose_interchange.protocol import JointSlot

# ---- Quaternion order helpers --------------------------------------------------


def quat_wxyz_to_xyzw(q: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert a quaternion from ``[w, x, y, z]`` to ``[x, y, z, w]`` order.

    Convenience shim for the MuJoCo → Pinocchio handoff where the two
    engines use opposite component orderings.

    Args:
        q: Length-4 array-like ordered ``[w, x, y, z]``.

    Returns:
        Length-4 float64 array ordered ``[x, y, z, w]``.

    Raises:
        ValueError: If ``q`` does not have shape ``(4,)``.
    """
    arr = np.asarray(q, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"quat must have shape (4,), got {arr.shape}")
    return np.array([arr[1], arr[2], arr[3], arr[0]], dtype=float)


def quat_xyzw_to_wxyz(q: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert a quaternion from ``[x, y, z, w]`` to ``[w, x, y, z]`` order.

    Inverse of :func:`quat_wxyz_to_xyzw`; used in the Pinocchio → MuJoCo
    handoff.

    Args:
        q: Length-4 array-like ordered ``[x, y, z, w]``.

    Returns:
        Length-4 float64 array ordered ``[w, x, y, z]``.

    Raises:
        ValueError: If ``q`` does not have shape ``(4,)``.
    """
    arr = np.asarray(q, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"quat must have shape (4,), got {arr.shape}")
    return np.array([arr[3], arr[0], arr[1], arr[2]], dtype=float)


# ---- Quaternion <-> intrinsic XYZ Euler (degrees) ------------------------------


def euler_xyz_deg_to_quat_wxyz(
    rotation_xyz_deg: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Convert intrinsic XYZ Euler angles (degrees) to a unit quaternion.

    Uses the canonical convention ``R = Rx(x) @ Ry(y) @ Rz(z)``.  The
    result is returned in ``[w, x, y, z]`` order and is always normalised.

    Args:
        rotation_xyz_deg: Length-3 array-like of ``[roll_x, pitch_y, yaw_z]``
            in degrees.

    Returns:
        Length-4 float64 unit quaternion ordered ``[w, x, y, z]``.

    Raises:
        ValueError: If ``rotation_xyz_deg`` does not have shape ``(3,)``.
    """
    arr = np.asarray(rotation_xyz_deg, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"rotation_xyz_deg must have shape (3,), got {arr.shape}")
    half = 0.5 * np.radians(arr)
    cx, cy, cz = np.cos(half)
    sx, sy, sz = np.sin(half)
    # q = qx * qy * qz where qx = (cx, sx, 0, 0) etc. (w-first convention).
    w = cx * cy * cz - sx * sy * sz
    x = sx * cy * cz + cx * sy * sz
    y = cx * sy * cz - sx * cy * sz
    z = cx * cy * sz + sx * sy * cz
    out = np.array([w, x, y, z], dtype=float)
    return out / np.linalg.norm(out)


def quat_wxyz_to_euler_xyz_deg(
    quat_wxyz: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Convert a unit quaternion to intrinsic XYZ Euler angles (degrees).

    Inverse of :func:`euler_xyz_deg_to_quat_wxyz`.  The conversion goes via
    a rotation matrix, which avoids the gimbal-lock sign ambiguity present
    in direct quaternion-to-Euler formulas.

    Args:
        quat_wxyz: Length-4 array-like quaternion ordered ``[w, x, y, z]``.
            Need not be pre-normalised; the function normalises internally.

    Returns:
        Length-3 float64 array ``[roll_x, pitch_y, yaw_z]`` in degrees.

    Raises:
        ValueError: If ``quat_wxyz`` does not have shape ``(4,)`` or has
            zero norm.
    """
    q = np.asarray(quat_wxyz, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"quat must have shape (4,), got {q.shape}")
    n = np.linalg.norm(q)
    if n == 0.0:
        raise ValueError("quat_wxyz_to_euler_xyz_deg: zero-norm quaternion")
    w, x, y, z = q / n
    # Build rotation matrix (right-handed, w-first).
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    matrix = np.array(
        [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]],
        dtype=float,
    )
    # Decompose using the canonical Rx@Ry@Rz convention.
    sy = float(np.clip(matrix[0, 2], -1.0, 1.0))
    yang = np.arcsin(sy)
    cy = np.cos(yang)
    if abs(cy) < 1e-9:
        xang = 0.0
        zang = float(np.arctan2(matrix[1, 0], matrix[1, 1]))
    else:
        xang = float(np.arctan2(-matrix[1, 2], matrix[2, 2]))
        zang = float(np.arctan2(-matrix[0, 1], matrix[0, 0]))
    return np.degrees(np.array([xang, float(yang), zang]))


# ---- Mock joint-slot fixture ---------------------------------------------------


def build_default_joint_layout(
    *,
    base_offset: int,
    units: str = "rad",
    sign: int = 1,
    name_prefix: str = "",
    name_suffix: str = "",
) -> dict[str, JointSlot]:
    """Return a hardcoded mock layout, one slot per canonical joint.

    The layout is a near-identity mapping: every canonical joint name in
    :data:`REFERENCE_GOLFER_FIELDS` gets its own 1-DOF slot starting at
    ``base_offset`` and incrementing by 1. The engine-side joint name is
    just ``name_prefix + canonical + name_suffix``.

    Adapters use this to provide a sensible default mock layout when no
    engine model is supplied; tests round-trip through this layout.
    """
    layout: dict[str, JointSlot] = {}
    for index, canonical in enumerate(REFERENCE_GOLFER_FIELDS):
        layout[canonical] = JointSlot(
            canonical_name=canonical,
            engine_name=f"{name_prefix}{canonical}{name_suffix}",
            start_index=base_offset + index,
            length=1,
            units=units,
            sign=sign,
        )
    return layout


# ---- Engine-q encode / decode helpers ------------------------------------------


def encode_joint_angles(
    joint_angles_deg: Mapping[str, float],
    layout: Mapping[str, JointSlot],
    engine_q: npt.NDArray[np.float64],
) -> None:
    """Write canonical joint angles into an engine ``q`` vector in-place.

    Joints absent from *layout* are silently skipped — the engine model may
    not expose every canonical DOF.  Unit conversion (deg ↔ rad) and sign
    flips are applied according to each slot's ``units`` and ``sign`` fields.

    Args:
        joint_angles_deg: Mapping of canonical joint name → angle in degrees.
        layout: Mapping of canonical joint name → :class:`~pose_interchange.protocol.JointSlot`
            describing where that joint lives in the engine ``q`` vector.
        engine_q: Pre-allocated float64 array that receives the encoded values.
            Modified in-place; the caller is responsible for its shape.
    """
    for name, slot in layout.items():
        angle_deg = float(joint_angles_deg.get(name, 0.0))
        value = angle_deg if slot.units == "deg" else float(np.radians(angle_deg))
        engine_q[slot.start_index] = slot.sign * value


def decode_joint_angles(
    engine_q: npt.NDArray[np.float64],
    layout: Mapping[str, JointSlot],
) -> dict[str, float]:
    """Read joint angles from an engine ``q`` vector and return them in degrees.

    Inverse of :func:`encode_joint_angles`.  Applies unit conversion and sign
    reversal so the returned values are always in degrees in the canonical
    (right-hand, positive-flexion) frame.

    Args:
        engine_q: Float64 array containing the engine generalised-coordinate
            vector.
        layout: Mapping of canonical joint name → :class:`~pose_interchange.protocol.JointSlot`
            that describes where each joint lives in ``engine_q``.

    Returns:
        Dict mapping canonical joint name → angle in degrees.
    """
    out: dict[str, float] = {}
    for name, slot in layout.items():
        value = float(engine_q[slot.start_index]) * slot.sign
        out[name] = value if slot.units == "deg" else float(np.degrees(value))
    return out
