"""Input validation and numeric kernels for the golf-term registry (SPN-4a).

Private helpers for :mod:`src.shared.python.spine.golf_terms`. Everything here
validates its inputs with explicit ``TypeError``/``ValueError`` and returns
fresh float arrays. Angles are degrees unless a name says ``_rad``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from src.shared.python.biomechanics.joint_conventions import anatomical_joint_angles
from src.shared.python.motion_matching.posture_metrics import segment_tilt

Handedness = Literal["right", "left"]
_ROT_TOL = 1e-6


def check_handedness(handedness: object) -> Handedness:
    """Return ``handedness`` if it is ``"right"`` or ``"left"``, else raise."""
    if handedness not in ("right", "left"):
        raise ValueError(f"handedness must be 'right' or 'left', got {handedness!r}")
    return handedness  # type: ignore[return-value]


def handed_sign(handedness: Handedness) -> float:
    """ADR-0055 item 4: trail/lead sign applied to ``+AR`` and ``+LB``.

    Right-handed golfer: ``trail_axial_rotation = -AR``,
    ``lead_lateral_flexion = -LB``; left-handed: ``+``.
    """
    return -1.0 if handedness == "right" else 1.0


def require_inputs(
    term_id: str, inputs: Mapping[str, Any], names: Sequence[str]
) -> None:
    """Raise ``ValueError`` naming every entry of ``names`` absent from ``inputs``."""
    if not isinstance(inputs, Mapping):
        raise TypeError(f"inputs must be a mapping, got {type(inputs).__name__}")
    missing = [n for n in names if n not in inputs]
    if missing:
        raise ValueError(f"{term_id}: missing required inputs {', '.join(missing)}")


def float_array(value: object, name: str, ndim: int | tuple[int, ...]) -> np.ndarray:
    """Finite float array of ``ndim`` dimensions (int or tuple of allowed)."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array: {exc}") from exc
    allowed = (ndim,) if isinstance(ndim, int) else ndim
    if arr.ndim not in allowed:
        raise ValueError(f"{name} must have ndim in {allowed}, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def unit_vector(value: object, name: str) -> np.ndarray:
    """Finite nonzero 3-vector, normalised."""
    vec = float_array(value, name, 1)
    if vec.shape != (3,) or np.linalg.norm(vec) < 1e-12:
        raise ValueError(f"{name} must be a finite nonzero 3-vector")
    return vec / np.linalg.norm(vec)


def rotation_series(value: object, name: str) -> np.ndarray:
    """Return proper rotations as an ``(N, 3, 3)`` array (a lone matrix -> N=1)."""
    rot = float_array(value, name, (2, 3))
    if rot.ndim == 2:
        rot = rot[None]
    if rot.shape[1:] != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3) or (N, 3, 3), got {rot.shape}")
    gram = np.swapaxes(rot, 1, 2) @ rot
    if not np.allclose(gram, np.eye(3), atol=_ROT_TOL, rtol=0) or not np.allclose(
        np.linalg.det(rot), 1.0, atol=_ROT_TOL, rtol=0
    ):
        raise ValueError(f"{name} must hold proper rotations (orthonormal, det +1)")
    return rot


def isb_spine_angles_deg(rotation: object, name: str) -> np.ndarray:
    """ISB ``isb_spine`` ``(FE, LB, AR)`` in degrees, shape ``(N, 3)``.

    Gimbal-locked samples (``|LB| = 90 deg``) cannot separate FE from AR: those
    two components are NaN (unavailable, never zero).
    """
    rot = rotation_series(rotation, name)
    result = anatomical_joint_angles(rot, "isb_spine", degrees=True)
    angles = np.array(result.values, dtype=float).reshape(-1, 3)
    singular = np.asarray(result.singular, dtype=bool).reshape(-1)
    angles[singular, 0] = np.nan
    angles[singular, 2] = np.nan
    return angles


def horizontal_target(up: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Unit component of ``target`` perpendicular to ``up`` (toward target)."""
    flat = target - (target @ up) * up
    norm = np.linalg.norm(flat)
    if norm < 1e-9:
        raise ValueError("target must not be parallel to up")
    return flat / norm


def axis_series(value: object, name: str) -> np.ndarray:
    """Trunk axes as ``(N, 3)`` (a lone vector -> N=1)."""
    axes = float_array(value, name, (1, 2))
    if axes.ndim == 1:
        axes = axes[None]
    if axes.shape[1] != 3 or np.any(np.linalg.norm(axes, axis=1) < 1e-12):
        raise ValueError(f"{name} must hold nonzero 3-vectors, got shape {axes.shape}")
    return axes


def tilt_components(axes: np.ndarray, up: np.ndarray, target: np.ndarray) -> np.ndarray:
    """``(N, 3)`` of ``[total, toward_target, lateral]`` tilt in degrees.

    Delegates to ``posture_metrics.segment_tilt``: ``forward`` is the target
    direction in the horizontal plane and ``right = forward x up`` (so that
    ``forward, up, right`` follow X, Y, Z of the ISB frame).
    """
    forward = horizontal_target(up, target)
    right = np.cross(forward, up)
    rows = []
    for axis in axes:
        tilt = segment_tilt(axis, up, forward, right)
        rows.append((tilt.total_deg, tilt.forward_deg, tilt.lateral_deg))
    return np.asarray(rows, dtype=float)


def index_in_range(value: object, size: int, name: str) -> int:
    """Validate an integer index against ``size``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if not 0 <= int(value) < size:
        raise ValueError(f"{name}={int(value)} out of range for {size} columns")
    return int(value)


def rotate_to_z_up(points: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Rotate ``(N, 3)`` points so that unit ``up`` maps onto +Z (shortest arc)."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(up, z)
    s = float(np.linalg.norm(axis))
    c = float(up @ z)
    if s < 1e-12:
        return points if c > 0 else points * np.array([1.0, -1.0, -1.0])
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rot = np.eye(3) + s * kx + (1.0 - c) * (kx @ kx)
    return points @ rot.T
