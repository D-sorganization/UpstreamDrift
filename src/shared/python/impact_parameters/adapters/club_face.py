"""Shared club-face geometry and rigid-body kernel for ClubheadSeries (GCV-16).

The face-centre offset and face axes are defined here **once**, in the club
body frame, and every engine adapter reduces its own forward kinematics to the
same per-sample rigid-body state (origin pose, origin velocity, angular
velocity).  :func:`rigid_body_series` is the only place that transports that
state to the face centre, so the frame math is not repeated per engine.

Club body frame (native convention, ``motion_matching.club_models``): origin
at the head, shaft along -y toward the grip.  The default face axes are the
face normal +x, the grip axis -y and the toe axis -z so that
``high = toe x normal`` points up the face toward the shaft.  The face-centre
offset defaults to the body origin; reconcile it with the GCV-11 face geometry
by constructing a :class:`ClubFaceSpec` with that offset (single integration
point, no per-engine change).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..clubhead_series import ClubheadSeries

_ORTHO_TOL = 1e-9


def _unit3(value: object, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 3-vector, got {arr.shape}")
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        raise ValueError(f"{name} must be nonzero")
    return arr / norm


@dataclass(frozen=True)
class ClubFaceSpec:
    """Face centre and face axes in the club body frame (defined once)."""

    face_center_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal_body: tuple[float, float, float] = (1.0, 0.0, 0.0)
    toe_body: tuple[float, float, float] = (0.0, 0.0, -1.0)
    grip_body: tuple[float, float, float] = (0.0, -1.0, 0.0)

    def __post_init__(self) -> None:
        centre = np.asarray(self.face_center_body_m, dtype=float)
        if centre.shape != (3,) or not np.all(np.isfinite(centre)):
            raise ValueError("face_center_body_m must be a finite 3-vector")
        n = _unit3(self.normal_body, "normal_body")
        toe = _unit3(self.toe_body, "toe_body")
        grip = _unit3(self.grip_body, "grip_body")
        if abs(float(n @ toe)) > _ORTHO_TOL:
            raise ValueError("toe_body must be orthogonal to normal_body")
        if abs(float(n @ grip)) > _ORTHO_TOL or abs(float(toe @ grip)) > _ORTHO_TOL:
            raise ValueError("grip_body must be orthogonal to normal_body and toe_body")

    def axes(self) -> np.ndarray:
        """Unit (normal, toe, grip) stacked as rows, shape (3, 3)."""
        return np.stack(
            [
                _unit3(self.normal_body, "normal_body"),
                _unit3(self.toe_body, "toe_body"),
                _unit3(self.grip_body, "grip_body"),
            ]
        )


NATIVE_CLUB_FACE = ClubFaceSpec()


def _check_samples(times: np.ndarray, **arrays: np.ndarray) -> int:
    n = int(times.shape[0])
    shapes = {
        "origins_m": (n, 3),
        "rotations": (n, 3, 3),
        "origin_velocities_mps": (n, 3),
        "angular_velocities_rps": (n, 3),
    }
    for name, arr in arrays.items():
        if arr.shape != shapes[name]:
            raise ValueError(f"{name} must have shape {shapes[name]}, got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite")
    return n


def empty_pose_twist(
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Zeroed per-sample buffers (position, rotation, linear, angular velocity).

    Each engine adapter fills these from its own FK before calling
    :func:`rigid_body_series`. Raises ``ValueError`` for a negative ``n``.
    """
    if n < 0:
        raise ValueError(f"sample count must be non-negative, got {n}")
    return np.zeros((n, 3)), np.zeros((n, 3, 3)), np.zeros((n, 3)), np.zeros((n, 3))


def rigid_body_series(
    times_s: object,
    origins_m: object,
    rotations: object,
    origin_velocities_mps: object,
    angular_velocities_rps: object,
    spec: ClubFaceSpec = NATIVE_CLUB_FACE,
) -> ClubheadSeries:
    """Transport a club-body rigid state to the face centre.

    Inputs are world-frame: body-origin position ``(N, 3)``, body-to-world
    rotation ``(N, 3, 3)``, body-origin linear velocity ``(N, 3)`` and body
    angular velocity ``(N, 3)``.  Postcondition: the returned series holds the
    face-centre position, the velocity of that material point
    (``v_o + w x R c``) and unit-length world face axes.
    """
    t = np.asarray(times_s, dtype=float)
    if t.ndim != 1:
        raise ValueError("times_s must be 1-D")
    p = np.asarray(origins_m, dtype=float)
    rot = np.asarray(rotations, dtype=float)
    v = np.asarray(origin_velocities_mps, dtype=float)
    w = np.asarray(angular_velocities_rps, dtype=float)
    _check_samples(
        t,
        origins_m=p,
        rotations=rot,
        origin_velocities_mps=v,
        angular_velocities_rps=w,
    )
    ortho = np.einsum("nij,nkj->nik", rot, rot) - np.eye(3)
    if float(np.max(np.abs(ortho))) > 1e-6:
        raise ValueError("rotations must be orthonormal")
    offset = np.einsum("nij,j->ni", rot, np.asarray(spec.face_center_body_m))
    axes = np.einsum("nij,aj->nai", rot, spec.axes())
    return ClubheadSeries(
        times_s=t,
        face_center_m=p + offset,
        velocity_mps=v + np.cross(w, offset),
        face_normal=axes[:, 0],
        toe_axis=axes[:, 1],
        grip_axis=axes[:, 2],
    )


def check_trajectory(
    times_s: object, q: object, v: object, name: str = "q"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a ``(times, q, v)`` rollout shared by the joint-space adapters."""
    t = np.asarray(times_s, dtype=float)
    qa = np.asarray(q, dtype=float)
    va = np.asarray(v, dtype=float)
    if t.ndim != 1 or t.shape[0] < 2:
        raise ValueError("times_s must be 1-D with at least 2 samples")
    if qa.ndim != 2 or qa.shape[0] != t.shape[0]:
        raise ValueError(f"{name} must have shape (N, nq) with N={t.shape[0]}")
    if va.ndim != 2 or va.shape[0] != t.shape[0]:
        raise ValueError(f"velocities must have shape (N, nv) with N={t.shape[0]}")
    if not (np.all(np.isfinite(qa)) and np.all(np.isfinite(va))):
        raise ValueError("positions and velocities must be finite")
    if not math.isfinite(float(t[0])):
        raise ValueError("times_s must be finite")
    return t, qa, va
