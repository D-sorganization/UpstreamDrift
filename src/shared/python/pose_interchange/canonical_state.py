"""Canonical-v2 dynamic state for cross-engine biomechanics (CC-2).

``CanonicalState`` is the engine-agnostic dynamic value type the Canonical Core
(EPIC #6772) routes through. Where ``canonical-v1`` (:class:`CanonicalPose`,
:mod:`pose_interchange.canonical`) carries a *pose* only, ``canonical-v2`` carries
full dynamic state ``(q, v, a, t)`` with a singularity-free quaternion floating
base. The contract is frozen in ``docs/conventions/canonical-v2.md``
(:doc:`ADR-0026 </adr/0026-canonical-dynamic-state-v2>`).

Layout (model with a floating base and ``n_j`` internal joint coordinates)::

    q  (nq = 7 + n_j):  [ base_xyz(3) | base_quat_wxyz(4) | joints(n_j) ]
    v  (nv = 6 + n_j):  [ base_lin(3) | base_ang(3, body frame) | joints(n_j) ]
    a  (nv):            same layout as v

``nq = nv + 1`` because the unit quaternion has one redundant coordinate, so the
base is updated on its manifold via :meth:`integrate` (built on
:func:`pose_interchange.se3.quat_exp`) — **never** by naive vector addition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from src.shared.python.pose_interchange import se3

ConventionTagV2 = Literal["canonical-v2"]
CONVENTION_TAG_V2: Final[ConventionTagV2] = "canonical-v2"

_BASE_NQ: Final[int] = 7  # xyz(3) + quat(4)
_BASE_NV: Final[int] = 6  # lin(3) + ang(3)
_QUAT_NORM_TOL: Final[float] = 1e-6

__all__ = ["CONVENTION_TAG_V2", "CanonicalState", "canonical_state_zero"]


def _readonly(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    out = np.ascontiguousarray(arr, dtype=float).copy()
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class CanonicalState:
    """Engine-agnostic dynamic state in the ``canonical-v2`` convention.

    Parameters
    ----------
    q
        Configuration, shape ``(7 + n_j,)``:
        ``[base_xyz, base_quat_wxyz, joint_coords]``. The base quaternion is
        scalar-first and must be unit norm.
    v, a
        Generalized velocity / acceleration, shape ``(6 + n_j,)``:
        ``[base_lin, base_ang, joint_speeds]``. Base angular velocity is in the
        body (local) frame.
    t
        Time in seconds.
    convention, frame, units
        Metadata tags; defaults freeze the canonical-v2 contract. Construction
        raises if ``convention`` is not ``"canonical-v2"`` so a consumer never
        silently mixes conventions.
    """

    q: npt.NDArray[np.float64]
    v: npt.NDArray[np.float64]
    a: npt.NDArray[np.float64]
    t: float = 0.0
    convention: str = CONVENTION_TAG_V2
    frame: str = "world_Zup"
    units: str = "SI"

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=float).reshape(-1)
        v = np.asarray(self.v, dtype=float).reshape(-1)
        a = np.asarray(self.a, dtype=float).reshape(-1)

        if q.shape[0] < _BASE_NQ:
            raise ValueError(
                f"q must have length at least {_BASE_NQ} (got {q.shape[0]})"
            )
        nv = q.shape[0] - 1
        if v.shape[0] != nv or a.shape[0] != nv:
            raise ValueError(
                f"nq must equal nv + 1: q has length {q.shape[0]} so v and a must "
                f"have length {nv} (got v={v.shape[0]}, a={a.shape[0]})"
            )
        for name, arr in (("q", q), ("v", v), ("a", a)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must contain only finite values")

        quat = q[3:_BASE_NQ]
        norm = float(np.linalg.norm(quat))
        if abs(norm - 1.0) > _QUAT_NORM_TOL:
            raise ValueError(
                f"base quaternion (q[3:7]) must have unit norm, got {norm:.6g}"
            )

        if self.convention != CONVENTION_TAG_V2:
            raise ValueError(
                f"convention must be {CONVENTION_TAG_V2!r}, got {self.convention!r}"
            )

        object.__setattr__(self, "q", _readonly(q))
        object.__setattr__(self, "v", _readonly(v))
        object.__setattr__(self, "a", _readonly(a))
        object.__setattr__(self, "t", float(self.t))

    # ---- shape accessors -----------------------------------------------------

    @property
    def nq(self) -> int:
        """Configuration dimension (``7 + n_joints``)."""
        return int(self.q.shape[0])

    @property
    def nv(self) -> int:
        """Tangent (velocity) dimension (``6 + n_joints``)."""
        return self.nq - 1

    @property
    def n_joints(self) -> int:
        """Number of internal joint coordinates."""
        return self.nq - _BASE_NQ

    @property
    def base_position(self) -> npt.NDArray[np.float64]:
        """Base position ``[x, y, z]`` in metres, world frame."""
        return self.q[0:3]

    @property
    def base_quat_wxyz(self) -> npt.NDArray[np.float64]:
        """Base orientation as a scalar-first unit quaternion ``(w, x, y, z)``."""
        return self.q[3:_BASE_NQ]

    @property
    def joint_q(self) -> npt.NDArray[np.float64]:
        """Internal joint coordinates (radians)."""
        return self.q[_BASE_NQ:]

    # ---- manifold operations -------------------------------------------------

    def integrate(self, dq: npt.ArrayLike) -> CanonicalState:
        """Return a new state with ``q`` advanced by tangent increment *dq*.

        *dq* has length :attr:`nv`: ``[base_lin(3), base_ang(3), joint(n_j)]``.
        Position and joint coordinates add directly; the base quaternion is
        updated on its manifold by right-multiplication with ``quat_exp(base_ang)``
        (body-frame increment). ``v``, ``a`` and ``t`` are carried unchanged.
        """
        d = np.asarray(dq, dtype=float).reshape(-1)
        if d.shape[0] != self.nv:
            raise ValueError(f"dq must have length nv={self.nv}, got {d.shape[0]}")
        new_pos = self.base_position + d[0:3]
        new_quat = se3.quat_normalize(
            se3.quat_multiply(self.base_quat_wxyz, se3.quat_exp(d[3:_BASE_NV]))
        )
        new_joints = self.joint_q + d[_BASE_NV:]
        new_q = np.concatenate([new_pos, new_quat, new_joints])
        return CanonicalState(
            q=new_q,
            v=self.v,
            a=self.a,
            t=self.t,
            convention=self.convention,
            frame=self.frame,
            units=self.units,
        )

    def difference(self, other: CanonicalState) -> npt.NDArray[np.float64]:
        """Return the tangent ``dq`` such that ``self.integrate(dq).q == other.q``.

        The inverse of :meth:`integrate`. The base rotation difference is taken
        on the manifold via :func:`pose_interchange.se3.quat_log`.
        """
        if other.n_joints != self.n_joints:
            raise ValueError(
                "difference requires states with the same n_joints "
                f"(self={self.n_joints}, other={other.n_joints})"
            )
        base_lin = other.base_position - self.base_position
        rel_quat = se3.quat_multiply(
            se3.quat_conjugate(self.base_quat_wxyz), other.base_quat_wxyz
        )
        base_ang = se3.quat_log(rel_quat)
        joint = other.joint_q - self.joint_q
        return np.concatenate([base_lin, base_ang, joint])

    # ---- constructors --------------------------------------------------------

    @classmethod
    def from_canonical_pose(cls, pose: object) -> CanonicalState:
        """Lift a ``canonical-v1`` :class:`CanonicalPose` to ``canonical-v2``.

        Maps the pelvis SE(3) (intrinsic-XYZ degrees -> quaternion) and joint
        angles (degrees -> radians, in reference-field order) into ``q`` with
        ``v = a = 0``. Imported locally so this module does not require the
        motion-matching reference-pose dependency unless the helper is used.
        """
        from src.shared.python.pose_interchange.canonical import CanonicalPose

        if not isinstance(pose, CanonicalPose):
            raise TypeError(f"pose must be a CanonicalPose, got {type(pose).__name__}")
        quat = se3.euler_xyz_deg_to_quat_wxyz(pose.pelvis_rotation_xyz_deg)
        joints_rad = np.array(list(pose.angles_full_dict_rad().values()), dtype=float)
        q = np.concatenate([pose.pelvis_translation_m, quat, joints_rad])
        nv = _BASE_NV + joints_rad.shape[0]
        zeros = np.zeros(nv)
        return cls(q=q, v=zeros, a=zeros.copy(), t=0.0)


def canonical_state_zero(n_joints: int) -> CanonicalState:
    """The neutral canonical-v2 state: base at origin, identity orientation, v=a=0."""
    if n_joints < 0:
        raise ValueError(f"n_joints must be non-negative, got {n_joints}")
    q = np.concatenate(
        [np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(n_joints)]
    )
    zeros = np.zeros(_BASE_NV + n_joints)
    return CanonicalState(q=q, v=zeros, a=zeros.copy(), t=0.0)
