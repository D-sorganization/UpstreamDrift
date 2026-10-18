"""Allocated trail-palm-on-lead-thumb force (HRG-H1, #13189). Reporting only.

In the full-body model the lead hand is the club body and the trail hand is
closed onto it by one rigid loop constraint, so only the total closure wrench
``lambda`` is observable: any palm-thumb force acts between the same two bodies
as the weld. The quantity reported here is therefore an **allocated equivalent
force**, ``method == "allocation"`` everywhere it is shown. It is not a measured
biological force, not a new contact, and it never enters the equations of
motion.

Definition (binding). ``force_on_lead_n`` and ``torque_on_lead_nm`` are the wrench
of the closing (trail) hand on the lead side, i.e. ``-lambda``, at the closure
site ``O``. It is carried to the patch point ``P`` as one force plus the
residual couple ``M_P = M_O - (P - O) x F`` (the couple a point contact cannot
carry is kept, not dropped). With ``n`` the unit normal from the thumb toward
the palm, a palm force pushes on the thumb along ``-n``:

* normal force ``F_n = max(0, -F . n)``; ``F . n >= 0`` is tension and is
  flagged ``unloaded`` rather than reported;
* tangential force ``F_t = |F - (F . n) n|`` (the part in the patch plane);
* friction ratio ``F_t / (mu F_n)`` against the spec ``static_friction``;
* share of the closure force ``F_n / |F|``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

METHOD = "allocation"
#: Tolerance on the unit length of the patch normal.
NORMAL_TOLERANCE = 1e-9
#: Closure forces below this (N) have no meaningful share or direction.
MIN_CLOSURE_FORCE_N = 1e-12

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class PalmThumbForce:
    """Allocated palm-on-thumb force at the patch, or why it is unavailable.

    An unavailable result carries ``available == False``, a non-empty ``reason``
    and ``None`` for every quantity (never a zero force). ``unloaded`` is
    ``True`` when the closure force pulls the patch apart.
    """

    available: bool
    reason: str = ""
    method: str = METHOD
    patch_point_m: Vec3 | None = None
    patch_normal: Vec3 | None = None
    force_on_lead_n: Vec3 | None = None
    couple_nm: Vec3 | None = None
    force_on_trail_n: Vec3 | None = None
    couple_on_trail_nm: Vec3 | None = None
    normal_force_n: float | None = None
    tangential_force_n: float | None = None
    unloaded: bool | None = None
    friction_ratio: float | None = None
    friction_reason: str = ""
    share_of_closure_force: float | None = None
    closure_force_n: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record; unavailable quantities stay ``None``, tuples become lists."""
        return {
            k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()
        }


def unavailable_palm_thumb(reason: str) -> PalmThumbForce:
    """A result that could not be resolved; ``reason`` must be non-empty."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("an unavailable palm-thumb force needs a non-empty reason")
    return PalmThumbForce(available=False, reason=reason)


def _vec(name: str, value: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _tuple(x: np.ndarray) -> Vec3:
    return (float(x[0]), float(x[1]), float(x[2]))


def _friction(
    tangential: float, normal: float, mu: float, unloaded: bool
) -> tuple[float | None, str]:
    if unloaded:
        return None, "unloaded: no normal force, so no friction-cone ratio"
    if mu == 0.0:
        return None, "static_friction is zero: the cone has no width"
    return tangential / (mu * normal), ""


def allocate_palm_thumb(
    *,
    force_on_lead_n: Sequence[float] | np.ndarray,
    torque_on_lead_nm: Sequence[float] | np.ndarray,
    reference_point_m: Sequence[float] | np.ndarray,
    patch_point_m: Sequence[float] | np.ndarray,
    patch_normal: Sequence[float] | np.ndarray,
    static_friction: float,
) -> PalmThumbForce:
    """Transport the closing-hand wrench to the patch and split it.

    All vectors share one frame. Preconditions: finite 3-vectors, a unit
    ``patch_normal`` (to 1e-9) and a finite ``static_friction >= 0``
    (``ValueError`` otherwise). Postconditions: ``force_on_lead_n`` and
    ``couple_nm`` plus ``(patch - reference) x force`` reproduce the input wrench
    exactly, and the trail-side reaction is the exact negative of both
    (Newton's third law).
    """
    force = _vec("force_on_lead_n", force_on_lead_n)
    torque = _vec("torque_on_lead_nm", torque_on_lead_nm)
    site = _vec("reference_point_m", reference_point_m)
    point = _vec("patch_point_m", patch_point_m)
    normal = _vec("patch_normal", patch_normal)
    if abs(float(np.linalg.norm(normal)) - 1.0) > NORMAL_TOLERANCE:
        raise ValueError("patch_normal must be a unit vector")
    if not (math.isfinite(static_friction) and static_friction >= 0.0):
        raise ValueError("static_friction must be finite and non-negative")
    couple = torque - np.cross(point - site, force)
    along = float(force @ normal)
    unloaded = along >= 0.0
    normal_force = 0.0 if unloaded else -along
    tangential = float(np.linalg.norm(force - along * normal))
    magnitude = float(np.linalg.norm(force))
    ratio, why = _friction(tangential, normal_force, static_friction, unloaded)
    share = None if magnitude < MIN_CLOSURE_FORCE_N else normal_force / magnitude
    return PalmThumbForce(
        available=True,
        patch_point_m=_tuple(point),
        patch_normal=_tuple(normal),
        force_on_lead_n=_tuple(force),
        couple_nm=_tuple(couple),
        force_on_trail_n=_tuple(-force),
        couple_on_trail_nm=_tuple(-couple),
        normal_force_n=normal_force,
        tangential_force_n=tangential,
        unloaded=unloaded,
        friction_ratio=ratio,
        friction_reason=why,
        share_of_closure_force=share,
        closure_force_n=magnitude,
    )


def allocate_from_closure(
    closure: Any,
    *,
    club_rotation: Sequence[Sequence[float]] | np.ndarray,
    club_origin: Sequence[float] | np.ndarray,
    patch: Any,
    static_friction: float,
) -> PalmThumbForce:
    """Allocate the palm-thumb force of a world-frame ``GripClosure``.

    The load on the lead side (the club) is the closure's second wrench,
    ``-lambda`` at the club site. ``patch`` (``point_club_m``, ``normal_club``) is
    carried to the world by ``club_rotation`` and ``club_origin``. An
    unavailable closure, or one without a torque, gives an unavailable result
    with the reason (never a zero force). Preconditions: ``club_rotation`` a
    proper 3x3 rotation, ``club_origin`` a finite 3-vector (``ValueError``).
    """
    rot = np.asarray(club_rotation, dtype=float)
    if rot.shape != (3, 3) or not np.isfinite(rot).all():
        raise ValueError("club_rotation must be a finite 3x3 matrix")
    if not (
        np.allclose(rot.T @ rot, np.eye(3), atol=NORMAL_TOLERANCE)
        and math.isclose(float(np.linalg.det(rot)), 1.0, abs_tol=NORMAL_TOLERANCE)
    ):
        raise ValueError("club_rotation must be a proper rotation")
    origin = _vec("club_origin", club_origin)
    if not closure.available:
        return unavailable_palm_thumb(closure.reason or "closure unavailable")
    on_club = closure.wrenches[-1]
    if on_club.force_n is None or on_club.torque_nm is None:
        return unavailable_palm_thumb("closure wrench has no force or torque half")
    return allocate_palm_thumb(
        force_on_lead_n=on_club.force_n,
        torque_on_lead_nm=on_club.torque_nm,
        reference_point_m=on_club.point_m,
        patch_point_m=rot @ _vec("patch.point_club_m", patch.point_club_m) + origin,
        patch_normal=rot @ _vec("patch.normal_club", patch.normal_club),
        static_friction=static_friction,
    )
