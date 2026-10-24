"""Engine-agnostic grip interface description (issue #11739, OSV-7).

One source of truth for the hand-club interface: per-hand grip frames on the
club, bushing parameters and contact material.  Positions come from the
full-body spec (never invented): the right hand grip point is the closure
``placement_b`` on the club body, the left hand grip point is the origin of the
left wrist joint's club-side frame (the club is that joint's child).  Both
frames share the closure rotation, whose x axis is the shaft axis, so bushing
axes are (along grip, across, across) for both hands.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.shared.python.grip_contact.club_dynamics import ClubDynamics
from src.shared.python.grip_contact.damping import (
    DEFAULT_DAMPING_RATIO,
    design_damping,
)
from src.shared.python.grip_contact.parameters import (
    BushingParameters,
    ContactMaterial,
    Vec3,
    default_bushing,
)

SIDES = ("L", "R")

#: Orthonormal permutation matrix converting from appearance grip frame
#: (z = grip axis butt->head, x = toe axis, y = trail axis)
#: to shared grip frame (x = grip axis butt->head, y = toe axis, z = trail axis).
R_APPEARANCE_TO_SHARED = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float
)
#: Inverse permutation converting from shared grip frame to appearance grip frame.
R_SHARED_TO_APPEARANCE = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=float
)


def _require_proper_rotation(rot: np.ndarray, what: str) -> None:
    """Raise ``ValueError`` unless ``rot`` is orthonormal with determinant +1."""
    if not (
        np.allclose(rot.T @ rot, np.eye(3), atol=1e-9)
        and math.isclose(float(np.linalg.det(rot)), 1.0, abs_tol=1e-9)
    ):
        raise ValueError(f"{what} must be a proper rotation")


def _relabel_grip_frame(
    target: Sequence[float] | np.ndarray, to_new: np.ndarray, from_new: np.ndarray
) -> np.ndarray:
    """Express ``target`` in a relabelled grip frame.

    ``to_new`` maps old-frame coordinates to new-frame coordinates and
    ``from_new`` is its inverse. Vectors and ``(N, 3)`` points are mapped by
    ``to_new``; a 3x3 rotation or 4x4 transform whose *child* frame is the grip
    frame keeps its translation and has its rotation post-multiplied by
    ``from_new``.
    """
    arr = np.asarray(target, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("input contains non-finite values")
    if arr.shape == (3,):
        return to_new @ arr
    if arr.shape == (3, 3):
        _require_proper_rotation(arr, "3x3 matrix")
        return arr @ from_new
    if arr.shape == (4, 4):
        _require_proper_rotation(arr[:3, :3], "4x4 matrix rotation")
        out = np.eye(4)
        out[:3, :3] = arr[:3, :3] @ from_new
        out[:3, 3] = arr[:3, 3]
        return out
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr @ to_new.T
    raise ValueError(
        f"unsupported shape {arr.shape}; expected (3,), (3, 3), (4, 4) or (N, 3)"
    )


def appearance_to_shared_grip_frame(
    target: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Convert a vector, points, rotation or transform to the shared grip frame.

    The appearance convention (:mod:`model_appearance.grip_hands` and meshes)
    places ``z`` along the grip axis (butt toward head), ``x`` along the toe
    axis and ``y`` along the trail axis. The shared convention
    (:mod:`grip_contact.pad_layout` and :class:`GripFrame`) places ``x`` along
    the grip axis, ``y`` along the toe axis and ``z`` along the trail axis.

    Accepts a finite 3-vector, an ``(N, 3)`` array of points, a proper 3x3
    rotation or a 4x4 transform with a proper rotation (child frame = grip
    frame). Raises ``ValueError`` for non-finite input, an unsupported shape or
    an improper rotation.

    Postcondition: ``shared_to_appearance_grip_frame`` is the exact inverse.
    """
    return _relabel_grip_frame(target, R_APPEARANCE_TO_SHARED, R_SHARED_TO_APPEARANCE)


def shared_to_appearance_grip_frame(
    target: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Exact inverse of :func:`appearance_to_shared_grip_frame`."""
    return _relabel_grip_frame(target, R_SHARED_TO_APPEARANCE, R_APPEARANCE_TO_SHARED)


@dataclass(frozen=True)
class GripFrame:
    """Grip frame on the club: pose of the grip frame in the club body frame."""

    side: str
    position_m: Vec3
    rotation: tuple[Vec3, Vec3, Vec3]  # rows of R (club <- grip)

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise ValueError(f"side must be 'L' or 'R', got {self.side!r}")
        pos = np.asarray(self.position_m, dtype=float)
        rot = np.asarray(self.rotation, dtype=float)
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError("position_m must be a finite 3-vector")
        if rot.shape != (3, 3) or not np.isfinite(rot).all():
            raise ValueError("rotation must be a finite 3x3 matrix")
        if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-9) or not math.isclose(
            float(np.linalg.det(rot)), 1.0, abs_tol=1e-9
        ):
            raise ValueError("rotation must be a proper rotation matrix")

    def matrix(self) -> np.ndarray:
        """4x4 homogeneous pose of the grip frame in the club frame."""
        out = np.eye(4)
        out[:3, :3] = np.asarray(self.rotation, dtype=float)
        out[:3, 3] = self.position_m
        return out


@dataclass(frozen=True)
class GripInterface:
    """Left and right grip frames plus compliance and contact parameters."""

    left: GripFrame
    right: GripFrame
    bushing: BushingParameters = field(default_factory=default_bushing)
    contact_material: ContactMaterial = field(default_factory=ContactMaterial)

    def __post_init__(self) -> None:
        if self.left.side != "L" or self.right.side != "R":
            raise ValueError("left/right slots require frames with side L/R")
        sep = np.linalg.norm(
            np.asarray(self.left.position_m) - np.asarray(self.right.position_m)
        )
        if sep <= 1e-6:
            raise ValueError("left and right grip frames must be distinct")

    def frame(self, side: str) -> GripFrame:
        """Return the grip frame of ``side`` ('L' or 'R')."""
        if side not in SIDES:
            raise ValueError(f"side must be 'L' or 'R', got {side!r}")
        return self.left if side == "L" else self.right

    @property
    def hand_separation_m(self) -> float:
        """Distance between the two grip points."""
        return float(
            np.linalg.norm(
                np.asarray(self.left.position_m) - np.asarray(self.right.position_m)
            )
        )

    @classmethod
    def from_spec(
        cls,
        spec: Mapping[str, Any],
        *,
        bushing: BushingParameters | None = None,
        contact_material: ContactMaterial | None = None,
        damping_ratio: float = DEFAULT_DAMPING_RATIO,
    ) -> GripInterface:
        """Build from a ``full-body-v1`` spec (closure plus club-parented joint).

        Raises:
            ValueError: if the closure or the club's parent joint is missing.
        """
        closure = spec.get("closure")
        if not closure:
            raise ValueError("spec has no closure")
        club = closure["body_b"]
        t_right = np.asarray(closure["placement_b"], dtype=float)
        gb = spec.get("grip_bushing")
        if (
            isinstance(gb, Mapping)
            and isinstance(gb.get("L"), Mapping)
            and "club_frame" in gb["L"]
        ):
            p_left = np.asarray(gb["L"]["club_frame"], dtype=float)[:3, 3]
        elif isinstance(gb, Mapping) and "club_left_frame" in gb:
            p_left = np.asarray(gb["club_left_frame"], dtype=float)[:3, 3]
        else:
            joints = [j for j in spec.get("joints", []) if j.get("child") == club]
            if len(joints) != 1:
                raise ValueError(
                    f"expected exactly one joint parenting the club, found {len(joints)}"
                )
            p_left = np.asarray(joints[0]["child_to_follower"], dtype=float)[:3, 3]
        rot = tuple(tuple(float(x) for x in row) for row in t_right[:3, :3])
        left = GripFrame("L", _v3(p_left), rot)  # type: ignore[arg-type]
        right = GripFrame("R", _v3(t_right[:3, 3]), rot)  # type: ignore[arg-type]
        if bushing is None:
            bushing = design_damping(
                default_bushing(),
                left,
                right,
                ClubDynamics.from_spec(spec),
                damping_ratio,
            )
        return cls(
            left=left,
            right=right,
            bushing=bushing,
            contact_material=contact_material or ContactMaterial(),
        )


def _v3(a: np.ndarray) -> Vec3:
    return (float(a[0]), float(a[1]), float(a[2]))
