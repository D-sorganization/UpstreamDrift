"""Delivered (effective) loft, bounce and aim (issue #8609).

Opening the face is a rotation of the **rigid head about the shaft
axis**, not about the vertical, which is why a wedge gains far less loft
than the amount it is opened.  With loft ``L``, lie ``lambda`` (shaft to
ground) and face-open angle ``Omega``:

    L_eff = arcsin[ sin L cos(Om)
                  + cos L cos(lam) sin(Om)
                  + sin^2(lam) sin L (1 - cos Om) ]

    dloft ~ dbounce ~ Om cos(lam)        daim ~ Om sin(lam)

At a wedge lie of 64 deg a 20 deg open face buys 8.5 deg of loft and
costs 18 deg of aim.  Shaft lean ``S`` is a pure pitch and subtracts
degree for degree from both loft and bounce.

Frame: the **head frame**, the one the mesh is built in -- origin at the
leading-edge point, ``+x`` rearward, ``+y`` heel to toe, ``+z`` up (see
:mod:`bunkershot3d.geometry.profile` and
:mod:`bunkershot3d.geometry.lofting`).  The head therefore **travels
toward** ``-x``: it strikes leading edge first, which is what
:data:`TRAVEL_AXIS_BODY` states and :func:`entry_velocity_m_s` builds.
The shaft axis is ``(0, -cos lambda, sin lambda)``, the same vector
:func:`bunkershot3d.geometry.lofting.shaft_axis` returns.

Until issue #9247 this module was written in a *mirrored* frame, with
``+x`` toward the target.  Its scalars were right and its vectors were
backwards, so a caller who applied :func:`delivered_rotation`'s
composition to a real mesh opened the face and leaned the shaft the
wrong way, and a caller who took ``+x`` for the travel direction drove
the head through the sand trailing edge first.  That inverted the
primary design variable: bounce dug instead of skidding.  Every angle
this module returns is unchanged; only the vectors' ``x`` components
moved, which is the whole content of the mirror.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .bounce import BounceAngle, GeometricBounce, MarketedBounce
from .wedge import WedgeGeometry

__all__ = [
    "TRAVEL_AXIS_BODY",
    "DeliveredGeometry",
    "DeliveryCondition",
    "aim_offset_deg",
    "aim_offset_first_order_deg",
    "deliver_wedge",
    "delivered_face_normal",
    "delivered_rotation",
    "delivered_sole_normal",
    "effective_bounce_deg",
    "effective_bounce_first_order_deg",
    "effective_loft_closed_form_deg",
    "effective_loft_deg",
    "effective_loft_first_order_deg",
    "entry_velocity_m_s",
]

_MAX_ATTACK_ANGLE_DEG = 90.0

TRAVEL_AXIS_BODY: tuple[float, float, float] = (-1.0, 0.0, 0.0)
"""Horizontal direction the head travels, in head coordinates.

The mesh puts the leading edge at the origin with ``+x`` rearward, so a
wedge that strikes leading edge first travels toward ``-x``.  Stated once
here because getting it backwards is issue #9247: driven ``+x`` the sole
is a ramp descending into the bed and more bounce digs deeper, which
inverts the tool's primary design variable.  A tuple rather than an
array so that no caller can mutate the convention in place.
"""


def _require_angle(name: str, value: float, limit: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if abs(number) > limit:
        raise ValueError(f"{name} must lie within +/-{limit} degrees, got {number}")
    return number


def _shaft_axis_unit(lie_deg: float) -> NDArray[np.float64]:
    """Unit shaft axis for a lie angle measured shaft-to-ground."""
    lie_rad = math.radians(lie_deg)
    return np.array([0.0, -math.cos(lie_rad), math.sin(lie_rad)])


def _rotation_about(axis: NDArray[np.float64], angle_rad: float) -> NDArray[np.float64]:
    """Rodrigues rotation matrix about a unit axis."""
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3)
        + math.sin(angle_rad) * skew
        + (1.0 - math.cos(angle_rad)) * (skew @ skew)
    )


def delivered_rotation(
    *, lie_deg: float, face_open_deg: float, shaft_lean_deg: float = 0.0
) -> NDArray[np.float64]:
    """Body-to-world rotation of a delivered head, in the head frame.

    Face opening about the shaft axis, then shaft lean as a pitch.  Both
    senses are negated against the mirrored frame this module used to be
    written in: rotations about an axis lying in the ``y-z`` plane -- the
    shaft axis, and ``y`` itself -- reverse under the ``x`` mirror, so
    ``+Omega`` opens the face and ``+S`` de-lofts *only* when the head's
    own ``+x``-rearward axes are used.  Applying the un-mirrored
    composition to a lofted mesh leaned the shaft backwards, which is
    half of issue #9247.

    This is the one definition of a delivered pose.  It used to be
    private, and the workbench reimplemented it from this module's
    docstring; that copy is what carried the mirror across the boundary.

    Args:
        lie_deg: Design lie, shaft to ground.
        face_open_deg: Rotation about the shaft axis; positive opens.
        shaft_lean_deg: Forward lean; positive de-lofts.

    Returns:
        ``(3, 3)`` rotation taking head coordinates to world.
    """
    opening = _rotation_about(_shaft_axis_unit(lie_deg), -math.radians(face_open_deg))
    lean_rad = -math.radians(shaft_lean_deg)
    pitch = np.array(
        [
            [math.cos(lean_rad), 0.0, math.sin(lean_rad)],
            [0.0, 1.0, 0.0],
            [-math.sin(lean_rad), 0.0, math.cos(lean_rad)],
        ]
    )
    return np.asarray(pitch @ opening, dtype=np.float64)


def entry_velocity_m_s(
    *, speed_m_s: float, attack_angle_deg: float
) -> NDArray[np.float64]:
    """Delivered velocity in head coordinates, leading edge first.

    Args:
        speed_m_s: Clubhead speed at impact.
        attack_angle_deg: Club-path angle to the horizontal, **negative
            for a descending blow**.

    Returns:
        ``(3,)`` velocity along :data:`TRAVEL_AXIS_BODY`, descending.

    Raises:
        ValueError: If the speed is not positive and finite, or the
            attack angle is not a descending blow.  A level or rising
            head never enters the sand, so there is no strike to place.
    """
    speed = float(speed_m_s)
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError(f"speed_m_s must be positive, got {speed_m_s!r}")
    attack = _require_angle(
        "attack_angle_deg", attack_angle_deg, _MAX_ATTACK_ANGLE_DEG - 1.0
    )
    if attack >= 0.0:
        raise ValueError(
            "attack_angle_deg must be negative for a descending blow; a level "
            f"or rising head never enters the sand, got {attack_angle_deg!r}"
        )
    attack_rad = math.radians(attack)
    horizontal = speed * math.cos(attack_rad)
    return np.array(
        [
            TRAVEL_AXIS_BODY[0] * horizontal,
            TRAVEL_AXIS_BODY[1] * horizontal,
            speed * math.sin(attack_rad),
        ],
        dtype=np.float64,
    )


def delivered_face_normal(
    *,
    loft_deg: float,
    lie_deg: float,
    face_open_deg: float,
    shaft_lean_deg: float = 0.0,
) -> NDArray[np.float64]:
    """Unit face normal after opening the face and leaning the shaft.

    The face looks along the travel direction, so a square, unlofted face
    normal is ``(-1, 0, 0)`` and loft tips it up.
    """
    loft_rad = math.radians(loft_deg)
    static = np.array([-math.cos(loft_rad), 0.0, math.sin(loft_rad)])
    return np.asarray(
        delivered_rotation(
            lie_deg=lie_deg,
            face_open_deg=face_open_deg,
            shaft_lean_deg=shaft_lean_deg,
        )
        @ static,
        dtype=np.float64,
    )


def delivered_sole_normal(
    *,
    bounce_deg: float,
    lie_deg: float,
    face_open_deg: float,
    shaft_lean_deg: float = 0.0,
) -> NDArray[np.float64]:
    """Unit outward (downward) sole normal after the same rotations.

    Positive bounce lifts the leading edge, so the sole's outward normal
    dips toward the travel direction ``-x``.
    """
    bounce_rad = math.radians(bounce_deg)
    static = np.array([-math.sin(bounce_rad), 0.0, -math.cos(bounce_rad)])
    return np.asarray(
        delivered_rotation(
            lie_deg=lie_deg,
            face_open_deg=face_open_deg,
            shaft_lean_deg=shaft_lean_deg,
        )
        @ static,
        dtype=np.float64,
    )


def effective_loft_closed_form_deg(
    *, loft_deg: float, lie_deg: float, face_open_deg: float
) -> float:
    """The exact arcsin relation for delivered loft (no shaft lean).

    This is the published closed form; :func:`effective_loft_deg` obtains
    the same number from the rigid-body rotation and the two agree to
    ~1e-11 degrees, which the test suite pins.
    """
    loft_rad = math.radians(loft_deg)
    lie_rad = math.radians(lie_deg)
    open_rad = math.radians(face_open_deg)
    value = (
        math.sin(loft_rad) * math.cos(open_rad)
        + math.cos(loft_rad) * math.cos(lie_rad) * math.sin(open_rad)
        + math.sin(lie_rad) ** 2 * math.sin(loft_rad) * (1.0 - math.cos(open_rad))
    )
    return math.degrees(math.asin(max(-1.0, min(1.0, value))))


def effective_loft_deg(
    *,
    loft_deg: float,
    lie_deg: float,
    face_open_deg: float,
    shaft_lean_deg: float = 0.0,
) -> float:
    """Delivered loft: the face normal's elevation above the horizontal."""
    normal = delivered_face_normal(
        loft_deg=loft_deg,
        lie_deg=lie_deg,
        face_open_deg=face_open_deg,
        shaft_lean_deg=shaft_lean_deg,
    )
    return math.degrees(math.asin(float(np.clip(normal[2], -1.0, 1.0))))


def effective_loft_first_order_deg(
    *,
    loft_deg: float,
    lie_deg: float,
    face_open_deg: float,
    shaft_lean_deg: float = 0.0,
) -> float:
    """First-order delivered loft: ``L + Om cos(lam) - S``."""
    return (
        float(loft_deg)
        + float(face_open_deg) * math.cos(math.radians(lie_deg))
        - float(shaft_lean_deg)
    )


def effective_bounce_deg(
    *,
    bounce_deg: float,
    lie_deg: float,
    face_open_deg: float,
    shaft_lean_deg: float = 0.0,
) -> float:
    """Delivered bounce: the sole plane's dihedral angle to the ground.

    Signed by the dip along the target line, so a forward shaft lean can
    drive it negative - the leading edge then sits below the sole's
    contact plane and the club digs.
    """
    normal = delivered_sole_normal(
        bounce_deg=bounce_deg,
        lie_deg=lie_deg,
        face_open_deg=face_open_deg,
        shaft_lean_deg=shaft_lean_deg,
    )
    dihedral = math.degrees(math.acos(float(np.clip(-normal[2], -1.0, 1.0))))
    return dihedral if -normal[0] >= 0.0 else -dihedral


def effective_bounce_first_order_deg(
    *,
    bounce_deg: float,
    lie_deg: float,
    face_open_deg: float,
    shaft_lean_deg: float = 0.0,
) -> float:
    """First-order delivered bounce: ``B + Om cos(lam) - S``."""
    return (
        float(bounce_deg)
        + float(face_open_deg) * math.cos(math.radians(lie_deg))
        - float(shaft_lean_deg)
    )


def aim_offset_deg(
    *,
    loft_deg: float,
    lie_deg: float,
    face_open_deg: float,
    shaft_lean_deg: float = 0.0,
) -> float:
    """Exact aim opened by the face rotation, in degrees right of target.

    Measured from the travel direction ``-x``, which is why the first
    argument to ``atan2`` is the normal's ``x`` component negated.
    """
    normal = delivered_face_normal(
        loft_deg=loft_deg,
        lie_deg=lie_deg,
        face_open_deg=face_open_deg,
        shaft_lean_deg=shaft_lean_deg,
    )
    return math.degrees(math.atan2(float(normal[1]), -float(normal[0])))


def aim_offset_first_order_deg(*, lie_deg: float, face_open_deg: float) -> float:
    """First-order aim offset: ``Om sin(lam)``."""
    return float(face_open_deg) * math.sin(math.radians(lie_deg))


@dataclass(frozen=True, slots=True)
class DeliveryCondition:
    """How the head is presented at impact.

    Attributes:
        face_open_deg: Rotation about the shaft axis; positive opens.
        shaft_lean_deg: Forward lean; positive de-lofts (tour: 4-14 deg).
        attack_angle_deg: Club-path angle to the horizontal, **negative
            for a descending blow** (tour: -2 to -12 deg).
    """

    face_open_deg: float = 0.0
    shaft_lean_deg: float = 0.0
    attack_angle_deg: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "face_open_deg",
            _require_angle("face_open_deg", self.face_open_deg, 90.0),
        )
        object.__setattr__(
            self,
            "shaft_lean_deg",
            _require_angle("shaft_lean_deg", self.shaft_lean_deg, 60.0),
        )
        object.__setattr__(
            self,
            "attack_angle_deg",
            _require_angle(
                "attack_angle_deg", self.attack_angle_deg, _MAX_ATTACK_ANGLE_DEG - 1.0
            ),
        )


@dataclass(frozen=True, slots=True)
class DeliveredGeometry:
    """What the sand actually sees.

    Attributes:
        effective_loft_deg: Delivered loft.
        effective_bounce: Delivered bounce, tagged with the same
            convention as the static bounce it came from.
        aim_offset_deg: How far right of target the face points.
        presentation_bounce_deg: Delivered bounce plus the attack angle.
            Positive means the sole still presents a skidding surface
            relative to the club path; negative means the leading edge
            leads into the sand and the club digs.  (DRFT's per-element
            attack angle beta is a different quantity - the angle between
            a surface element and the velocity vector - and is computed
            by the solver, not here.)
    """

    effective_loft_deg: float
    effective_bounce: BounceAngle
    aim_offset_deg: float
    presentation_bounce_deg: float


def deliver_wedge(
    geometry: WedgeGeometry,
    condition: DeliveryCondition,
    *,
    use_geometric_bounce: bool = False,
) -> DeliveredGeometry:
    """Apply a delivery condition to a wedge's static geometry.

    Args:
        geometry: The wedge design vector.
        condition: Face opening, shaft lean and attack angle.
        use_geometric_bounce: Report the patent (geometric) convention
            instead of the marketed one.  The convention is chosen here,
            explicitly, and carried by the returned type.

    Returns:
        The delivered loft, bounce, aim and presentation angle.
    """
    if not isinstance(geometry, WedgeGeometry):
        raise TypeError(f"expected a WedgeGeometry, got {type(geometry).__name__}")
    if not isinstance(condition, DeliveryCondition):
        raise TypeError(f"expected a DeliveryCondition, got {type(condition).__name__}")

    static: BounceAngle = (
        geometry.geometric_bounce if use_geometric_bounce else geometry.marketed_bounce
    )
    delivered_bounce_deg = effective_bounce_deg(
        bounce_deg=static.angle_deg,
        lie_deg=geometry.lie_deg,
        face_open_deg=condition.face_open_deg,
        shaft_lean_deg=condition.shaft_lean_deg,
    )
    bounce_type = GeometricBounce if use_geometric_bounce else MarketedBounce
    return DeliveredGeometry(
        effective_loft_deg=effective_loft_deg(
            loft_deg=geometry.loft_deg,
            lie_deg=geometry.lie_deg,
            face_open_deg=condition.face_open_deg,
            shaft_lean_deg=condition.shaft_lean_deg,
        ),
        effective_bounce=bounce_type(delivered_bounce_deg),
        aim_offset_deg=aim_offset_deg(
            loft_deg=geometry.loft_deg,
            lie_deg=geometry.lie_deg,
            face_open_deg=condition.face_open_deg,
            shaft_lean_deg=condition.shaft_lean_deg,
        ),
        presentation_bounce_deg=delivered_bounce_deg + condition.attack_angle_deg,
    )
