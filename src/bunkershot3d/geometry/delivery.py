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

Frame: ``+x`` is the target direction, ``+z`` is up, ``+y`` completes a
right-handed set (pointing away from the golfer).  The shaft axis is
``(0, -cos lambda, sin lambda)`` and a positive ``Omega`` opens the face.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .bounce import BounceAngle, GeometricBounce, MarketedBounce
from .wedge import WedgeGeometry

__all__ = [
    "DeliveredGeometry",
    "DeliveryCondition",
    "aim_offset_deg",
    "aim_offset_first_order_deg",
    "deliver_wedge",
    "delivered_face_normal",
    "delivered_sole_normal",
    "effective_bounce_deg",
    "effective_bounce_first_order_deg",
    "effective_loft_closed_form_deg",
    "effective_loft_deg",
    "effective_loft_first_order_deg",
]

_MAX_ATTACK_ANGLE_DEG = 90.0


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


def _delivery_rotation(
    lie_deg: float, face_open_deg: float, shaft_lean_deg: float
) -> NDArray[np.float64]:
    """Face opening about the shaft axis, then shaft lean as a pitch."""
    opening = _rotation_about(_shaft_axis_unit(lie_deg), math.radians(face_open_deg))
    lean_rad = math.radians(shaft_lean_deg)
    pitch = np.array(
        [
            [math.cos(lean_rad), 0.0, math.sin(lean_rad)],
            [0.0, 1.0, 0.0],
            [-math.sin(lean_rad), 0.0, math.cos(lean_rad)],
        ]
    )
    return pitch @ opening


def delivered_face_normal(
    *,
    loft_deg: float,
    lie_deg: float,
    face_open_deg: float,
    shaft_lean_deg: float = 0.0,
) -> NDArray[np.float64]:
    """Unit face normal after opening the face and leaning the shaft."""
    loft_rad = math.radians(loft_deg)
    static = np.array([math.cos(loft_rad), 0.0, math.sin(loft_rad)])
    return _delivery_rotation(lie_deg, face_open_deg, shaft_lean_deg) @ static


def delivered_sole_normal(
    *,
    bounce_deg: float,
    lie_deg: float,
    face_open_deg: float,
    shaft_lean_deg: float = 0.0,
) -> NDArray[np.float64]:
    """Unit outward (downward) sole normal after the same rotations."""
    bounce_rad = math.radians(bounce_deg)
    static = np.array([math.sin(bounce_rad), 0.0, -math.cos(bounce_rad)])
    return _delivery_rotation(lie_deg, face_open_deg, shaft_lean_deg) @ static


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
    return dihedral if normal[0] >= 0.0 else -dihedral


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
    """Exact aim opened by the face rotation, in degrees right of target."""
    normal = delivered_face_normal(
        loft_deg=loft_deg,
        lie_deg=lie_deg,
        face_open_deg=face_open_deg,
        shaft_lean_deg=shaft_lean_deg,
    )
    return math.degrees(math.atan2(float(normal[1]), float(normal[0])))


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
