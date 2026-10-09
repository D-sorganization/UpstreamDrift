"""Idealized face/loft coupling from rotation about a fixed shaft axis.

This is a geometric sensitivity, not a model of a golfer's grip, release,
shaft bend, impact contact, or measured clubhead delivery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DeliveredFace:
    """Target-frame face normal and its open-right angle and dynamic loft."""

    face_angle_deg: float
    dynamic_loft_deg: float
    shaft_rotation_deg: float
    face_normal: tuple[float, float, float]


def _shaft_axis(
    *, base_loft_deg: float, lie_deg: float, shaft_lean_deg: float
) -> tuple[float, float, float]:
    """Unit head-to-grip axis; +X target, +Y left, +Z up.

    Lie is shaft elevation above the ground, and positive lean means the
    grip's +X displacement divided by its +Z displacement is tan(lean).
    The golfer-side component is +Y. Lie and lean cannot be selected
    independently beyond the geometric domain of this construction.
    """
    values = (base_loft_deg, lie_deg, shaft_lean_deg)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("loft, lie, and shaft lean must be finite")
    if not -10.0 < base_loft_deg < 80.0:
        raise ValueError("base loft must be between -10 and 80 degrees")
    if not 0.0 < lie_deg < 90.0:
        raise ValueError("lie must be between 0 and 90 degrees")
    if not -45.0 < shaft_lean_deg < 45.0:
        raise ValueError("shaft lean must be between -45 and 45 degrees")
    lie = math.radians(lie_deg)
    sx = math.sin(lie) * math.tan(math.radians(shaft_lean_deg))
    sy_squared = math.cos(lie) ** 2 - sx**2
    if sy_squared <= 0.0:
        raise ValueError("shaft lean is incompatible with the specified lie")
    return sx, math.sqrt(sy_squared), math.sin(lie)


def delivery_from_shaft_rotation(
    shaft_rotation_deg: float,
    *,
    base_loft_deg: float,
    lie_deg: float,
    shaft_lean_deg: float,
) -> DeliveredFace:
    """Rotate a square face normal about the fixed head-to-grip shaft axis.

    Positive twist closes the face for the stated right-handed frame. The
    resulting face is open-right positive; loft is atan2(up, horizontal).
    The face is a rigid plane and is square at zero twist for each lean.
    """
    if not math.isfinite(shaft_rotation_deg) or abs(shaft_rotation_deg) > 30.0:
        raise ValueError("shaft rotation must be finite and within 30 degrees")
    sx, sy, sz = _shaft_axis(
        base_loft_deg=base_loft_deg,
        lie_deg=lie_deg,
        shaft_lean_deg=shaft_lean_deg,
    )
    loft = math.radians(base_loft_deg)
    nx, ny, nz = math.cos(loft), 0.0, math.sin(loft)
    twist = math.radians(shaft_rotation_deg)
    c, s = math.cos(twist), math.sin(twist)
    dot = sx * nx + sz * nz
    # Rodrigues' formula: n' = n cos(t) + (shaft × n) sin(t)
    #                       + shaft (shaft · n) (1 - cos(t)).
    x = nx * c + (sy * nz) * s + sx * dot * (1.0 - c)
    y = (sz * nx - sx * nz) * s + sy * dot * (1.0 - c)
    z = nz * c - (sy * nx) * s + sz * dot * (1.0 - c)
    return DeliveredFace(
        face_angle_deg=math.degrees(math.atan2(-y, x)),
        dynamic_loft_deg=math.degrees(math.atan2(z, math.hypot(x, y))),
        shaft_rotation_deg=shaft_rotation_deg,
        face_normal=(x, y, z),
    )


def delivery_from_face_angle(
    face_angle_deg: float,
    *,
    base_loft_deg: float,
    lie_deg: float,
    shaft_lean_deg: float,
) -> DeliveredFace:
    """Find the shaft twist giving the requested target-relative face angle.

    The scalar bisection is confined to the locally monotonic ±30° twist
    branch. A requested face angle outside that branch raises ValueError.
    """
    if not math.isfinite(face_angle_deg) or abs(face_angle_deg) > 15.0:
        raise ValueError("face angle must be finite and within 15 degrees")
    if face_angle_deg == 0.0:
        return delivery_from_shaft_rotation(
            0.0,
            base_loft_deg=base_loft_deg,
            lie_deg=lie_deg,
            shaft_lean_deg=shaft_lean_deg,
        )
    kwargs = {
        "base_loft_deg": base_loft_deg,
        "lie_deg": lie_deg,
        "shaft_lean_deg": shaft_lean_deg,
    }
    low = delivery_from_shaft_rotation(-30.0, **kwargs)
    high = delivery_from_shaft_rotation(30.0, **kwargs)
    if not high.face_angle_deg <= face_angle_deg <= low.face_angle_deg:
        raise ValueError("face angle cannot be reached on the local shaft-twist branch")
    lo, hi = -30.0, 30.0
    for _ in range(48):
        mid = (lo + hi) / 2.0
        state = delivery_from_shaft_rotation(mid, **kwargs)
        if state.face_angle_deg > face_angle_deg:
            lo = mid
        else:
            hi = mid
    return delivery_from_shaft_rotation((lo + hi) / 2.0, **kwargs)
