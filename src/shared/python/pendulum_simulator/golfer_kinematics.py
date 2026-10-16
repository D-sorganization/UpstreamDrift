"""Forward kinematics for golfer upper-body model.

Topology (from the sketch)
--------------------------
Fixed hub connects via a standoff to two shoulder joints that branch
into independent arm chains.  Both wrist endpoints attach to different
points on a shared club segment, closing the kinematic loop.

    Hub ─── RS (right shoulder)
     │           │
     │          RE (right elbow)
     │           │
     │          RH (right hand / wrist)──┐
     │                                   Club ── Clubhead
     │          LH (left hand / wrist)───┘
     │           │
     │          LE (left elbow)
     │           │
     └── LS (left shoulder)

Coordinate convention:
    Angles measured from downward vertical, positive counterclockwise.
    World frame: x→right, y→up, origin at hub.
    A segment of absolute angle ``a`` points along ``u(a) = (sin a, -cos a)``;
    ``theta_club`` is the absolute club angle, so every club point is
    ``rh + (s - grip_right) * u(theta_club)`` for an arc length ``s`` measured
    from the club base (#12608 fixed a flipped y sign here).

Closure decision (#12608, NSX-00 ``PLANAR_POINT2``):
    The hands do not enter the loop closure.  ``rh`` and ``lh`` are forearm
    end points (the wrist pins); the club is pinned at both wrists, exactly as
    the NSX-00 planar two-arm fixture closes the trail hand onto a grip point.
    The hand angles ``alpha_rh``/``alpha_lh`` are free hand DOFs that move no
    point of this model, so their constraint-Jacobian columns are zero.  The
    closure therefore has two independent rows and its Jacobian has rank 2 on
    the constraint manifold; ``golfer_constraints`` rows 2-3 are an orthogonal
    rotation of rows 0-1 and add no rank there.

Canonical source (#12671, ADR-0056):
    The forward kinematics are implemented once, in the vendored Tools package
    ``double_pendulum_golf.golfer_kinematics``.  This module keeps the
    UpstreamDrift import path as a thin adapter: ``forward_kinematics`` and
    ``_hub_position`` delegate to Tools.  The backend-agnostic club helpers
    below stay here because the JAX twin and the native backend need a
    traceable, contract-checked form of the same formula; the parity test
    ``test_golfer_canonical_parity`` pins them to the Tools club points.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from src.shared.python.core.contracts import ensure, require

from ._canonical_golfer import delegate
from .physics_golfer import GolferParams

__all__ = [
    "club_shaft_offset",
    "club_shaft_offset_derivative",
    "forward_kinematics",
    "point_on_club",
]

#: Hub endpoint (end of the standoff from the fixed origin); Tools canonical.
_hub_position: Callable[[float, GolferParams], tuple[float, float]] = delegate(
    "golfer_kinematics", "_hub_position"
)

#: All joint positions in the world frame; Tools canonical.  ``q`` has shape
#: ``(8,)`` or ``(16,)``; the keys are ``origin``, ``hub``, ``rs``, ``re``,
#: ``rh``, ``ls``, ``le``, ``lh``, ``club_base``, ``club_tip``, ``grip_right``,
#: ``grip_left`` and, when the scapula links are present, ``rscap``/``lscap``.
forward_kinematics: Callable[
    [np.ndarray, GolferParams], dict[str, tuple[float, float]]
] = delegate("golfer_kinematics", "forward_kinematics")


def club_shaft_offset(theta_club: Any, offset: Any, xp: Any = np) -> tuple[Any, Any]:
    """Displacement ``offset * u(theta_club)`` along the shaft, ``u = (sin, -cos)``.

    The single formula for club points (#12608, #12669).  It is
    backend-agnostic, so ``xp`` may be ``numpy`` or ``jax.numpy``, and it carries
    no contracts so JAX can trace it; ``point_on_club`` adds them for NumPy.
    """
    return offset * xp.sin(theta_club), -offset * xp.cos(theta_club)


def club_shaft_offset_derivative(
    theta_club: Any, offset: Any, xp: Any = np
) -> tuple[Any, Any]:
    """``d/d theta_club`` of ``club_shaft_offset``: ``offset * (cos, sin)``.

    This is the club-angle column of every club-point Jacobian.  It is
    backend-agnostic in the same way as ``club_shaft_offset``.
    """
    return offset * xp.cos(theta_club), offset * xp.sin(theta_club)


def point_on_club(
    anchor: tuple[float, float], theta_club: float, offset: float
) -> tuple[float, float]:
    """Point ``offset`` metres from ``anchor`` along the club shaft.

    The shaft direction is ``u(theta_club) = (sin, -cos)``, the same direction
    as every arm segment and as ``golfer_constraints`` rows 2-3.

    Preconditions:
        ``theta_club`` and ``offset`` are finite.
    Postconditions:
        The returned point is exactly ``|offset|`` from ``anchor``.
    """
    require(
        bool(np.isfinite(theta_club) and np.isfinite(offset)),
        "theta_club and offset must be finite",
        (theta_club, offset),
    )
    dx, dy = club_shaft_offset(theta_club, offset)
    x = anchor[0] + float(dx)
    y = anchor[1] + float(dy)
    ensure(
        bool(np.isclose(np.hypot(x - anchor[0], y - anchor[1]), abs(offset))),
        "club point must lie |offset| from its anchor",
        (x, y),
    )
    return (x, y)
