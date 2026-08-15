"""Ball lie model for bunkershot3d (issue #8613).

Models the golf ball position and burial state in a sand bed.

Lie types:
- STANDARD: Ball sitting on surface (depth ~ 0)
- BURIED: "Fried egg" - partially buried in a crater
- PLUGGED: More than half the ball below surface
- TEED_UP: Ball sits on a mound above the surface

Physical properties match USGA/R&A regulations:
- Mass: 45.93g maximum
- Diameter: 42.67mm minimum
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

from src.shared.python.contracts import require

__all__ = [
    "BallLie",
    "BallLieType",
    "BallProperties",
    "compute_exposed_cap_area",
    "compute_exposed_cap_fraction",
    "compute_submersion_depth",
]

# USGA/R&A regulation ball properties (borrowed from rules, not measured)
_USGA_BALL_MASS_KG: float = 0.04593  # 45.93g max
_USGA_BALL_DIAMETER_M: float = 0.04267  # 42.67mm min


class BallLieType(enum.Enum):
    """Classification of ball lie in sand.

    Determines the physics model used for sand-ball interaction.
    """

    TEED_UP = "teed_up"  # Ball above surface (on mound)
    STANDARD = "standard"  # Ball sitting on surface
    BURIED = "buried"  # Partially buried ("fried egg")
    PLUGGED = "plugged"  # More than half buried


@dataclass(frozen=True, slots=True)
class BallProperties:
    """Physical properties of a golf ball.

    Attributes:
        mass_kg: Ball mass [kg]. Default is USGA maximum (45.93g).
        diameter_m: Ball diameter [m]. Default is USGA minimum (42.67mm).
    """

    mass_kg: float = _USGA_BALL_MASS_KG
    diameter_m: float = _USGA_BALL_DIAMETER_M

    def __post_init__(self) -> None:
        require(self.mass_kg > 0, "mass must be positive", self.mass_kg)
        require(self.diameter_m > 0, "diameter must be positive", self.diameter_m)

    @property
    def radius_m(self) -> float:
        """Ball radius [m]."""
        return self.diameter_m / 2.0

    @property
    def volume_m3(self) -> float:
        """Ball volume [m^3]."""
        return (4.0 / 3.0) * math.pi * self.radius_m**3

    @property
    def moi_kg_m2(self) -> float:
        """Moment of inertia about any diameter [kg.m^2].

        Solid sphere: I = (2/5) * m * r^2
        """
        return (2.0 / 5.0) * self.mass_kg * self.radius_m**2


@dataclass(slots=True)
class BallLie:
    """Ball position and lie in the sand bed.

    Coordinate system:
    - x: Forward (direction of play)
    - y: Left (perpendicular to play direction)
    - z: Up (normal to sand surface)

    Attributes:
        depth_m: How far the ball center is below the sand surface [m].
                 Positive = buried, negative = above surface.
        x_m: Ball center x position [m].
        y_m: Ball center y position [m].
    """

    depth_m: float
    x_m: float = 0.0
    y_m: float = 0.0
    _ball: BallProperties = BallProperties()

    def __post_init__(self) -> None:
        max_above = self._ball.radius_m
        max_below = self._ball.diameter_m
        require(
            -max_above <= self.depth_m <= max_below,
            f"depth must be in [{-max_above:.4f}, {max_below:.4f}] m",
            self.depth_m,
        )

    @property
    def lie_type(self) -> BallLieType:
        """Classify the lie based on burial depth."""
        r = self._ball.radius_m
        if self.depth_m < 0:
            return BallLieType.TEED_UP
        if self.depth_m < r * 0.3:  # Less than 30% of radius
            return BallLieType.STANDARD
        if self.depth_m < r:  # Less than half diameter
            return BallLieType.BURIED
        return BallLieType.PLUGGED

    def center_z_m(self, ball: BallProperties | None = None) -> float:
        """Compute ball center z-coordinate [m].

        Ball center sits at radius above the contact point, minus depth.
        A ball on the surface (depth=0) has center at z = radius.

        Args:
            ball: Ball properties. Uses default if None.

        Returns:
            Ball center z-coordinate [m].
        """
        b = ball or self._ball
        return b.radius_m - self.depth_m


def compute_submersion_depth(lie: BallLie, ball: BallProperties) -> float:
    """Compute how much of the ball is below the sand surface.

    For a ball sitting on the surface (depth=0), submersion is 0.
    For a half-buried ball (depth=radius), submersion is radius.
    For a fully buried ball (depth=diameter), submersion is diameter.

    Args:
        lie: Ball lie specification.
        ball: Ball physical properties.

    Returns:
        Submersion depth [m], clamped to [0, diameter].
    """
    if lie.depth_m <= 0:
        return 0.0
    return min(lie.depth_m, ball.diameter_m)


def _upper_hemisphere_area_m2(ball: BallProperties) -> float:
    """Curved area of the ball's upper hemisphere [m^2].

    The spherical-cap area ``A = 2 * pi * r * h`` evaluated at the cap
    height of a hemisphere, ``h = r``, which is the area a ball resting
    on an undisturbed surface presents to the air.
    """
    return 2.0 * math.pi * ball.radius_m**2


def _exposed_fraction(lie: BallLie, ball: BallProperties) -> float:
    """Share of the upper hemisphere still reachable by flying sand.

    ``1`` while the ball is at or above the surface, tapering linearly to
    ``0`` once it has sunk a full diameter and its top is level with the
    surface.

    Args:
        lie: Ball lie specification.
        ball: Ball physical properties.

    Returns:
        A fraction in ``[0, 1]``, decreasing in ``lie.depth_m``.
    """
    if lie.depth_m <= 0.0:
        return 1.0
    if lie.depth_m >= ball.diameter_m:
        return 0.0
    return 1.0 - lie.depth_m / ball.diameter_m


def compute_exposed_cap_area(lie: BallLie, ball: BallProperties) -> float:
    """Compute the ball area a sand splash can still transfer momentum to.

    The reference state is a ball resting on an undisturbed surface
    (``depth_m = 0``): its underside is against the sand and its upper
    hemisphere faces the air, so the exposed area is
    ``2 * pi * r^2``.  Burial eats into that hemisphere, and a ball sunk
    by a full diameter has its top level with the surface and exposes
    nothing.  Between those two ends the hemisphere is tapered linearly
    in depth::

        A(d) = 2 * pi * r^2 * (1 - d / D)   for 0 <= d <= D

    Why a linear taper rather than the spherical-cap height above the
    surface: the cap of height ``h = D - d`` gives ``A = 2 * pi * r * h``,
    which at ``d = 0`` is ``4 * pi * r^2`` -- the *whole* sphere,
    including the underside resting on the sand, which no splash can
    reach.  That form is therefore discontinuous with the reference state
    it should reduce to.  The linear taper is continuous at both ends and
    strictly decreasing in between, which is what the splash-transfer
    model downstream relies on.  It is a stated modelling convention, not
    a derivation from cap geometry.

    Args:
        lie: Ball lie specification.
        ball: Ball physical properties.

    Returns:
        Exposed area [m^2] in ``[0, 2 * pi * r^2]``, decreasing in
        ``lie.depth_m``.
    """
    return _upper_hemisphere_area_m2(ball) * _exposed_fraction(lie, ball)


def compute_exposed_cap_fraction(lie: BallLie, ball: BallProperties) -> float:
    """Compute the exposed share of the ball's upper hemisphere.

    :func:`compute_exposed_cap_area` normalised by the reference area of a
    ball resting on an undisturbed surface, so the result is the dimensionless
    taper itself: ``1`` at or above the surface, ``0`` once the ball has sunk a
    full diameter.

    :mod:`bunkershot3d.ball.splash` uses it as the share of the moving sand
    taken to be on a path that meets the ball. That reading is a **modelling
    convention**, stated as such in the launch provenance record: it is the
    same linear taper documented on :func:`compute_exposed_cap_area`, not a
    measured interception ratio.

    Args:
        lie: Ball lie specification.
        ball: Ball physical properties.

    Returns:
        A fraction in ``[0, 1]``, decreasing in ``lie.depth_m``.
    """
    return _exposed_fraction(lie, ball)
