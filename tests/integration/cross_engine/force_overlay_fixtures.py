"""Shared synthetic fixtures for the force-overlay parity suite (FTO-21, #11306).

Every engine row of ``test_force_overlay_parity.py`` is built from the **one**
parameter set below, so all engines are asked the same physical question.
All fixtures are ``synthetic_``: analytic statics, not measured data.

World frame is Z-up (ADR-0026). The pendulum hinge axis is world +y and a
joint angle ``q = 0`` hangs the link straight down (-z). The link's centre of
mass sits half a link length from the pivot.

Analytic expectations (the issue's physics):

* hanging at rest: joint reaction on the link is ``(0, 0, +m*g)`` at the
  pivot and the proximal axial load is ``+m*g`` (tension);
* held inverted at ``theta`` from vertical (``q = pi - theta``): the actuator
  torque about the hinge axis is ``+m*g*(l/2)*sin(theta)`` (it opposes the
  gravity moment) and the axial load is ``-m*g*cos(theta)`` (compression);
* resting body: contact forces sum to ``(0, 0, +m*g)`` on the ground plane.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

#: Body name every engine builder must give the pendulum link.
LINK = "link"
#: Joint name every engine builder must give the pendulum hinge.
HINGE = "hinge"
#: Name of the resting body in the contact fixture.
BOX = "box"


@dataclass(frozen=True)
class PendulumFixture:
    """Single-link pendulum on a hinge about world +y, plus a resting body.

    Preconditions (validated once, here): ``mass``, ``length``, ``gravity``,
    ``pivot_height`` and ``box_side`` are positive and finite; ``theta`` is
    strictly between 0 and pi/2 so ``sin`` and ``cos`` are both positive.
    """

    mass: float = 2.0
    length: float = 1.0
    gravity: float = 9.81
    theta: float = 0.3
    pivot_height: float = 1.0
    box_side: float = 0.2

    def __post_init__(self) -> None:
        for name in ("mass", "length", "gravity", "pivot_height", "box_side"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive and finite, got {value!r}")
        if not (math.isfinite(self.theta) and 0.0 < self.theta < math.pi / 2):
            raise ValueError(f"theta must be in (0, pi/2), got {self.theta!r}")

    @property
    def weight(self) -> float:
        """Weight ``m*g`` [N]."""
        return self.mass * self.gravity

    @property
    def com_distance(self) -> float:
        """Pivot-to-centre-of-mass distance ``l/2`` [m]."""
        return 0.5 * self.length

    @property
    def pivot(self) -> tuple[float, float, float]:
        """World position of the hinge."""
        return (0.0, 0.0, self.pivot_height)

    @property
    def inverted_angle(self) -> float:
        """Joint angle ``q`` of the inverted hold (``pi - theta``)."""
        return math.pi - self.theta

    @property
    def hold_torque(self) -> float:
        """Actuator torque about +y holding the inverted pendulum [N*m]."""
        return self.weight * self.com_distance * math.sin(self.theta)

    @property
    def inverted_axial_load(self) -> float:
        """Proximal axial load of the inverted hold, compression negative [N]."""
        return -self.weight * math.cos(self.theta)

    @property
    def box_rest_height(self) -> float:
        """Body-origin height of the resting body on the ground plane [m]."""
        return 0.5 * self.box_side


#: The one shared parameter set used by every engine row.
STANDARD = PendulumFixture()
