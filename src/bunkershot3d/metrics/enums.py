"""Enumerations shared across the designer-metrics package (issue #8614).

Kept in one leaf module so :mod:`bunkershot3d.metrics.trace` and the metric
modules that consume it can both name a verdict without importing each other.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["DigSkidVerdict", "WrenchReference"]


class WrenchReference(str, Enum):
    """Point a recorded contact moment is taken about.

    Result schema v2 stores ``/wrench/torque`` without recording the point, so
    the caller must say. The merged backend work (#8612,
    :func:`bunkershot3d.backends.mpm.contact.contact_wrench_on_body`) reports it
    about the body centre of mass, which is the default everywhere here.
    """

    CENTRE_OF_MASS = "centre_of_mass"
    HEAD_ORIGIN = "head_origin"


class DigSkidVerdict(str, Enum):
    """Which of the two failure modes a strike is heading toward.

    Acushnet's adjustable-bounce patent US11766593B1 names both sides: too
    little effective bounce and the head digs, losing clubhead speed; too much
    and the leading edge reaches the ball. The verdict is therefore two-sided
    with a deliberate band between, not a threshold on a single scalar.
    """

    DIG = "dig"
    SKID = "skid"
    MARGINAL = "marginal"
