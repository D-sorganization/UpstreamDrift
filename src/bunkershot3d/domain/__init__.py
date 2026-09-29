"""Narrow domain value objects for BunkerShot3D (issue #8608, ADR-0032).

ADR-0032 decision 1: ``BunkerShotConfig``'s flat delegating accessors are
replaced by narrow value objects passed directly to the code that needs them.
That satisfies the Law of Demeter by *reducing coupling* rather than by adding
forwarding methods -- a forwarding accessor keeps every consumer depending on
the root config, which is the coupling the rule exists to prevent.

Each object is frozen, validates its own invariants on construction, and names
every physical quantity with its unit (see :mod:`bunkershot3d.units`).

What lives where, so the packages do not grow duplicates of each other:

* :class:`DomainBox` -- the *numerical* simulation container. The *physical*
  sand patch is :class:`~bunkershot3d.sand.bed.BunkerBedGeometry`.
* :class:`GrainPopulation`, :class:`ContactMaterial` -- a DEM discretisation.
  What the sand *is* is :class:`~bunkershot3d.sand.state.SandState`.
* :class:`SwingCondition` -- the kinematic delivery. The geometric delivery is
  :class:`~bunkershot3d.geometry.delivery.DeliveryCondition`, which this
  composes rather than restates.
* :class:`~bunkershot3d.geometry.wedge.WedgeGeometry` -- the design vector,
  which stays in :mod:`bunkershot3d.geometry` where it was built (#8609).
"""

from __future__ import annotations

from .box import BoundaryCondition, DomainBox
from .grains import ContactMaterial, GrainPopulation
from .solver import SolverSettings
from .swing import (
    DRFT_INERTIAL_CROSSOVER_MPS,
    MAX_PLAUSIBLE_CLUBHEAD_SPEED_MPS,
    SwingCondition,
    TrajectorySource,
)

__all__ = [
    "DRFT_INERTIAL_CROSSOVER_MPS",
    "MAX_PLAUSIBLE_CLUBHEAD_SPEED_MPS",
    "BoundaryCondition",
    "ContactMaterial",
    "DomainBox",
    "GrainPopulation",
    "SolverSettings",
    "SwingCondition",
    "TrajectorySource",
]
