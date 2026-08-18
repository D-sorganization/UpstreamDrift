"""Exceptions for the BunkerShot3D sand state model (issue #8610).

All of these derive from
:class:`~bunkershot3d.exceptions.BunkerShot3DValueError` (issue #8608), and so
from both the package root :class:`~bunkershot3d.exceptions.BunkerShot3DError`
and the standard :class:`ValueError`. A caller who only cares that a
configuration was rejected can catch either of those; a caller who needs to
distinguish *why* catches the specific subclass.

These are raised, never asserted: ``python -O`` strips ``assert`` statements,
and a feasibility guard that disappears under optimisation is not a guard.
"""

from __future__ import annotations

from ..exceptions import BunkerShot3DValueError

__all__ = [
    "BedGeometryError",
    "InfeasibleBedError",
    "MoistureRegimeError",
    "PackingStateError",
    "ParticleSizeDistributionError",
    "ProvenanceError",
    "SandModelError",
]


class SandModelError(BunkerShot3DValueError):
    """Base class for every sand-model rejection."""


class ParticleSizeDistributionError(SandModelError):
    """A sieve analysis is malformed or a percentile is unresolvable."""


class PackingStateError(SandModelError):
    """A void ratio / density / relative-density combination is unphysical."""


class MoistureRegimeError(SandModelError):
    """A moisture regime was misdeclared, or a suction term is non-physical."""


class BedGeometryError(SandModelError):
    """A bunker bed has non-positive extents or an impossible slope."""


class InfeasibleBedError(SandModelError):
    """A grain population cannot physically fill the requested bed.

    This is the guard for defect B29: the canonical configuration asked for
    50,000 grains of d = 0.4 mm in a 0.4 x 0.3 x 0.1 m domain, a solid volume
    fraction of 1.4e-4 and a settled bed 0.023 mm deep.
    """


class ProvenanceError(SandModelError):
    """A sand state omits provenance for an honesty-critical property."""
