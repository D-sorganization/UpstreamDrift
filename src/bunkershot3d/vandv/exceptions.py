"""Exceptions for the V&V suite (issue #8616).

Every one of these is raised by a plain ``raise``, never an ``assert``:
``python -O`` strips assertions and ``DBC_LEVEL=off`` disables contracts,
and a V&V guard that evaporates under an optimisation flag is worse than
no guard at all.

All of them descend from :class:`~bunkershot3d.exceptions.BunkerShot3DValueError`
and therefore from :class:`ValueError`, so callers written before this
package existed keep working.
"""

from __future__ import annotations

from ..exceptions import BunkerShot3DValueError

__all__ = [
    "ConservationClassError",
    "NoReferenceDataError",
    "SolutionVerificationError",
    "VandVError",
    "VerificationError",
]


class VandVError(BunkerShot3DValueError):
    """Base class for every verification and validation failure."""


class VerificationError(VandVError):
    """A code-verification case could not be formed, or was ill-posed.

    Raised for a malformed refinement series, a degenerate verification
    configuration, or a case whose reference quantity is so close to zero
    that the test would pass vacuously.
    """


class SolutionVerificationError(VandVError):
    """A grid-convergence study could not be formed.

    Raised for refinement ratios at or below one, for fewer grids than the
    requested estimator needs, or for an apparent-order iteration that
    will not converge.
    """


class ConservationClassError(VandVError):
    """A conservation residual was tested in the wrong way.

    The research digest is explicit that the two conservation classes need
    two different tests: round-off quantities (mass, linear and angular
    momentum) get a fixed absolute tolerance and **no** order test, while
    truncation quantities (energy under a non-symplectic scheme) get an
    order test on the residual, because the residual *should* scale as
    ``dt^p``.  Running an order test on a round-off residual measures the
    floating-point unit, not the model.
    """


class NoReferenceDataError(VandVError):
    """Validation was attempted against data that does not exist.

    The literature search behind issue #8616 found no published value
    anywhere for ball launch angle, speed or spin from a splash shot, for
    clubhead deceleration in sand, for the energy split, for ejecta mass,
    or for the coefficient of restitution through a sand layer.  This is
    the exception that stops the suite inventing one.
    """
