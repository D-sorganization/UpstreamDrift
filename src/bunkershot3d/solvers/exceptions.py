"""Solver exception hierarchy (issue #8611, epic #8607).

Every check in this package that guards a *result* -- as opposed to a
convenience -- raises.  ``assert`` is never used: ``python -O`` strips
assertions, and a validity envelope that silently disappears under an
optimisation flag is worse than no envelope at all.
"""

from __future__ import annotations

__all__ = [
    "CalibrationError",
    "OutOfEnvelopeError",
    "SolverError",
    "SolverInputError",
]


from ..exceptions import BunkerShot3DError


class SolverError(BunkerShot3DError):
    """Base class for every failure raised by :mod:`bunkershot3d.solvers`."""


class SolverInputError(SolverError, ValueError):
    """A solver was handed an intrusion state it cannot interpret."""


class OutOfEnvelopeError(SolverError):
    """The query lies outside the solver's stated validity envelope.

    This is a *refusal*, not a numerical failure.  ADR-0032 requires that
    a solver used outside its calibrated envelope say so rather than
    return a plausible number, and the research addendum makes that the
    single most important feature of the F0 tier: at 25 m/s a bunker shot
    sits roughly 60x outside 3D-RFT's stated Froude limit and ~20x beyond
    any published validation.

    Attributes:
        verdict: The :class:`~bunkershot3d.solvers.envelope.ValidityVerdict`
            that triggered the refusal, so a caller that deliberately wants
            the extrapolated number can inspect the groups and re-run with
            a permissive :class:`~bunkershot3d.solvers.envelope.RefusalPolicy`.
    """

    def __init__(self, message: str, *, verdict: object | None = None) -> None:
        super().__init__(message)
        self.verdict = verdict


class CalibrationError(SolverError, ValueError):
    """A calibration input is unusable or a fitted constant is missing."""
