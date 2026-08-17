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
    "ShotTruncatedError",
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


class ShotTruncatedError(SolverError):
    """The integration window ended before the head came out of the sand.

    This is a *settings* failure, and it is raised here so that it reads
    as one.  Before it existed the window silently ended mid-strike and
    the complaint surfaced several layers away, as
    :func:`~bunkershot3d.metrics.divot.divot_metrics` refusing to locate
    an exit crossing -- an exception from a metrics function for a
    problem the caller could only fix in ``ShotSettings`` (issue #8700).

    Attributes:
        result: The partial :class:`~bunkershot3d.solvers.shot.ShotResult`,
            so a caller can see how far the march actually got rather
            than re-running it to find out.
        max_time_s: The window that ran out.
        time_reached_s: The last sample time in the partial trace.
    """

    def __init__(
        self,
        message: str,
        *,
        result: object,
        settings: object,
    ) -> None:
        super().__init__(message)
        self.result = result
        self.max_time_s = float(getattr(settings, "max_time_s", float("nan")))
        times = getattr(result, "times_s", None)
        self.time_reached_s = (
            float(times[-1]) if times is not None and len(times) else 0.0
        )


class CalibrationError(SolverError, ValueError):
    """A calibration input is unusable or a fitted constant is missing."""
