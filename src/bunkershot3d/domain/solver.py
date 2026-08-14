"""Solver output settings (issue #8608, ADR-0032 decision 1)."""

from __future__ import annotations

from dataclasses import dataclass

from ..units import hz_to_period_s
from ._validate import require_positive, require_positive_int

__all__ = ["SolverSettings"]


@dataclass(frozen=True, slots=True)
class SolverSettings:
    """How often a run reports, and how much of it it reports.

    Deliberately *not* the integration timestep. Conflating the output sampling
    rate with the integrator step is defect B30: the Chrono backend used
    ``1 / output_rate_hz`` as ``dt`` and integrated at ~11 900x the Rayleigh
    limit. The timestep is derived from stability criteria by
    :mod:`bunkershot3d.backends.stability`; the sampling rate is a reporting
    choice and lives here.

    Attributes:
        output_rate_hz: Result sampling rate.
        downsample_grains: Keep every n-th grain when writing grain state.
    """

    output_rate_hz: float
    downsample_grains: int = 1

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            DomainInvariantError: The rate is not positive and finite, or the
                downsampling stride is not a positive integer.
        """
        object.__setattr__(
            self,
            "output_rate_hz",
            require_positive(self.output_rate_hz, "output_rate_hz"),
        )
        object.__setattr__(
            self,
            "downsample_grains",
            require_positive_int(self.downsample_grains, "downsample_grains"),
        )

    @property
    def output_period_s(self) -> float:
        """Interval between reported samples."""
        return hz_to_period_s(self.output_rate_hz)
